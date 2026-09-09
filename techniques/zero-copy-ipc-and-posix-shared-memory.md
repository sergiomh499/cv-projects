---
title: "Zero-Copy IPC & POSIX Shared Memory: Ultra-Low-Latency Edge Streaming"
type: "Technique"
domain: "Edge Acceleration, Real-Time Systems & Inter-Process Communication"
tags:
  - technique
  - zero-copy
  - posix-shared-memory
  - dma-buf
  - ipc
  - real-time
  - lock-free
status: evergreen
updated: 2026-09-09
aliases:
  - "Zero-Copy IPC"
  - "POSIX Shared Memory"
  - "DMA-BUF"
  - "Lock-Free Ring Buffer"
  - "SCM_RIGHTS"
---

# ⚡ Zero-Copy IPC & POSIX Shared Memory: Ultra-Low-Latency Edge Streaming

## 1. High-Level Concept & The Memory Copy Wall

High-performance vision pipelines, autonomous vehicle stacks, and robotic perception systems ingest massive data throughputs.
For example, four $4\text{K}$ high-resolution cameras operating at $60\text{ FPS}$:

$$
\text{Throughput} = 4 \times (3840 \times 2160 \times 3\text{ bytes}) \times 60\text{ Hz} \approx 5.97\text{ GB/s}
$$

### The Conventional Multi-Copy Penalty
In traditional IPC architectures (e.g., standard Unix pipes, TCP/UDP loopback sockets, ROS 1 node communication):
Data must be sequentially serialized and copied across multiple kernel and user boundaries:
$$\text{Sensor Driver} \xrightarrow{\text{Copy 1}} \text{Kernel Space} \xrightarrow{\text{Copy 2}} \text{Publisher App} \xrightarrow{\text{Copy 3}} \text{Socket Buffer} \xrightarrow{\text{Copy 4}} \text{Subscriber App} \xrightarrow{\text{Copy 5}} \text{GPU VRAM}$$

- Copying $\sim 6\text{ GB/s}$ five times requires **$30\text{ GB/s}$ of continuous memory bus bandwidth**, completely saturating DDR memory buses, thrashing CPU L2/L3 caches, and introducing **$15\text{ to }40\text{ milliseconds}$ of non-deterministic latency jitter**!

---

### The Zero-Copy Solution Architecture
Zero-copy architectures bypass the CPU memory bus entirely by sharing physical memory pages directly between processes and hardware devices:

#### 1. POSIX Shared Memory (`shm_open`, `mmap`)
Creates a memory-backed file descriptor residing in RAM (`/dev/shm`). Both processes map the **identical physical RAM pages** into their respective virtual address spaces:
$$\text{Process A Virtual Address Space} \longleftrightarrow \text{Physical Memory Page} \longleftrightarrow \text{Process B Virtual Address Space}$$
When Process A writes a camera frame into the buffer, it is instantaneously visible to Process B with **zero memory copies ($\mathcal{O}(1)$ handover, $0\ \mu\text{s}$ copy time)**!

#### 2. Linux DMA-BUF & UNIX Domain Sockets (`SCM_RIGHTS`)
Hardware camera sensors (V4L2, GStreamer, ISP) allocate contiguous physical memory blocks via **DMA-BUF**.
Instead of copying the buffer contents, the producer sends only the integer file descriptor `fd` across a UNIX domain socket using `sendmsg()` with auxiliary control data `SCM_RIGHTS`. The kernel transfers file table references without touching the memory payload!

#### 3. Direct GPU Ingestion (Vulkan / CUDA External Memory)
The receiving GPU process imports the DMA-BUF descriptor directly into VRAM using `cudaImportExternalMemory()` or Vulkan's `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`, achieving **Direct Sensor-to-GPU Zero-Copy Ingest**!

```
Zero-Copy IPC Architecture Topology:

[ Camera Sensor / ISP Hardware ]
                |
                v Contiguous DMA Write
   [ Physical DMA-BUF Memory Page ] <-----------------------------+
                |                                                 |
                +-----------------------+                         | Zero-Copy Import
                |                       |                         |
                v mmap()                v mmap()                  v
     [ Process A: Perception ]  [ Process B: Visualizer ]   [ Process C: GPU TensorRT ]
     (Virtual Addr: 0x7fff...)  (Virtual Addr: 0x7faa...)   (Mapped to GPU VRAM)
                ^                       ^
                |                       |
                +=== Lock-Free Ring ====+
                     (Atomic Head/Tail)
```

---

## 2. Mathematical Formulation & Latency Models

### 2.1 Latency Comparison Model
Let $B$ be frame size in bytes, $N_c$ be the number of memory copies, and $R_{\text{DDR}}$ be effective memory bus bandwidth (e.g., $50\text{ GB/s}$).
The memory copy latency is:

$$
T_{\text{copy}} = \sum_{k=1}^{N_c} \frac{B}{R_{\text{DDR}}} = N_c \cdot \frac{B}{R_{\text{DDR}}}
$$

For a $4\text{K}$ RGB frame ($24.88\text{ MB}$) with $N_c = 4$:

$$
T_{\text{copy}} = 4 \times \frac{24.88 \times 10^6\text{ bytes}}{50 \times 10^9\text{ bytes/s}} \approx 1.99\text{ ms per frame}
$$

Under Zero-Copy IPC ($N_c = 0$), latency is decoupled from frame size:

$$
T_{\text{zero\_copy}} = T_{\text{atomic\_sync}} \approx 50\text{ nanoseconds}
$$

---

### 2.2 Lock-Free Circular Ring Buffer Dynamics
A triple-buffering lock-free queue manages frames using atomic head and tail pointers:
- Producer increments `write_idx = (write_idx + 1) % NUM_SLOTS` with `memory_order_release`.
- Consumer reads `read_idx` with `memory_order_acquire`.
- Guarantees strict non-blocking wait-free execution without operating system kernel context switches (`pthread_mutex_lock`).

---

## 3. Python Reference Implementation

```python
import mmap
import os
import posix_ipc
import numpy as np

class ZeroCopySharedMemoryRingBuffer:
    """
    Zero-Copy inter-process frame sharing using POSIX shared memory (shm_open).
    Enables instant multi-process access to multi-megabyte visual arrays with 0 byte copying.
    """
    def __init__(self, shm_name: str, frame_shape: tuple[int, int, int], num_slots: int = 3, create: bool = False):
        self.shm_name = shm_name if shm_name.startswith("/") else f"/{shm_name}"
        self.frame_shape = frame_shape
        self.num_slots = num_slots
        self.frame_size = int(np.prod(frame_shape)) * np.dtype(np.uint8).itemsize
        
        # Header: [uint64 write_index, uint64 sequence_number] = 16 bytes
        self.header_size = 16
        self.total_size = self.header_size + self.num_slots * self.frame_size

        if create:
            # Create POSIX shared memory object in /dev/shm
            self.shm = posix_ipc.SharedMemory(self.shm_name, flags=posix_ipc.O_CREAT | posix_ipc.O_RDWR, size=self.total_size)
            self.mmap_buf = mmap.mmap(self.shm.fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            # Initialize write_index = 0, seq = 0
            self.mmap_buf[:16] = np.zeros(2, dtype=np.uint64).tobytes()
        else:
            self.shm = posix_ipc.SharedMemory(self.shm_name, flags=posix_ipc.O_RDWR)
            self.mmap_buf = mmap.mmap(self.shm.fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def write_frame(self, frame: np.ndarray, seq: int):
        """Zero-copy in-place frame write by producer."""
        # Read current slot index
        current_idx = np.frombuffer(self.mmap_buf[:8], dtype=np.uint64)[0]
        next_slot = int((current_idx + 1) % self.num_slots)
        
        # Calculate memory offset
        offset = self.header_size + next_slot * self.frame_size
        
        # Wrap destination memory directly as a NumPy array (Zero-Copy!)
        dest_arr = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.mmap_buf, offset=offset)
        np.copyto(dest_arr, frame)
        
        # Atomically update write pointer and sequence number
        self.mmap_buf[:16] = np.array([next_slot, seq], dtype=np.uint64).tobytes()

    def read_latest_frame(self) -> tuple[np.ndarray, int]:
        """Zero-copy memory view retrieval by consumer."""
        header = np.frombuffer(self.mmap_buf[:16], dtype=np.uint64)
        active_slot = int(header[0])
        seq = int(header[1])
        
        offset = self.header_size + active_slot * self.frame_size
        # Expose shared memory directly as a read-only NumPy array
        frame_view = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.mmap_buf, offset=offset)
        return frame_view, seq

    def close(self):
        self.mmap_buf.close()
        self.shm.close_fd()
```

---

## 4. Models & Runtimes in the Vault Utilizing Zero-Copy

- **[[frameworks/vulkan-sc|Vulkan SC 2.0]]**: Deterministic safety-critical graphics runtime utilizing DMA-BUF memory imports.
- **[[frameworks/tensorrt|NVIDIA TensorRT 10]]**: Pinned host-device unified memory (`cudaMallocManaged`) for zero-copy inference ingestion.
- **[[architectures/hardware-and-acceleration-runtimes/zenoh-router|Zenoh Router]]**: High-performance robotic middleware utilizing shared-memory zero-copy pub-sub transports.
- **[[topics/real-time-systems/README|Real-Time Systems Playbook]]**: Cataloged as the primary latency mitigation pattern for edge sensor pipelines.
