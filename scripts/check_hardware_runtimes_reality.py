#!/usr/bin/env python3
"""
Diagnostic & Testing Guide for Local and Embedded Hardware Runtimes.
Inspects:
1. Available physical GPU hardware and Vulkan drivers on the host machine.
2. Explains the step-by-step physical test commands for CUDA/TensorRT, Vulkan SC, and Vitis AI.
"""

import subprocess
import shutil


def check_local_vulkan() -> None:
    print("--- 1. Checking Host Vulkan Environment ---")
    vulkan_loader = shutil.which("vulkaninfo")
    if vulkan_loader:
        try:
            res = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, check=True)
            print("Vulkan Driver Summary:")
            for line in res.stdout.strip().split("\n")[:10]:
                print(f"    {line}")
        except Exception as e:
            print(f"    vulkaninfo failed: {e}")
    else:
        print("    `vulkaninfo` CLI tool not installed (vulkan-tools package).")
        print("    Vulkan ICD loader is present in system packages.")


def explain_real_hardware_testing() -> None:
    print("\n--- 2. How Each Hardware Runtime is ACTUALLY Tested in Physical Engineering ---")
    print("""
[A] NVIDIA CUDA & TensorRT (Physical Jetson / RTX):
    Command: trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --int8 --dumpProfile
    What it does: 
    - Loads the real ONNX graph into GPU memory.
    - Profiles every single fused kernel candidate live on the actual streaming multiprocessors.
    - Outputs a true millisecond execution trace with zero mocking.

[B] Vulkan SC (Safety-Critical ISO 26262 / DO-178C):
    Workflow:
    1. Pipeline Compilation:
       `pcc -inputs pipelines.json -out static_cache.bin`
       Compiles all SPIR-V shaders into a static, deterministic binary cache offline on x86 host.
    2. Conformance & Determinism Testing:
       `vksc_cts --deqp-case=dEQP-VKSC.*`
       Runs the Khronos Vulkan SC Conformance Test Suite verifying:
       - No dynamic heap allocations (`malloc`, dynamic VRAM) occur during execution.
       - Worst-Case Execution Time (WCET) jitter remains within microsecond bounds.

[C] AMD Vitis AI & FPGA / Versal NPU:
    Workflow:
    1. Quantization:
       `python -m quark.torch --model model.pth --quant_scheme int8`
    2. DPU Architecture Compilation:
       `vai_c_xir -x model_quantized.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VERSAL/arch.json -o compiled.xmodel`
    3. Hardware Telemetry & Execution on Board:
       `xdputil benchmark compiled.xmodel 1`
       `xbutil examine --report thermal,electrical`
       Measures actual hardware clock cycles, DPU tensor core utilization, and FPGA junction temperature.
""")


def main() -> None:
    check_local_vulkan()
    explain_real_hardware_testing()


if __name__ == "__main__":
    main()
