#!/usr/bin/env python3
"""
Cookbook 15: Real-Time RMS & EDF Perception Pipeline Task Scheduler.

Implements hard real-time scheduling for multi-rate vision pipelines:
1. Rate Monotonic Scheduling (RMS) - Static priority assignment where priority is inversely proportional to period T.
2. Earliest Deadline First (EDF) - Dynamic priority assignment where task closest to deadline executes first.
3. Schedulability analysis via Liu & Layland utilization bounds:
   - RMS bound: U = sum(C_i / T_i) <= n * (2^(1/n) - 1)
   - EDF bound: U = sum(C_i / T_i) <= 1.0
4. Discrete-event simulator modeling preemption, task releases, execution, and deadline miss detection.

References:
    Liu & Layland, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment", JACM 1973.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TaskSpec:
    """Specification of a periodic real-time perception task."""
    name: str
    wcet: int      # Worst-Case Execution Time (C_i) in milliseconds
    period: int    # Period (T_i) in milliseconds
    deadline: int  # Relative deadline (D_i) in milliseconds (implicit deadline: D_i = T_i)


@dataclass
class TaskJob:
    """Instance of a released job."""
    task_name: str
    release_time: int
    remaining_time: int
    absolute_deadline: int
    period: int


class RealTimePerceptionScheduler:
    """Discrete-time scheduler for periodic perception tasks."""

    def __init__(self, tasks: list[TaskSpec]):
        self.tasks = sorted(tasks, key=lambda t: t.period)

    def compute_utilization(self) -> float:
        """Total processor utilization U = sum(C_i / T_i)."""
        return sum(t.wcet / t.period for t in self.tasks)

    def check_rms_schedulability(self) -> tuple[bool, float, float]:
        """Checks Liu & Layland RMS utilization bound: U <= n*(2^(1/n) - 1)."""
        n = len(self.tasks)
        u = self.compute_utilization()
        bound = n * (2.0 ** (1.0 / n) - 1.0)
        return (u <= bound), u, bound

    def check_edf_schedulability(self) -> tuple[bool, float]:
        """Checks EDF utilization bound: U <= 1.0."""
        u = self.compute_utilization()
        return (u <= 1.0), u

    def simulate(
        self,
        duration_ms: int,
        policy: Literal["RMS", "EDF"] = "RMS",
    ) -> dict:
        """Simulates task execution over duration_ms with 1ms granularity."""
        timeline: list[str | None] = []  # Running task at each millisecond
        active_jobs: list[TaskJob] = []
        deadline_misses: list[tuple[str, int]] = []
        preemptions: int = 0
        last_running_task: str | None = None

        for t in range(duration_ms):
            # 1. Release periodic jobs
            for task in self.tasks:
                if t % task.period == 0:
                    job = TaskJob(
                        task_name=task.name,
                        release_time=t,
                        remaining_time=task.wcet,
                        absolute_deadline=t + task.deadline,
                        period=task.period,
                    )
                    active_jobs.append(job)

            # 2. Check for deadline misses
            missed = [j for j in active_jobs if j.absolute_deadline <= t and j.remaining_time > 0]
            for m in missed:
                deadline_misses.append((m.task_name, t))
            active_jobs = [j for j in active_jobs if j.absolute_deadline > t or j.remaining_time == 0]

            # 3. Filter completed jobs
            active_jobs = [j for j in active_jobs if j.remaining_time > 0]

            # 4. Priority selection
            if not active_jobs:
                current_task = None
            else:
                if policy == "RMS":
                    # Static priority: shortest period has highest priority
                    active_jobs.sort(key=lambda j: (j.period, j.release_time))
                elif policy == "EDF":
                    # Dynamic priority: earliest absolute deadline has highest priority
                    active_jobs.sort(key=lambda j: (j.absolute_deadline, j.release_time))
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                selected_job = active_jobs[0]
                current_task = selected_job.task_name
                selected_job.remaining_time -= 1

            # Detect preemptions
            if (
                last_running_task is not None
                and current_task is not None
                and last_running_task != current_task
            ):
                # Check if the previous task had remaining work
                prev_active = [j for j in active_jobs if j.task_name == last_running_task and j.remaining_time > 0]
                if prev_active:
                    preemptions += 1

            timeline.append(current_task)
            last_running_task = current_task

        return {
            "policy": policy,
            "duration_ms": duration_ms,
            "utilization": self.compute_utilization(),
            "preemptions": preemptions,
            "deadline_misses": deadline_misses,
            "timeline": timeline,
        }


def run_demo():
    print("=== Hard Real-Time Perception Pipeline Task Scheduler Demo ===")

    # Typical Autonomous Vehicle Perception Task Set
    tasks = [
        TaskSpec(name="IMU_EKF_Fusion", wcet=1, period=5, deadline=5),     # 200 Hz, C=1ms, U=0.20
        TaskSpec(name="Camera_Ingest",  wcet=2, period=10, deadline=10),   # 100 Hz, C=2ms, U=0.20
        TaskSpec(name="CBF_Safety_QP",  wcet=3, period=20, deadline=20),   # 50 Hz,  C=3ms, U=0.15
        TaskSpec(name="YOLO_Detector",  wcet=7, period=33, deadline=33),   # 30 Hz,  C=7ms, U=0.21
    ]

    scheduler = RealTimePerceptionScheduler(tasks)
    u_total = scheduler.compute_utilization()
    print(f"Task Set Total Processor Utilization: {u_total*100:.1f}%\n")

    print("Task Set Specifications:")
    for t in tasks:
        print(f"  - {t.name:<16}: WCET={t.wcet}ms, Period={t.period}ms, Freq={1000/t.period:.0f}Hz, Util={t.wcet/t.period*100:.1f}%")

    # RMS Schedulability
    rms_ok, u, rms_bound = scheduler.check_rms_schedulability()
    print(f"\n1. RMS Schedulability: {'GUARANTEED' if rms_ok else 'NOT GUARANTEED BY BOUND'}")
    print(f"   Utilization {u*100:.1f}% vs Liu-Layland Bound {rms_bound*100:.1f}%")

    # EDF Schedulability
    edf_ok, _ = scheduler.check_edf_schedulability()
    print(f"2. EDF Schedulability: {'GUARANTEED (U <= 1.0)' if edf_ok else 'OVERUTILIZED'}")

    # Simulate 100ms
    sim_rms = scheduler.simulate(duration_ms=100, policy="RMS")
    sim_edf = scheduler.simulate(duration_ms=100, policy="EDF")

    print("\n3. Simulation Results (100 ms Horizon):")
    print(f"   [RMS] Preemptions: {sim_rms['preemptions']}, Deadline Misses: {len(sim_rms['deadline_misses'])}")
    print(f"   [EDF] Preemptions: {sim_edf['preemptions']}, Deadline Misses: {len(sim_edf['deadline_misses'])}")

    assert len(sim_rms["deadline_misses"]) == 0, "RMS experienced unexpected deadline misses!"
    assert len(sim_edf["deadline_misses"]) == 0, "EDF experienced unexpected deadline misses!"

    # Print execution slice visualization for first 40ms
    print("\n4. Execution Timeline Slice (First 40 ms):")
    timeline_str = "".join([t[0] if t else "." for t in sim_rms["timeline"][:40]])
    print(f"   Ticks: [0{'':<38}40]")
    print(f"   Tasks: [{timeline_str}]")
    print("   Legend: I=IMU, C=Camera, C=CBF, Y=YOLO, .=Idle")

    print("\n[+] Real-time perception scheduler verification PASSED.")


if __name__ == "__main__":
    run_demo()
