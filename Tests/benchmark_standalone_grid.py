"""Measure compiled standalone grid throughput without a full expensive cohort run.

Run from repository root:
    python -m Tests.benchmark_standalone_grid --grid-size 10 --steps 100000
"""
import argparse
import time
import numpy as np
from Standalone.model import ModelParams, run_model, njit


def benchmark(grid_size=10, steps=100000, save_freq=1000):
    if njit is None:
        raise RuntimeError("Install numba to benchmark the compiled solver")
    if grid_size < 1 or steps < 1 or save_freq < 1:
        raise ValueError("Require positive grid size, steps, save frequency")
    # Exclude one-time JIT compilation from steady-state throughput.
    run_model(ModelParams(1e-3, 1e-4, 2), save_freq=1)
    diffusion = np.logspace(-5, -2, grid_size)
    start = time.perf_counter()
    grid = [run_model(ModelParams(float(d), 1e-4, steps), save_freq=save_freq)
            for d in diffusion]
    elapsed = time.perf_counter() - start
    snapshots_bytes = sum(x.intermediateStates.nbytes for x in grid)
    print(f"grid_size={grid_size} steps={steps} seconds={elapsed:.3f}")
    print(f"steps_per_second={grid_size * steps / elapsed:.1f}")
    print(f"snapshot_megabytes={snapshots_bytes / 1e6:.3f}")
    # Estimate is linear extrapolation only, not a full-grid measurement.
    estimated = elapsed * (1000 / grid_size) * (1_000_000 / steps)
    print(f"estimated_1000x1e6_seconds={estimated:.1f} (extrapolated, not measured)")
    estimated_snapshot_mb = (1000 * ((1_000_000 - 1) // 1000 + 1) * 101 * 4) / 1e6
    print(f"publication_grid_snapshot_megabytes={estimated_snapshot_mb:.1f} (exact array payload, excluding Python overhead)")
    return elapsed, snapshots_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--save-freq", type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.grid_size, args.steps, args.save_freq)


if __name__ == "__main__":
    main()
