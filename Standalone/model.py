"""Independent spherical finite-volume Fisher-KPP solver for D/k estimation.

The original PDEgrid stores its field and accumulated increments in float32.
This implementation preserves that storage precision and update ordering,
while depending only on NumPy. Validate numerically against original
Model3Dand1D.RunModel1D before asserting scientific equivalence.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ModelParams:
    d: float
    k: float
    steps: float
    sideLen: int = 201
    dx: float = 1.0
    dt: float = 1.0


@dataclass
class ModelResult:
    params: ModelParams
    finalState: np.ndarray
    intermediateStates: np.ndarray | None = None
    saveFreq: int | None = None


def step(field, d, k, dx=1.0, dt=1.0, fraction=1.0):
    """One simultaneous explicit diffusion/reaction update, no-flux boundaries."""
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 1 or len(field) < 2:
        raise ValueError("Radial field must have at least two cells")
    n = len(field)
    increments = np.zeros(n, dtype=np.float32)
    scaling = (d * fraction * dt) / dx
    r = dx * 0.5
    increments[0] += (scaling / ((1.0 / 3.0) * r**3)) * (r**2 * (float(field[1]) - float(field[0])))
    for i in range(1, n - 1):
        rp = (i + 0.5) * dx
        rm = (i - 0.5) * dx
        flux = rp**2 * (float(field[i + 1]) - float(field[i])) - rm**2 * (float(field[i]) - float(field[i - 1]))
        increments[i] += (scaling / ((1.0 / 3.0) * (rp**3 - rm**3))) * flux
    rp = (n - 0.5) * dx
    rm = (n - 1.5) * dx
    increments[-1] += (scaling / ((1.0 / 3.0) * (rp**3 - rm**3))) * (-rm**2 * (float(field[-1]) - float(field[-2])))
    # Match the original: reaction reads the old field, not the diffused field.
    for i in range(n):
        value = float(field[i])
        increments[i] += value * k * fraction * dt * (1.0 - value)
    # Match float32 storage and sequential accumulation.
    for i in range(n):
        field[i] = np.float32(field[i] + increments[i])
    return field



# Optional acceleration. Keep the pure NumPy implementation as a fallback.
try:
    from numba import njit
except ImportError:
    njit = None


def _run_model_compiled_impl(d, k, steps, n, dx, dt, save_freq):
    """Compile the complete time loop, not merely a single update."""
    field = np.zeros(n, dtype=np.float32)
    field[0] = np.float32(1e-6 / (dx ** 3))
    whole = int(steps)
    fractional = steps - whole
    count = (whole - 1) // save_freq + 1 if save_freq > 0 and whole > 0 else 0
    snapshots = np.empty((count, n), dtype=np.float32)
    increments = np.zeros(n, dtype=np.float32)
    saved = 0
    for t in range(whole):
        if save_freq > 0 and t % save_freq == 0:
            snapshots[saved, :] = field
            saved += 1
        _step_compiled_impl(field, increments, d, k, dx, dt, 1.0)
    if fractional > 0:
        _step_compiled_impl(field, increments, d, k, dx, dt, fractional)
    return field, snapshots


def _step_compiled_impl(field, increments, d, k, dx, dt, fraction):
    n = len(field)
    scaling = (d * fraction * dt) / dx
    r = dx * 0.5
    increments[0] += (scaling / ((1.0 / 3.0) * r ** 3)) * (
        r ** 2 * (float(field[1]) - float(field[0])))
    for i in range(1, n - 1):
        rp = (i + 0.5) * dx
        rm = (i - 0.5) * dx
        flux = rp ** 2 * (float(field[i + 1]) - float(field[i])) - rm ** 2 * (
            float(field[i]) - float(field[i - 1]))
        increments[i] += (scaling / ((1.0 / 3.0) * (rp ** 3 - rm ** 3))) * flux
    rp = (n - 0.5) * dx
    rm = (n - 1.5) * dx
    increments[-1] += (scaling / ((1.0 / 3.0) * (rp ** 3 - rm ** 3))) * (
        -rm ** 2 * (float(field[-1]) - float(field[-2])))
    for i in range(n):
        value = float(field[i])
        increments[i] += value * k * fraction * dt * (1.0 - value)
    for i in range(n):
        field[i] = np.float32(field[i] + increments[i])
        increments[i] = np.float32(0.0)


if njit is not None:
    _step_compiled_impl = njit(cache=True)(_step_compiled_impl)
    _run_model_compiled_impl = njit(cache=True)(_run_model_compiled_impl)


def run_model(params, save_freq=None, accelerated=True):
    """Run original initialization, checkpoint timing, and fractional final step."""
    if params.sideLen < 3 or params.dx <= 0 or params.dt <= 0 or params.steps < 0:
        raise ValueError("Invalid model geometry or time")
    if save_freq is not None and save_freq < 1:
        raise ValueError("save_freq must be positive")
    if accelerated and njit is not None:
        final, snapshots = _run_model_compiled_impl(
            params.d, params.k, params.steps, params.sideLen // 2 + 1,
            params.dx, params.dt, save_freq if save_freq is not None else 0)
        return ModelResult(params, final, snapshots if save_freq is not None else None, save_freq)
    field = np.zeros(params.sideLen // 2 + 1, dtype=np.float32)
    field[0] = np.float32(1e-6 / (params.dx**3))
    fractional, whole = math.modf(params.steps)
    snapshots = []
    for t in range(int(whole)):
        if save_freq is not None and t % save_freq == 0:
            snapshots.append(field.copy())
        step(field, params.d, params.k, params.dx, params.dt)
    if fractional > 0:
        step(field, params.d, params.k, params.dx, params.dt, fractional)
    return ModelResult(params, field, np.asarray(snapshots) if save_freq is not None else None, save_freq)


def shell_volumes(n):
    """Weights for the radial least-squares objective."""
    edges = np.arange(n, dtype=float)
    upper = edges + 0.5
    lower = np.maximum(edges - 0.5, 0.0)
    return (4.0 / 3.0) * np.pi * (upper**3 - lower**3)


def grid_search(target, grid, weights=None):
    """Return (grid entry, snapshot index, weighted squared error)."""
    target = np.asarray(target)
    weights = shell_volumes(len(target)) if weights is None else np.asarray(weights)
    best = (None, None, float("inf"))
    for entry in grid:
        for index, state in enumerate(entry.intermediateStates):
            error = float(np.sum((target - state)**2 * weights))
            if error < best[2]:
                best = (entry, index, error)
    if best[0] is None:
        raise ValueError("Grid contains no snapshots")
    return best
