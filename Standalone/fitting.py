"""Standalone grid-initialized Levenberg-Marquardt D/k fitting.

Independent of DkMethod's original modules. The LM iteration follows the
original LMclassNew algorithm, including bounded finite differences, diagonal
damping, first-step scaling, acceptance and stopping rules. This is a
reimplementation for parity testing, not a claim of verified equivalence.
"""
from dataclasses import dataclass, replace
import numpy as np
from .model import ModelParams, run_model, grid_search, shell_volumes


@dataclass
class FitResult:
    id: str
    params: ModelParams
    weighted_prediction: np.ndarray
    error: float
    iterations: int


def levenberg_marquardt(function, initial, observed, bounds=((-5, -2), (0, 1e6)),
                        max_iters=100, tolerance=1e-4):
    """Return best parameters, weighted prediction, error and iterations."""
    bounds = np.asarray(bounds, dtype=float)
    x = np.asarray(initial, dtype=float).copy()
    if x.shape != (2,) or np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError("Initial guess lies outside fitting bounds")
    observed = np.asarray(observed, dtype=float)
    scaler = 1.0
    accepted = False
    accepted_steps = 0
    stuck = 0
    improvement = []
    best_x = x.copy()
    best_y = None
    best_error = float("inf")
    previous_error = 0.0
    next_x = None
    next_y = None
    jac = None

    def jacobian(point, y):
        columns = []
        for j in range(2):
            delta = 1e-4 * (bounds[j, 1] - bounds[j, 0])
            trial = point.copy()
            if bounds[j, 0] < point[j] + delta < bounds[j, 1]:
                trial[j] += delta
                columns.append((function(trial) - y) / delta)
            else:
                trial[j] -= delta
                columns.append((y - function(trial)) / delta)
        return np.stack(columns, axis=1)

    for iteration in range(max_iters):
        if iteration == 0:
            y = function(x)
            jac = jacobian(x, y)
            hessian = jac.T @ jac
            best_y = y.copy()
            best_error = float(np.dot(observed - y, observed - y))
        elif accepted:
            accepted = False
            jac = jacobian(next_x, next_y)
            y = next_y
            x = next_x
            best_error = previous_error
            scaler /= 9.0
            hessian = jac.T @ jac
            stuck = 0
            improvement.append(previous_error)
        else:
            stuck += 1
            scaler *= 11.0

        while True:
            damping = hessian + scaler * np.diag(np.diag(hessian))
            delta = np.linalg.lstsq(damping.astype(float), (jac.T @ (observed - y)).astype(float), rcond=-1)[0]
            if iteration == 0 and np.max(np.abs(delta) / (bounds[:, 1] - bounds[:, 0])) > 0.1:
                scaler *= 11.0
            else:
                break

        next_x = np.clip(x + delta, bounds[:, 0], bounds[:, 1])
        next_y = function(next_x)
        previous_error = float(np.dot(observed - next_y, observed - next_y))
        if best_error - previous_error > 0:
            best_y = next_y.copy()
            best_x = next_x.copy()
            best_error = previous_error
            accepted = True
        if stuck == 25:
            break
        if len(improvement) >= 10 and improvement[-10] - improvement[-1] < tolerance:
            break
    return best_x, best_y, best_error, iteration + 1


def fit_patient(patient, grid, enhancing=0.9, nonenhancing=0.45, bounds=((-5, -2), (0, 1e6))):
    """Fit a RadialPatient using the original radial-weighted objective."""
    target = patient.density(length=101, enhancing=enhancing, nonenhancing=nonenhancing)
    weights = shell_volumes(len(target))
    start_entry, snapshot, _ = grid_search(target, grid, weights)
    initial_params = replace(start_entry.params, steps=snapshot * start_entry.saveFreq)
    observed = target * np.sqrt(weights)

    def predict(x):
        params = replace(initial_params, d=10 ** x[0], steps=x[1])
        return run_model(params).finalState * np.sqrt(weights)

    x, prediction, error, iterations = levenberg_marquardt(
        predict, (np.log10(initial_params.d), initial_params.steps), observed, bounds=bounds
    )
    params = replace(initial_params, d=10 ** x[0], steps=x[1])
    return FitResult(patient.id, params, prediction, error, iterations)


def result_row(fit):
    return {"id": fit.id, "D": fit.params.d, "k": fit.params.k,
            "t": fit.params.steps, "D/k": fit.params.d / fit.params.k}
