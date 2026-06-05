"""Shared constants, JAX setup, seeds, and numerical utilities.

Adapted from an original project notebook written by myself. Codex and
ChatGPT helped organize settings and remove duplicate functions.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

GLOBAL_SEED = 1234

DEFAULT_EPOCHS = 2500
DEFAULT_LR = 1e-2
DEFAULT_INIT_SCALE = 1e-2

DEFAULT_CHECK_EVERY = 50
DEFAULT_PATIENCE = 12
DEFAULT_MIN_DELTA = 1e-7

DEFAULT_Q_GRID = 60
DEFAULT_ERR_GRID = 200

DEFAULT_R_STEP = 4
DEFAULT_REFINE_LOCAL = True

LOSS_GOOD_MAX = 2e-2

BENCH_N = 3
BENCH_WARMUP = 1


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Set the NumPy random seed from an int and return None."""
    np.random.seed(int(seed))


def scale_V_to_unit(V, Vmin, Vmax):
    """Scale a numeric array from [Vmin, Vmax] to [-1, 1] and return an array."""
    V = np.asarray(V, dtype=float)
    if Vmax == Vmin:
        return np.zeros_like(V)
    return 2.0 * (V - Vmin) / (Vmax - Vmin) - 1.0


def compute_error_stats(y_true, y_pred):
    """Compare true and predicted numeric arrays and return max, mean, and RMS absolute errors."""
    err = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    return {
        "max_abs_error": float(np.max(err)),
        "mean_abs_error": float(np.mean(err)),
        "rms_error": float(np.sqrt(np.mean(err ** 2))),
    }


def mean_squared_error(y_true, y_pred):
    """Compute mean squared error between two numeric arrays and return a float."""
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(diff ** 2))


def compute_error_stats_with_mse(y_true, y_pred):
    """Compare true and predicted arrays and return absolute error stats plus MSE."""
    stats = compute_error_stats(y_true, y_pred)
    stats["mse"] = mean_squared_error(y_true, y_pred)
    return stats


def test_error_summary(y_true, y_pred):
    """Compare exact and predicted test arrays and return named error metrics."""
    stats = compute_error_stats(y_true, y_pred)
    return {
        "test_loss": mean_squared_error(y_true, y_pred),
        "test_mean_abs_error": stats["mean_abs_error"],
        "test_max_abs_error": stats["max_abs_error"],
        "test_rms_error": stats["rms_error"],
    }


set_seed(GLOBAL_SEED)
