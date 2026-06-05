"""Dataset builders, splits, and target normalization helpers.

Adapted from original notebook data preparation cells. Codex and ChatGPT
helped organize the routines into a shared file.
"""

from __future__ import annotations

import numpy as np

from common import scale_V_to_unit
from models import LipkinModel, PairingModel


def make_lipkin_data(
    N=20,
    epsilon=1.0,
    use_symmetric_sector=True,
    Vmin=0.0,
    Vmax=2.0,
    n_train=20,
    n_test=100,
    k_levels=3,
):
    """Build Lipkin train/test grids from scalar settings and return arrays plus metadata in a dict.

    This was moved from repeated notebook cells into a shared helper during the
    Codex and ChatGPT organization pass.
    """
    lipkin = LipkinModel(
        N=N,
        epsilon=epsilon,
        use_symmetric_sector=use_symmetric_sector,
    )

    V_train = np.linspace(Vmin, Vmax, n_train)
    V_test = np.linspace(Vmin, Vmax, n_test)

    E_train = lipkin.spectrum(V_train, k=k_levels)
    E_test = lipkin.spectrum(V_test, k=k_levels)

    X_train = scale_V_to_unit(V_train, Vmin, Vmax).reshape(-1, 1)
    X_test = scale_V_to_unit(V_test, Vmin, Vmax).reshape(-1, 1)

    return {
        "lipkin": lipkin,
        "V_train": V_train,
        "V_test": V_test,
        "E_train": E_train,
        "E_test": E_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": E_train,
        "y_test": E_test,
        "Vmin": Vmin,
        "Vmax": Vmax,
        "k_levels": k_levels,
    }



def make_pairing_data(
    pnum=10,
    hnum=10,
    delta=1.0,
    gmin=0.0,
    gmax=2.0,
    n_train=40,
    n_test=200,
    k_levels=3,
):
    """Build pairing train/test grids from scalar settings and return arrays plus metadata in a dict.

    This mirrors the Lipkin data helper and keeps pairing result notebooks from
    rebuilding the same setup logic in several places.
    """
    pairing = PairingModel(pnum=pnum, hnum=hnum, delta=delta)

    g_train = np.linspace(gmin, gmax, n_train)
    g_test = np.linspace(gmin, gmax, n_test)

    E_train = pairing.spectrum(g_train, k=k_levels)
    E_test = pairing.spectrum(g_test, k=k_levels)

    X_train = scale_V_to_unit(g_train, gmin, gmax).reshape(-1, 1)
    X_test = scale_V_to_unit(g_test, gmin, gmax).reshape(-1, 1)

    return {
        "pairing": pairing,
        "g_train": g_train,
        "g_test": g_test,
        "E_train": E_train,
        "E_test": E_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": E_train,
        "y_test": E_test,
        "gmin": gmin,
        "gmax": gmax,
        "k_levels": k_levels,
    }


def make_train_validation_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.25):
    """Split arrays into train and validation subsets using a float validation fraction and return arrays plus indices."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_total = X.shape[0]
    n_val = max(1, int(round(val_fraction * n_total)))

    val_idx = np.linspace(0, n_total - 1, n_val + 2, dtype=int)[1:-1]
    val_idx = np.unique(val_idx)

    train_mask = np.ones(n_total, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], train_idx, val_idx
