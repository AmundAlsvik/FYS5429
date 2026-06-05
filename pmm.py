"""Parametric matrix model surrogate helpers.

Mostly based on original PMM code from Morten. Codex
and ChatGPT helped organize the implementation and add small shared utilities.
"""

from __future__ import annotations

import time

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap

from common import (
    GLOBAL_SEED,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_INIT_SCALE,
    DEFAULT_CHECK_EVERY,
    DEFAULT_PATIENCE,
    DEFAULT_MIN_DELTA,
    test_error_summary,
)


class EigenvaluePMM:
    """Real symmetric affine PMM whose inputs are feature arrays and whose outputs are low eigenvalues.

    The PMM training logic is mostly from the original code by Morten. Codex
    and ChatGPT helped generalize it in a reusable class.
    """

    def __init__(
        self,
        n,
        X_train,
        y_train,
        k,
        l,
        seed=GLOBAL_SEED,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        init_scale=DEFAULT_INIT_SCALE,
        check_every=DEFAULT_CHECK_EVERY,
        patience=DEFAULT_PATIENCE,
        min_delta=DEFAULT_MIN_DELTA,
        print_every=500,
        verbose=True,
    ):
        """Store PMM dimensions, training arrays, and optimizer settings and returns None."""
        self.n = int(n)
        self.X_train = jnp.asarray(X_train, dtype=jnp.float64)
        self.y_train = jnp.asarray(y_train, dtype=jnp.float64)
        self.k = int(k)
        self.l = int(l)

        self.seed = int(seed)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.init_scale = float(init_scale)

        self.check_every = int(check_every)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.print_every = int(print_every)
        self.verbose = bool(verbose)

        self.dtype = jnp.float64

        self.best_param = None
        self.best_loss = np.inf
        self.best_step = 0
        self.train_time = None

    def get_param(self):
        """Create an initial flattened float parameter vector and return it as a NumPy array."""
        num_params = (self.l + 1) * self.n * self.n
        rng = np.random.default_rng(self.seed)
        theta = self.init_scale * rng.standard_normal(num_params)
        return theta.astype(np.float64)

    def split_params(self, theta):
        """Reshape a flattened parameter vector into symmetric PMM matrices and return a JAX array."""
        theta = jnp.asarray(theta, dtype=self.dtype)
        O = jnp.reshape(theta, (self.l + 1, self.n, self.n))
        O = (O + jnp.swapaxes(O, 1, 2)) / 2.0
        return O

    def predict(self, theta, X_data):
        """Evaluate PMM eigenvalue predictions for parameter theta and input array X_data."""
        O = self.split_params(theta)
        k_eff = min(self.k, self.n)

        def pmm_ev(x, O_):
            """Evaluate one PMM matrix at input x and return its lowest eigenvalues."""
            M = O_[0] + jnp.einsum("i,ijk->jk", x, O_[1:])
            E = jnp.linalg.eigvalsh(M)
            return E[:k_eff]

        return vmap(pmm_ev, in_axes=(0, None))(jnp.asarray(X_data, dtype=self.dtype), O)

    def loss(self, theta, X_data, y_true):
        """Compute mean squared eigenvalue loss for theta, inputs, and target array."""
        y_pred = self.predict(theta, X_data)
        return jnp.mean((y_true - y_pred) ** 2)

    def adam_step(self, g, i, m, v):
        """Apply one Adam update from gradient and moment arrays and return update plus new moments."""
        b1, b2, eps = 0.9, 0.999, 1e-8
        m_new = b1 * m + (1 - b1) * g
        v_new = b2 * v + (1 - b2) * (g * g)
        mhat = m_new / (1 - b1 ** (i + 1))
        vhat = v_new / (1 - b2 ** (i + 1))
        delta_theta = mhat / (jnp.sqrt(vhat) + eps)
        return delta_theta, m_new, v_new

    def train(self):
        """Train the PMM with Adam and return the best flattened parameter vector.

        The optimizer loop comes from the original PMM notebook, with the
        bookkeeping separated out by Codex.
        """
        theta = jnp.array(self.get_param(), dtype=self.dtype)

        train_loss = lambda th: self.loss(th, self.X_train, self.y_train)
        jloss = jit(train_loss)
        jgrad = jit(grad(train_loss))

        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)

        best_theta = theta
        best_loss = float(jloss(theta))
        best_step = 0
        no_improve = 0

        t0 = time.perf_counter()

        for i in range(self.epochs):
            g = jgrad(theta)
            delta_theta, m, v = self.adam_step(g, i, m, v)
            theta = theta - self.learning_rate * delta_theta

            if self.verbose and i % self.print_every == 0:
                loss_now = float(jloss(theta))
                print(f"Epoch {i:6d} | train loss = {loss_now:.3e}")

            if (i + 1) % self.check_every == 0:
                loss_now = float(jloss(theta))

                if loss_now < best_loss - self.min_delta:
                    best_loss = loss_now
                    best_theta = theta
                    best_step = i + 1
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    break

        t1 = time.perf_counter()

        self.best_param = np.array(best_theta, dtype=np.float64)
        self.best_loss = float(best_loss)
        self.best_step = int(best_step)
        self.train_time = float(t1 - t0)

        return self.best_param


def build_M_matrices(theta_best, n, l):
    """Convert a flattened PMM parameter vector into symmetric matrices and return an array."""
    theta_best = np.asarray(theta_best, dtype=np.float64)
    O = theta_best.reshape((l + 1, n, n))
    O = 0.5 * (O + np.swapaxes(O, 1, 2))
    return O


def build_M0_M1(theta_best, n):
    """Extract the affine one feature PMM matrices M0 and M1 from a flattened parameter vector."""
    O = build_M_matrices(theta_best, n=n, l=1)
    return O[0], O[1]


def pmm_parameter_count(n_pmm, n_features=1):
    """Compute the affine PMM parameter count from dimension and feature count."""
    return int((int(n_features) + 1) * int(n_pmm) ** 2)


def evaluate_pmm_on_grid(theta_best, n, X_grid, k, l=1):
    """Evaluate low eigenvalues of a trained PMM on an input grid and return a prediction array.

    This function replaced repeated prediction code in the original Lipkin and
    pairing result notebooks, this was a suggestion from Codex.
    """
    X_grid = np.asarray(X_grid, dtype=float)
    O = build_M_matrices(theta_best, n=n, l=l)
    matrices = O[0][None, :, :] + np.einsum("mi,ijk->mjk", X_grid, O[1:])
    return np.linalg.eigvalsh(matrices)[:, :k].astype(float)


def train_eigenvalue_pmm_on_data(
    data,
    n_pmm,
    k,
    seed,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    init_scale=DEFAULT_INIT_SCALE,
    verbose=False,
):
    """Train one PMM from result notebook data and return predictions and metrics.

    Takes a data dict with X_train, y_train, X_test and E_test, a PMM dimension,
    the number of target eigenvalues, and training settings. Returns a dict with
    the trained matrices, predictions, timing values, and test errors.
    """
    model = EigenvaluePMM(
        n=n_pmm,
        X_train=data["X_train"],
        y_train=data["y_train"],
        k=k,
        l=1,
        seed=seed,
        epochs=epochs,
        learning_rate=lr,
        init_scale=init_scale,
        verbose=verbose,
    )
    theta_best = model.train()
    M0, M1 = build_M0_M1(theta_best, n=n_pmm)
    y_pred = evaluate_pmm_on_grid(theta_best, n=n_pmm, X_grid=data["X_test"], k=k, l=1)

    return {
        "n_pmm": int(n_pmm),
        "k": int(k),
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "init_scale": float(init_scale),
        "train_loss": float(model.best_loss),
        "best_step": int(model.best_step),
        "train_time_s": float(model.train_time),
        "theta_best": np.asarray(theta_best),
        "M0": M0,
        "M1": M1,
        "pmm_pred": y_pred,
        **test_error_summary(data["E_test"], y_pred),
    }


def compressed_pmm_predictions(M0, M1, B, X_grid, k):
    """Evaluate compressed one feature PMM eigenvalue predictions on a grid.

    This function also replaced repeated prediction code, and was a suggestion from Codex.
    """
    rows = []
    for x in np.asarray(X_grid):
        x_scalar = float(np.ravel(x)[0])
        H = M0 + x_scalar * M1
        H_red = B.T @ H @ B
        rows.append(np.linalg.eigvalsh(H_red)[:k])
    return np.asarray(rows, dtype=float)
