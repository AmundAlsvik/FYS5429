"""Neural network and recurrent baseline models.

Adapted from original baseline notebook code. Codex and ChatGPT helped turn
the neural network and GRU experiments into reusable training helpers.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import GLOBAL_SEED, set_seed, compute_error_stats, compute_error_stats_with_mse
from data import make_train_validation_split


def set_torch_seed(seed: int = GLOBAL_SEED) -> None:
    """Set NumPy and PyTorch seeds from an int and return None."""
    set_seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters of a torch module and return an int."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPRegressor(nn.Module):
    """Small feedforward regressor used as the static neural network baseline."""
    def __init__(self, input_dim: int, output_dim: int, hidden_width: int, activation: str = "tanh"):
        """Create an MLP from integer dimensions, hidden width, and activation name and returns None."""
        super().__init__()

        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            act(),
            nn.Linear(hidden_width, hidden_width),
            act(),
            nn.Linear(hidden_width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a torch input tensor through the MLP and return a prediction tensor."""
        return self.net(x)


def train_single_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_width: int,
    activation: str,
    weight_decay: float,
    lr: float = 1e-3,
    epochs: int = 2000,
    batch_size: int = 16,
    seed: int = GLOBAL_SEED,
    device: str = "cpu",
):
    """Train one MLP configuration on train/validation arrays and return metrics plus the model.

    This is part of the neural network grid search skeleton organized with
    Codex and ChatGPT from the original notebook code.
    """
    set_torch_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_width=hidden_width,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")

    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    t1 = time.perf_counter()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t).cpu().numpy()

    stats = compute_error_stats(y_val, y_val_pred)

    return {
        "model": model,
        "hidden_width": int(hidden_width),
        "activation": activation,
        "weight_decay": float(weight_decay),
        "val_mse": float(best_val_loss),
        "parameter_count": int(count_parameters(model)),
        "train_time": float(t1 - t0),
        "y_val_pred": y_val_pred,
        **stats,
    }


def train_final_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_width: int,
    activation: str,
    weight_decay: float,
    lr: float = 1e-3,
    epochs: int = 2000,
    batch_size: int = 16,
    seed: int = GLOBAL_SEED,
    device: str = "cpu",
):
    """Train the selected MLP on all training data and return final metrics plus the model.

    This final training layer was organized by Codex
    so the result notebooks coul reuse one baseline function.
    """
    set_torch_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_width=hidden_width,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    t1 = time.perf_counter()

    train_stats = evaluate_mlp_on_data(model, X_train, y_train, device=device)

    return {
        "model": model,
        "hidden_width": int(hidden_width),
        "activation": activation,
        "weight_decay": float(weight_decay),
        "parameter_count": int(count_parameters(model)),
        "train_time": float(t1 - t0),
        "train_mse": train_stats["mse"],
        "train_max_abs_error": train_stats["max_abs_error"],
        "train_mean_abs_error": train_stats["mean_abs_error"],
        "train_rms_error": train_stats["rms_error"],
    }


def evaluate_mlp_on_data(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
    """Evaluate an MLP on NumPy input/target arrays and return predictions plus error stats."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    model = model.to(device)
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()

    stats = compute_error_stats_with_mse(y, y_pred)
    stats["y_pred"] = y_pred
    return stats


def run_mlp_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    widths=(32, 64, 128),
    activations=("tanh", "relu"),
    weight_decays=(0.0, 1e-5),
    lr: float = 1e-3,
    epochs: int = 2000,
    batch_size: int = 16,
    seed: int = GLOBAL_SEED,
    device: str = "cpu",
):
    """Train several MLP configs from iterable options and return a summary table plus best result.

    The grid search structure was adapted from original code written by me to perform
    experiments, with improvements from Codex.
    """
    rows = []
    best_result = None

    for hidden_width, activation, weight_decay in itertools.product(widths, activations, weight_decays):
        result = train_single_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_width=hidden_width,
            activation=activation,
            weight_decay=weight_decay,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )

        rows.append(
            {
                "hidden_width": result["hidden_width"],
                "activation": result["activation"],
                "weight_decay": result["weight_decay"],
                "parameter_count": result["parameter_count"],
                "train_time": result["train_time"],
                "val_mse": result["val_mse"],
                "val_max_abs_error": result["max_abs_error"],
                "val_mean_abs_error": result["mean_abs_error"],
                "val_rms_error": result["rms_error"],
            }
        )

        if best_result is None or result["val_mse"] < best_result["val_mse"]:
            best_result = result

    summary = pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)
    return summary, best_result


def make_sequence_train_validation_split(X_seq: np.ndarray, Y: np.ndarray, val_fraction: float = 0.25):
    """Split sequence tensors into train and validation subsets and return arrays plus indices."""
    X_seq = np.asarray(X_seq)
    Y = np.asarray(Y)
    n_total = X_seq.shape[0]
    n_val = max(1, int(round(float(val_fraction) * n_total)))

    val_idx = np.linspace(0, n_total - 1, n_val + 2, dtype=int)[1:-1]
    val_idx = np.unique(val_idx)
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    return X_seq[train_idx], Y[train_idx], X_seq[val_idx], Y[val_idx], train_idx, val_idx


class GRUSequenceRegressor(nn.Module):
    """GRU sequence to sequence regressor used for time evolution baselines.

    Codex and ChatGPT helped separate this recurrent baseline from the time
    evolution result notebook.
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.0):
        """Create a GRU readout model from integer dimensions and dropout settings and returns None."""
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.readout = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a sequence tensor through the GRU and return a sequence prediction tensor."""
        out, _ = self.gru(x)
        return self.readout(out)


def train_gru_sequence_model(
    X_train_seq: np.ndarray,
    Y_train: np.ndarray,
    X_eval_seq: np.ndarray,
    Y_eval: np.ndarray,
    hidden_dim=64,
    num_layers=1,
    dropout=0.0,
    epochs=10000,
    lr=1e-3,
    seed: int = GLOBAL_SEED,
    device: str = "cpu",
    dtype=torch.float64,
    verbose: bool = True,
    print_every: int = 1000,
):
    """Train one GRU configuration on sequence arrays and return predictions, metrics, and model.

    This reusable trainer was organized with Codex and ChatGPT from the
    original time evolution baseline cells.
    """
    set_torch_seed(seed)
    X_train_t = torch.tensor(X_train_seq, dtype=dtype, device=device)
    Y_train_t = torch.tensor(np.asarray(Y_train)[:, :, None], dtype=dtype, device=device)
    X_eval_t = torch.tensor(X_eval_seq, dtype=dtype, device=device)

    model = GRUSequenceRegressor(
        input_dim=X_train_t.shape[-1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % int(print_every) == 0):
            print(f"GRU epoch {epoch:5d} | train mse = {loss_value:.6e}")

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        Y_train_pred = model(X_train_t).cpu().numpy()[..., 0]
        Y_eval_pred = model(X_eval_t).cpu().numpy()[..., 0]

    return {
        "model": model,
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "parameter_count": int(count_parameters(model)),
        "train_time_s": float(train_time),
        "best_epoch": int(best_epoch),
        "best_train_mse": float(best_loss),
        "Y_train_pred_seq": Y_train_pred,
        "Y_eval_pred_seq": Y_eval_pred,
        "Y_train_pred": Y_train_pred,
        "Y_eval_pred": Y_eval_pred,
        "train_stats": compute_error_stats_with_mse(Y_train, Y_train_pred),
        "eval_stats": compute_error_stats_with_mse(Y_eval, Y_eval_pred),
    }


def run_gru_grid_search(
    X_train_seq: np.ndarray,
    Y_train: np.ndarray,
    hidden_dims=(32, 64, 128),
    num_layers_options=(1, 2, 3),
    dropouts=(0.0,),
    lr=1e-3,
    epochs=10000,
    val_fraction=0.25,
    seed: int = GLOBAL_SEED,
    device: str = "cpu",
    dtype=torch.float64,
    verbose: bool = False,
):
    """Train several GRU configs from iterable options and return a validation table plus best config.

    Codex and ChatGPT helped build this search loop so GRU experiments could
    be run from the result notebook with consistent metadata.
    """
    split = make_sequence_train_validation_split(X_train_seq, Y_train, val_fraction=val_fraction)
    X_fit, Y_fit, X_val, Y_val, train_idx, val_idx = split

    rows = []
    best_key = None

    for hidden_dim, num_layers, dropout in itertools.product(hidden_dims, num_layers_options, dropouts):
        result = train_gru_sequence_model(
            X_fit,
            Y_fit,
            X_val,
            Y_val,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            lr=lr,
            seed=seed,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        val_stats = result["eval_stats"]
        row = {
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "parameter_count": int(result["parameter_count"]),
            "train_time_s": float(result["train_time_s"]),
            "best_epoch": int(result["best_epoch"]),
            "best_train_mse": float(result["best_train_mse"]),
            "val_mse": float(val_stats["mse"]),
            "val_mean_abs_error": float(val_stats["mean_abs_error"]),
            "val_max_abs_error": float(val_stats["max_abs_error"]),
            "val_rms_error": float(val_stats["rms_error"]),
        }
        rows.append(row)
        if best_key is None or row["val_mse"] < best_key:
            best_key = row["val_mse"]

    summary = pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)
    best_config = summary.iloc[0].to_dict()
    return summary, best_config, {"train_idx": train_idx, "val_idx": val_idx}


def train_final_gru_from_config(
    X_train_seq: np.ndarray,
    Y_train: np.ndarray,
    X_test_seq: np.ndarray,
    Y_test: np.ndarray,
    config,
    lr=1e-3,
    epochs=10000,
    seed: int = GLOBAL_SEED,
    device: str = "cpu",
    dtype=torch.float64,
    verbose: bool = False,
):
    """Train the selected GRU config on all training sequences and return test predictions and metrics.

    This wrapper keeps the final GRU training step consistent with the grid
    search structure, allowing for easy comparison across different configurations.
    This was suggested by Codex and implemented by me.
    """
    result = train_gru_sequence_model(
        X_train_seq,
        Y_train,
        X_test_seq,
        Y_test,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.0)),
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )
    result["Y_test_pred"] = result.pop("Y_eval_pred")
    result["Y_test_pred_seq"] = result.pop("Y_eval_pred_seq")
    result["test_stats"] = result.pop("eval_stats")
    return result
