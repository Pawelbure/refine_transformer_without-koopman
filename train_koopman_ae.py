#!/usr/bin/env python
# train_koopman_ae.py
#
# Script 2:
# - loads pre-generated two-body data from data/two_body_dataset_*.npz
# - builds and trains a Koopman autoencoder on normalized trajectories
# - enforces multi-step Koopman consistency (K^k z_t ≈ z_{t+k})
# - saves best model + training curves + a rollout sanity plot

import os
import glob
import math
import shutil
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import *

from experiment_configs import get_experiment_config, DEFAULT_EXPERIMENT

# ============================================================
# Global config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Dataset: windowed sequences from normalized trajectories
# ============================================================
class WindowedTrajectoryDataset(Dataset):
    """
    data: np.ndarray (N_traj, T, x_dim), normalized
    Each sample: x_seq: (SEQ_LEN, x_dim)
    """
    def __init__(self, data, seq_len):
        assert data.ndim == 3
        self.data = data
        self.seq_len = seq_len

        self.index = []  # list of (traj_idx, start_t)
        N, T, _ = data.shape
        for n in range(N):
            # Include the last possible window that ends exactly at T
            max_start = T - seq_len + 1
            if max_start <= 0:
                continue
            for t0 in range(max_start):
                self.index.append((n, t0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        n, t0 = self.index[idx]
        x_seq = self.data[n, t0:t0 + self.seq_len, :]  # (seq_len, x_dim)
        return torch.from_numpy(x_seq.astype(np.float32))


# Backward compatibility: handle the historic misspelling used in older checkpoints
# or scripts so that either name resolves to the same dataset class.
WindowedTrajectoryDatset = WindowedTrajectoryDataset


# ============================================================
# Models: Encoder, Decoder, KoopmanAE
# ============================================================
class Encoder(nn.Module):
    def __init__(self, x_dim=4, latent_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        # x: (..., x_dim)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=8, x_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, z):
        # z: (..., latent_dim)
        return self.net(z)


class KoopmanAE(nn.Module):
    """
    Koopman autoencoder:
      - Encoder: x_t -> z_t
      - Decoder: z_t -> x_t
      - Linear Koopman operator K: z_{t+1} ≈ z_t @ K

    Multi-step consistency: K^k z_t ≈ z_{t+k} for k=1..K_MAX
    """
    def __init__(self, x_dim=4, latent_dim=8, hidden_dim=64):
        super().__init__()
        self.encoder = Encoder(x_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, x_dim, hidden_dim)
        # Learnable Koopman operator K (latent_dim x latent_dim)
        self.K = nn.Parameter(torch.eye(latent_dim))

    def forward(self, x_seq):
        """
        x_seq: (B, T, x_dim)
        returns:
          x_rec: (B, T, x_dim)
          z_seq: (B, T, latent_dim)
        """
        B, T, x_dim = x_seq.shape
        z_seq = self.encoder(x_seq.view(B * T, x_dim)).view(B, T, -1)
        x_rec = self.decoder(z_seq.view(B * T, -1)).view(B, T, x_dim)
        return x_rec, z_seq

    def koopman_step(self, z):
        """
        One Koopman step in latent space: z_next = z @ K
        z: (..., latent_dim)
        """
        return z @ self.K


# ============================================================
# Training KoopmanAE with multi-step consistency
# ============================================================
def train_koopman_ae(model, train_loader, val_loader,
                     num_epochs, koopman_lambda, k_max,
                     lr, device, out_dir,
                     val_data_norm=None, train_data_norm=None,
                     state_mean=None, state_std=None,
                     seq_len=None, rollout_steps=None, orbit_plot_every=None,
                     train_sample_indices=None,
                     start_epoch=0,
                     optimizer_state_dict=None,
                     best_val_loss_init=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    mse = nn.MSELoss()

    best_val_loss = float("inf") if best_val_loss_init is None else float(best_val_loss_init)
    history = {"train": [], "val": []}

    if train_sample_indices is None:
        train_sample_indices = _fixed_sample_indices(train_data_norm, num_samples=10, seed=0)

    for epoch in range(1, num_epochs + 1):
        global_epoch = start_epoch + epoch
        # -------------------
        # Train
        # -------------------
        model.train()
        train_loss = 0.0
        for x_seq in train_loader:
            x_seq = x_seq.to(device)  # (B,T,x_dim)

            optimizer.zero_grad()
            x_rec, z_seq = model(x_seq)

            # Reconstruction loss
            recon_loss = mse(x_rec, x_seq)

            # Multi-step Koopman consistency
            B, T, d = z_seq.shape
            multi_step_loss = 0.0
            # limit k_max by sequence length
            effective_k_max = min(k_max, T - 1)
            for k in range(1, effective_k_max + 1):
                z_past_k = z_seq[:, :-k, :]  # (B, T-k, d)
                z_fut_k  = z_seq[:,  k:, :]  # (B, T-k, d)

                z_pred_k = z_past_k
                for _ in range(k):
                    z_pred_k = model.koopman_step(z_pred_k)

                multi_step_loss = multi_step_loss + mse(z_pred_k, z_fut_k)

            koopman_loss = multi_step_loss / effective_k_max

            loss = recon_loss + koopman_lambda * koopman_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_seq.size(0)

        train_loss /= len(train_loader.dataset)

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq in val_loader:
                x_seq = x_seq.to(device)
                x_rec, z_seq = model(x_seq)

                recon_loss = mse(x_rec, x_seq)

                B, T, d = z_seq.shape
                multi_step_loss = 0.0
                effective_k_max = min(k_max, T - 1)
                for k in range(1, effective_k_max + 1):
                    z_past_k = z_seq[:, :-k, :]
                    z_fut_k  = z_seq[:,  k:, :]

                    z_pred_k = z_past_k
                    for _ in range(k):
                        z_pred_k = model.koopman_step(z_pred_k)

                    multi_step_loss = multi_step_loss + mse(z_pred_k, z_fut_k)

                koopman_loss = multi_step_loss / effective_k_max
                loss = recon_loss + koopman_lambda * koopman_loss
                val_loss += loss.item() * x_seq.size(0)

        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Plot 2D orbit every `orbit_plot_every` epochs (if data is provided)
        if (
            orbit_plot_every is not None
            and global_epoch % orbit_plot_every == 0
            and val_data_norm is not None
            and state_mean is not None
            and state_std is not None
            and seq_len is not None
            and rollout_steps is not None
        ):
            print(f"  -> plotting 2D Koopman orbit for epoch {epoch:03d}")
            plot_koopman_orbit_for_epoch(
                model,
                val_data_norm=val_data_norm,
                state_mean=state_mean,
                state_std=state_std,
                seq_len=seq_len,
                rollout_steps=rollout_steps,
                out_dir=out_dir,
                device=device,
                epoch=global_epoch,
            )

        # Plot a handful of training samples every 2 epochs
        if (
            train_data_norm is not None
            and global_epoch % 2 == 0
            and state_mean is not None
            and state_std is not None
            and seq_len is not None
            and rollout_steps is not None
        ):
            plot_koopman_training_samples(
                model,
                train_data_norm=train_data_norm,
                state_mean=state_mean,
                state_std=state_std,
                seq_len=seq_len,
                rollout_steps=rollout_steps,
                out_dir=out_dir,
                device=device,
                epoch=epoch,
                num_samples=10,
                sample_indices=train_sample_indices,
            )
            
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "koopman_lambda": koopman_lambda,
                    "k_max": k_max,
                },
                os.path.join(out_dir, "koopman_ae_best.pt"),
            )

        if global_epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{ts}] [KoopmanAE] Epoch {global_epoch:03d} | Train: {train_loss:.4e} | "
                f"Val: {val_loss:.4e}"
            )

    # -------------------
    # Plot loss curves
    # -------------------
    epochs = np.arange(start_epoch + 1, start_epoch + num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train"], label="Train loss")
    plt.plot(epochs, history["val"],   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KoopmanAE training/validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "koopman_ae_loss.png"))
    plt.close()

    return best_val_loss, history


# ============================================================
# Simple Koopman rollout sanity check (on normalized data)
# ============================================================
def koopman_rollout(model, x_seq_init, rollout_steps, device):
    """
    x_seq_init: (T0, x_dim), normalized
    Rolls out in latent space using K, decodes to x, returns (T0 + rollout_steps, x_dim).
    """
    model.eval()
    encoder = model.encoder
    decoder = model.decoder

    x_seq_init_t = torch.from_numpy(x_seq_init.astype(np.float32)).to(device)  # (T0, x_dim)
    T0, x_dim = x_seq_init_t.shape

    with torch.no_grad():
        z_seq = encoder(x_seq_init_t).unsqueeze(0)  # (1, T0, d)
        x_list = [x_seq_init_t]  # list of (N,x_dim), first is (T0, x_dim)

        z = z_seq[:, -1, :]  # last latent in window (1,d)

        for _ in range(rollout_steps):
            # one step in latent space via K
            z = model.koopman_step(z)  # (1,d)
            x_next = decoder(z)        # (1,x_dim)
            x_list.append(x_next)      # keep 2D shape

        x_all = torch.cat(x_list, dim=0)  # (T0 + rollout_steps, x_dim)

    return x_all.cpu().numpy()


def plot_koopman_rollout_example(model, val_data_norm, state_mean, state_std,
                                 seq_len, rollout_steps, t_eval, out_dir, device):
    """
    Take first validation trajectory, a window of length seq_len, then
    roll out further 'rollout_steps' using K only.

    Produces:
      - time-series plot of x1, y1, x2, y2 over time (true vs rollout)
      - 2D orbit plot in the x-y plane (true vs rollout)
    """
    if val_data_norm.shape[0] == 0:
        return

    x_traj_norm = val_data_norm[0]      # (T,4)
    T_total = x_traj_norm.shape[0]
    if T_total < seq_len + rollout_steps + 1:
        rollout_steps = max(1, T_total - seq_len - 1)

    start_idx = 0
    x_init_norm = x_traj_norm[start_idx:start_idx + seq_len]  # (seq_len,4)

    # Koopman rollout (normalized)
    x_pred_norm = koopman_rollout(
        model, x_init_norm, rollout_steps, device
    )  # (seq_len + rollout_steps, 4)

    # Ground truth segment (normalized)
    x_true_norm = x_traj_norm[start_idx:start_idx + seq_len + rollout_steps]

    # Denormalize for plotting
    x_pred = x_pred_norm * state_std + state_mean
    x_true = x_true_norm * state_std + state_mean
    x_dim = x_true.shape[1]

    # Time axis for this segment
    t_seg = t_eval[start_idx:start_idx + seq_len + rollout_steps]

    # ----------------------------------------------------------
    # 1) Time-series plot: x1, y1, x2, y2 vs t
    # ----------------------------------------------------------
    var_names = ["x1", "y1", "x2", "y2"] if x_dim == 4 else [f"x{i}" for i in range(x_dim)]
    plt.figure(figsize=(10, 2 * x_dim))
    for i in range(x_dim):
        plt.subplot(x_dim, 1, i + 1)
        plt.plot(t_seg, x_true[:, i], label="True", linewidth=1.5)
        plt.plot(t_seg, x_pred[:, i], "--", label="Koopman rollout", linewidth=1.2)

        # mark boundary between initial window and predicted steps
        t_boundary = t_seg[seq_len - 1]
        plt.axvline(t_boundary, color="k", linestyle=":", linewidth=1)

        if i == 0:
            plt.title("KoopmanAE: true vs K-only rollout (validation example)")
            plt.legend(loc="upper right", fontsize=8)
        plt.ylabel(var_names[i])
        plt.grid(True, alpha=0.3)
        if i == x_dim - 1:
            plt.xlabel("t")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "koopman_rollout_example.png"))
    plt.close()

    # ----------------------------------------------------------
    # 2) 2D orbit plot: (x1,y1) and (x2,y2), true vs rollout
    # ----------------------------------------------------------
    plt.figure(figsize=(7, 7))
    idx_boundary = seq_len - 1

    if x_dim == 4:
        x1_true, y1_true = x_true[:, 0], x_true[:, 1]
        x2_true, y2_true = x_true[:, 2], x_true[:, 3]

        x1_pred, y1_pred = x_pred[:, 0], x_pred[:, 1]
        x2_pred, y2_pred = x_pred[:, 2], x_pred[:, 3]

        # True orbits
        plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
        plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

        # Predicted orbits
        plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (K-rollout)", linewidth=1.5, color="C0")
        plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (K-rollout)", linewidth=1.5, color="C1")

        # Mark prediction start (after seq_len-1)
        plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                    color="C0", marker="o", s=40, label="Start pred M1 (true)")
        plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                    color="C1", marker="o", s=40, label="Start pred M2 (true)")
    else:
        plt.plot(x_true[:, 0], x_true[:, 1], label="True", linewidth=1.5, color="C0")
        plt.plot(x_pred[:, 0], x_pred[:, 1], "--", label="K-rollout", linewidth=1.5, color="C1")
        plt.scatter(x_true[idx_boundary, 0], x_true[idx_boundary, 1],
                    color="C0", marker="o", s=40, label="Prediction start")

    plt.title("KoopmanAE: 2D orbit, true vs K-only rollout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "koopman_rollout_orbit_example.png"))
    plt.close()

def plot_koopman_orbit_for_epoch(model, val_data_norm, state_mean, state_std,
                                 seq_len, rollout_steps, out_dir, device, epoch):
    """
    During training: use first validation trajectory,
    take a window of length seq_len, roll out 'rollout_steps' using K,
    and save a 2D orbit plot with the epoch number in the filename.
    """
    if val_data_norm.shape[0] == 0:
        return

    x_traj_norm = val_data_norm[0]   # (T,4)
    T_total = x_traj_norm.shape[0]
    if T_total < seq_len + rollout_steps + 1:
        rollout_steps = max(1, T_total - seq_len - 1)

    start_idx = 0
    x_init_norm = x_traj_norm[start_idx:start_idx + seq_len]  # (seq_len,4)

    # Koopman rollout (normalized)
    x_pred_norm = koopman_rollout(
        model, x_init_norm, rollout_steps, device
    )  # (seq_len + rollout_steps, 4)

    # Ground truth segment (normalized)
    x_true_norm = x_traj_norm[start_idx:start_idx + seq_len + rollout_steps]

    # Denormalize
    x_pred = x_pred_norm * state_std + state_mean
    x_true = x_true_norm * state_std + state_mean
    x_dim = x_true.shape[1]

    plt.figure(figsize=(7, 7))

    idx_boundary = seq_len - 1
    if x_dim == 4:
        x1_true, y1_true = x_true[:, 0], x_true[:, 1]
        x2_true, y2_true = x_true[:, 2], x_true[:, 3]

        x1_pred, y1_pred = x_pred[:, 0], x_pred[:, 1]
        x2_pred, y2_pred = x_pred[:, 2], x_pred[:, 3]

        # True orbits
        plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
        plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

        # Predicted orbits
        plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (K-rollout)", linewidth=1.5, color="C0")
        plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (K-rollout)", linewidth=1.5, color="C1")

        # Mark prediction start
        plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                    color="C0", marker="o", s=40, label="Start pred M1")
        plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                    color="C1", marker="o", s=40, label="Start pred M2")
    else:
        plt.plot(x_true[:, 0], x_true[:, 1], label="True", linewidth=1.5, color="C0")
        plt.plot(x_pred[:, 0], x_pred[:, 1], "--", label="K-rollout", linewidth=1.5, color="C1")
        plt.scatter(x_true[idx_boundary, 0], x_true[idx_boundary, 1],
                    color="C0", marker="o", s=40, label="Prediction start")

    plt.title(f"KoopmanAE orbit (epoch {epoch:03d})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, f"koopman_orbit_epoch_{epoch:03d}.png")
    plt.savefig(fname)
    plt.close()


def _fixed_sample_indices(train_data_norm, num_samples=10, seed=0):
    if train_data_norm.shape[0] == 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(seed=seed)
    return rng.choice(
        train_data_norm.shape[0],
        size=min(num_samples, train_data_norm.shape[0]),
        replace=False,
    )


def plot_koopman_training_samples(model, train_data_norm, state_mean, state_std,
                                  seq_len, rollout_steps, out_dir, device,
                                  epoch, num_samples=10, sample_indices=None):
    """
    Draw multiple random training trajectories and plot their Koopman rollouts
    as 2D orbits. Results are saved under
    out_dir/training_samples/{epoch}epochs/.
    """
    if train_data_norm.shape[0] == 0:
        return

    if sample_indices is None:
        sample_indices = _fixed_sample_indices(
            train_data_norm, num_samples=num_samples, seed=0
        )

    epoch_dir = os.path.join(out_dir, "training_samples", f"{epoch}epochs")
    os.makedirs(epoch_dir, exist_ok=True)

    for idx in sample_indices:
        x_traj_norm = train_data_norm[idx]
        if x_traj_norm.shape[0] < seq_len + 1:
            continue

        steps = min(rollout_steps, max(1, x_traj_norm.shape[0] - seq_len - 1))
        x_init_norm = x_traj_norm[:seq_len]

        x_pred_norm = koopman_rollout(model, x_init_norm, steps, device)
        x_true_norm = x_traj_norm[:seq_len + steps]

        x_pred = x_pred_norm * state_std + state_mean
        x_true = x_true_norm * state_std + state_mean
        x_dim = x_true.shape[1]

        plt.figure(figsize=(7, 7))
        idx_boundary = seq_len - 1

        if x_dim >= 4:
            x1_true, y1_true = x_true[:, 0], x_true[:, 1]
            x2_true, y2_true = x_true[:, 2], x_true[:, 3]
            x1_pred, y1_pred = x_pred[:, 0], x_pred[:, 1]
            x2_pred, y2_pred = x_pred[:, 2], x_pred[:, 3]

            plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
            plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

            plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (K-rollout)", linewidth=1.5, color="C0")
            plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (K-rollout)", linewidth=1.5, color="C1")

            plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                        color="C0", marker="o", s=40, label="Start pred M1")
            plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                        color="C1", marker="o", s=40, label="Start pred M2")
        else:
            plt.plot(x_true[:, 0], x_true[:, 1], label="True", linewidth=1.5, color="C0")
            plt.plot(x_pred[:, 0], x_pred[:, 1], "--", label="K-rollout", linewidth=1.5, color="C1")

            plt.scatter(x_true[idx_boundary, 0], x_true[idx_boundary, 1],
                        color="C0", marker="o", s=40, label="Prediction start")

        plt.title(f"KoopmanAE train sample {idx} (epoch {epoch:03d})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(epoch_dir, f"koopman_train_sample_{idx}.png")
        plt.savefig(fname)
        plt.close()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help="Name of experiment configuration to use.",
    )
    parser.add_argument(
        "--koopman_mode",
        choices=["reuse", "continue", "train"],
        help=(
            "How to handle an existing Koopman model: reuse the latest checkpoint, "
            "continue training it, or train a fresh model."
        ),
    )
    parser.add_argument(
        "--reuse_koopman",
        action="store_true",
        help="DEPRECATED: Equivalent to --koopman_mode=reuse.",
    )
    args = parser.parse_args()

    # Determine Koopman handling mode
    koopman_mode = args.koopman_mode
    if koopman_mode is None and args.reuse_koopman:
        koopman_mode = "reuse"

    cfg      = get_experiment_config(args.experiment)

    EXP_DATA_ROOT = f"{cfg.name}/{cfg.DATA_ROOT}"
    EXP_OUTPUT_ROOT = f"{cfg.name}/outputs"

    os.makedirs(EXP_OUTPUT_ROOT, exist_ok=True)

    sim_cfg  = cfg.simulation
    ds_cfg   = cfg.dataset
    k_cfg    = cfg.koopman

    LATENT_DIM = k_cfg.LATENT_DIM
    HIDDEN_DIM = k_cfg.HIDDEN_DIM

    SEQ_LEN     = ds_cfg.SEQ_LEN
    HORIZON     = ds_cfg.HORIZON
    BATCH_SIZE  = k_cfg.BATCH_SIZE
    EPOCHS      = k_cfg.EPOCHS
    LR          = k_cfg.LR
    K_MAX       = k_cfg.K_MAX
    KOOPMAN_LAMBDA = k_cfg.KOOPMAN_LAMBDA

    ROLLOUT_STEPS = cfg.transformer.ROLLOUT_STEPS

    # 1) Load dataset
    dataset_pattern = f"{sim_cfg.PROBLEM.replace('-', '_')}_dataset_*.npz"
    ds_file = find_latest_dataset(pattern=dataset_pattern, data_dir=EXP_DATA_ROOT)
    print(f"Loading dataset from: {ds_file}")
    data = np.load(ds_file)

    t_eval = data["t_eval"]                 # (T,)
    state_mean = data["state_mean"]         # (4,)
    state_std  = data["state_std"]          # (4,)

    train_norm = data["train_norm"]         # (N_train, T, 4)
    val_norm   = data["val_norm"]           # (N_val,   T, 4)
    # test_norm  = data["test_norm"]        # (N_test,  T, 4)  # not used yet

    N_train, T, x_dim = train_norm.shape
    print(f"Train norm shape: {train_norm.shape}, Val norm shape: {val_norm.shape}")

    # Keep a fixed set of training trajectories for reproducible plotting
    train_sample_indices = _fixed_sample_indices(train_norm, num_samples=10, seed=0)

    # 2) Build windowed datasets + loaders
    train_dataset = WindowedTrajectoryDatset(train_norm, seq_len=SEQ_LEN)
    val_dataset   = WindowedTrajectoryDataset(val_norm,   seq_len=SEQ_LEN)

    if len(train_dataset) == 0:
        raise ValueError(
            f"No training windows available: trajectory length {T} < SEQ_LEN {SEQ_LEN}. "
            "Increase trajectory length or reduce SEQ_LEN in the experiment config."
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)

    print(f"Windowed train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    # ------------------------------------------------------
    # Ask whether to reuse or continue the latest trained KoopmanAE model
    # ------------------------------------------------------
    # Prepare the new output directory name, but don't create it yet so the glob below
    # only sees previously-finished runs.
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(EXP_OUTPUT_ROOT, f"koopman_ae_{time_tag}")

    # Find latest KoopmanAE experiment folder
    koopman_dirs = sorted(
        glob.glob(os.path.join(EXP_OUTPUT_ROOT, "koopman_ae_*"))
    )
    prev_koopman_dirs = [d for d in koopman_dirs if os.path.abspath(d) != os.path.abspath(out_dir)]
    completed_koopman_dirs = [
        d for d in prev_koopman_dirs
        if os.path.exists(os.path.join(d, "koopman_ae_best.pt"))
    ]

    if prev_koopman_dirs and not completed_koopman_dirs:
        print("Found Koopman output directories without checkpoints; skipping them for reuse.")

    reuse_model = False
    continue_training = False
    reuse_path = None
    resume_optimizer_state = None
    start_epoch = 0
    best_val_from_ckpt = None

    reuse_source_dir = completed_koopman_dirs[-1] if completed_koopman_dirs else None
    reuse_source_ckpt = os.path.join(reuse_source_dir, "koopman_ae_best.pt") if reuse_source_dir else None

    if koopman_mode is None and reuse_source_dir and os.path.exists(reuse_source_ckpt):
        print("\nA previously trained KoopmanAE was found:")
        print(f"  {reuse_source_ckpt}")
        choice = input("Choose action [r]euse / [c]ontinue / [n]ew (default=n): ").strip().lower()
        if choice == "r":
            koopman_mode = "reuse"
        elif choice == "c":
            koopman_mode = "continue"
        else:
            koopman_mode = "train"
    elif koopman_mode is None:
        koopman_mode = "train"

    if koopman_mode in {"reuse", "continue"} and not reuse_source_ckpt:
        print("\nNo previous Koopman checkpoint found. Training a new model instead.\n")
        koopman_mode = "train"

    if koopman_mode == "reuse":
        print(f"\nReusing existing KoopmanAE checkpoint from: {reuse_source_ckpt}")
        shutil.copytree(reuse_source_dir, out_dir, dirs_exist_ok=True)
        reuse_model = True
        reuse_path = os.path.join(out_dir, "koopman_ae_best.pt")
        print(f"Copied previous KoopmanAE outputs into new run directory: {out_dir}")
    elif koopman_mode == "continue":
        print(f"\nContinuing training from latest KoopmanAE checkpoint: {reuse_source_ckpt}")
        shutil.copytree(reuse_source_dir, out_dir, dirs_exist_ok=True)
        ckpt = torch.load(reuse_source_ckpt, map_location=DEVICE)
        reuse_path = os.path.join(out_dir, "koopman_ae_best.pt")
        torch.save(ckpt, reuse_path)  # ensure checkpoint exists in new run directory
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_from_ckpt = float(ckpt.get("val_loss", float("inf")))
        resume_optimizer_state = ckpt.get("optimizer_state_dict")
        continue_training = True
        print(f"Copied previous KoopmanAE outputs into new run directory: {out_dir}")
    else:
        koopman_mode = "train"
        # Ensure the fresh output directory exists before training starts when not reusing.
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------
    # 4) Build KoopmanAE and train or reuse
    # ------------------------------------------------------
    model = KoopmanAE(x_dim=x_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

    if reuse_model:
        print(f"\nLoading existing KoopmanAE from: {reuse_path}")
        ckpt = torch.load(reuse_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded model. Skipping training.\n")
        best_val_loss = ckpt["val_loss"]
        history = None  # optional
    else:
        if continue_training:
            print(f"\nResuming KoopmanAE training from epoch {start_epoch}...\n")
            ckpt = torch.load(reuse_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            best_val_loss_init = ckpt.get("val_loss")
        else:
            print("\nTraining new KoopmanAE model...\n")
            best_val_loss_init = None

        best_val_loss, history = train_koopman_ae(
            model,
            train_loader,
            val_loader,
            num_epochs=EPOCHS,
            koopman_lambda=KOOPMAN_LAMBDA,
            k_max=K_MAX,
            lr=LR,
            device=DEVICE,
            out_dir=out_dir,
            # NEW: for orbit plotting during training
            val_data_norm=val_norm,
            train_data_norm=train_norm,
            state_mean=state_mean,
            state_std=state_std,
            seq_len=SEQ_LEN,
            rollout_steps=ROLLOUT_STEPS,
            orbit_plot_every=2,
            train_sample_indices=train_sample_indices,
            start_epoch=start_epoch if continue_training else 0,
            optimizer_state_dict=resume_optimizer_state if continue_training else None,
            best_val_loss_init=best_val_loss_init,
        )

    # Save or augment meta info about this run
    koopman_info_path = os.path.join(out_dir, "koopman_info.txt")
    if reuse_model:
        with open(koopman_info_path, "a") as f:
            f.write("\n")
            f.write(f"Reused checkpoint from: {reuse_source_dir}\n")
            f.write(f"Reuse invocation dataset file: {ds_file}\n")
    else:
        with open(koopman_info_path, "w") as f:
            f.write(f"Dataset file: {ds_file}\n")
            f.write(f"SEQ_LEN: {SEQ_LEN}\n")
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"LR: {LR}\n")
            f.write(f"LATENT_DIM: {LATENT_DIM}\n")
            f.write(f"HIDDEN_DIM: {HIDDEN_DIM}\n")
            f.write(f"KOOPMAN_LAMBDA: {KOOPMAN_LAMBDA}\n")
            f.write(f"K_MAX: {K_MAX}\n")
            if continue_training:
                f.write(f"Resumed from: {reuse_source_dir}\n")
                f.write(f"Start epoch: {start_epoch}\n")

    print(f"Best validation loss: {best_val_loss:.4e}")
    if reuse_model:
        print(f"Using existing model from: {reuse_path}")
    else:
        print(f"Saved model and loss curves in: {out_dir}")

    # 5) Koopman rollout sanity-check plot
    #    Reload best model (just to be clean)
    ckpt_path = reuse_path if reuse_model else os.path.join(out_dir, "koopman_ae_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)

    final_epoch = ckpt.get("epoch", "final")
    plot_koopman_training_samples(
        model,
        train_data_norm=train_norm,
        state_mean=state_mean,
        state_std=state_std,
        seq_len=SEQ_LEN,
        rollout_steps=ROLLOUT_STEPS,
        out_dir=out_dir,
        device=DEVICE,
        epoch=final_epoch,
        num_samples=10,
        sample_indices=train_sample_indices,
    )

    # We use val_norm for an example, and a shorter rollout (e.g., 200 steps)
    plot_koopman_rollout_example(
        model,
        val_norm,
        state_mean,
        state_std,
        seq_len=SEQ_LEN,
        rollout_steps=ROLLOUT_STEPS,
        t_eval=t_eval,
        out_dir=out_dir,
        device=DEVICE,
    )

    print("Generated Koopman rollout sanity plot.")


if __name__ == "__main__":
    main()