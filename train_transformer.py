#!/usr/bin/env python
# train_transformer.py
#
# Script 3:
# - loads pre-generated two-body data from data/two_body_dataset_*.npz
# - loads latest trained KoopmanAE (encoder/decoder) from train_koopman_ae.py checkpoints
# - builds windowed datasets on normalized trajectories
# - trains a Transformer in latent space to predict next latent state
# - evaluates one-step MSE and long rollout
# - saves model + loss curves + a 2D orbit rollout plot

import os
import glob
import math
from datetime import datetime

import argparse

from experiment_configs import get_experiment_config, DEFAULT_EXPERIMENT
from train_koopman_ae import KoopmanAE

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Global config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

from utils import *
    
# ============================================================
# Dataset for Transformer: windowed x_in / x_out
# ============================================================
class WindowedSequenceDataset(Dataset):
    """
    data: np.ndarray (N_traj, T_total, x_dim), normalized
    Each sample:
      x_in  : (SEQ_LEN, x_dim)
      x_out : (HORIZON, x_dim)
    """
    def __init__(self, data, seq_len, horizon):
        assert data.ndim == 3
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

        self.index = []  # list of (traj_idx, start_t)
        N, T, _ = data.shape
        for n in range(N):
            max_start = T - seq_len - horizon + 1
            if max_start <= 0:
                continue
            for t0 in range(max_start):
                self.index.append((n, t0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        n, t0 = self.index[idx]
        x_traj = self.data[n]  # (T_total, x_dim)
        x_in  = x_traj[t0:t0 + self.seq_len]                  # (SEQ_LEN, x_dim)
        x_out = x_traj[t0 + self.seq_len:t0 + self.seq_len + self.horizon]  # (HORIZON, x_dim)
        return {
            "x_in":  torch.from_numpy(x_in.astype(np.float32)),
            "x_out": torch.from_numpy(x_out.astype(np.float32)),
            "t0": torch.tensor(t0, dtype=torch.long),
        }
    
# ============================================================
# Transformer in latent space
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x, start_pos: int = 0, position_ids=None):
        """
        If ``position_ids`` is provided (shape: [B, T]), use absolute positions
        for each item in the batch. Otherwise, fall back to a shared offset
        slice starting at ``start_pos``.
        """
        if position_ids is not None:
            # Clamp to buffer length to avoid OOB if rollout length exceeds max
            pos = position_ids.clamp(max=self.pe.size(1) - 1)
            pe = self.pe[0, pos, :]
            return x + pe

        T = x.size(1)
        return x + self.pe[:, start_pos:start_pos + T, :]


class LatentTransformer(nn.Module):
    def __init__(self,
                 latent_dim=8,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.0,
                 max_len=500):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_norm = nn.LayerNorm(latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.readout = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_seq, pos_offset: int = 0, position_ids=None):
        """
        z_seq: (B, T, latent_dim)
        position_ids: None or (B, T) absolute positions for PE slicing
        returns z_next_pred: (B, latent_dim)
        """
        z_seq = self.input_norm(z_seq)
        z_seq = self.pos_encoder(z_seq, start_pos=pos_offset, position_ids=position_ids)
        h = self.transformer(z_seq)      # (B, T, latent_dim)
        h_last = h[:, -1, :]
        return self.readout(h_last)

# ============================================================
# Training utilities
# ============================================================
def train_transformer(dyn_model, encoder, decoder,
                      train_loader, val_loader,
                      num_epochs, lr, device, out_dir,
                      test_norm, val_norm, train_norm,
                      state_mean, state_std,
                      t_eval, seq_len, rollout_steps,
                      horizon,
                      x_weight=1.0,
                      teacher_forcing_start=1.0,
                      teacher_forcing_end=0.2,
                      latent_noise_std=0.0,
                      grad_clip=0.0,
                      fine_tune_encoder=False,
                      train_sample_indices=None,
                      start_epoch=0,
                      optimizer_state_dict=None,
                      best_val_loss_init=None,
                      best_rollout_loss_init=None):
    dyn_model.to(device)
    encoder.to(device)
    decoder.to(device)

    # fine-tune encoder optionally, keep decoder frozen for stable reconstructions
    for p in encoder.parameters():
        p.requires_grad = bool(fine_tune_encoder)
    for p in decoder.parameters():
        p.requires_grad = False
    encoder.train()
    decoder.eval()

    trainable_params = list(dyn_model.parameters())
    if fine_tune_encoder:
        trainable_params += list(encoder.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    mse = nn.MSELoss()

    best_val_loss = float("inf") if best_val_loss_init is None else float(best_val_loss_init)
    best_rollout_loss = float("inf") if best_rollout_loss_init is None else float(best_rollout_loss_init)
    history = {"train": [], "val": [], "rollout_val": []}
    if train_sample_indices is None:
        train_sample_indices = _fixed_sample_indices(train_norm, num_samples=10, seed=0)

    def teacher_force_prob(epoch_idx: int) -> float:
        if num_epochs == 1:
            return teacher_forcing_end
        alpha = (epoch_idx - 1) / (num_epochs - 1)
        return teacher_forcing_start + alpha * (teacher_forcing_end - teacher_forcing_start)

    for epoch in range(1, num_epochs + 1):
        global_epoch = start_epoch + epoch
        # linear decay of teacher forcing
        tf_ratio = teacher_forcing_end + (teacher_forcing_start - teacher_forcing_end) * max(0.0, (num_epochs - epoch) / max(1, num_epochs - 1))
        # -------------------
        # Train
        # -------------------
        dyn_model.train()
        encoder.train()
        train_loss = 0.0
        for batch in train_loader:
            x_in  = batch["x_in"].to(device)   # (B, T, x_dim)
            x_out = batch["x_out"].to(device)  # (B, H, x_dim)
            t0    = batch["t0"].to(device)     # (B,)

            optimizer.zero_grad()

            z_seq = encoder(x_in)                 # (B,T,d)
            z_out_all = encoder(x_out.view(-1, x_out.size(-1)))
            z_out_all = z_out_all.view(x_out.size(0), x_out.size(1), -1)

            pos_seq = t0.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)

            if latent_noise_std > 0.0:
                z_seq = z_seq + torch.randn_like(z_seq) * latent_noise_std

            loss_latent = 0.0
            loss_x = 0.0
            pos_cursor = z_seq.size(1) - 1

            for h in range(horizon):
                start_pos = max(0, pos_cursor - z_seq.size(1) + 1)
                position_ids = pos_seq[:, -z_seq.size(1):]
                z_next_pred = dyn_model(z_seq, pos_offset=start_pos, position_ids=position_ids)  # (B,d)

                z_next_true = z_out_all[:, h, :]
                loss_latent = loss_latent + mse(z_next_pred, z_next_true)

                x_next_pred = decoder(z_next_pred)
                x_next_true = x_out[:, h, :]
                loss_x = loss_x + mse(x_next_pred, x_next_true)

                use_teacher = torch.rand(1, device=device) < tf_ratio
                z_feed = z_next_true if use_teacher else z_next_pred

                z_seq = torch.cat([z_seq, z_feed.unsqueeze(1)], dim=1)
                z_seq = z_seq[:, -seq_len:, :]
                next_pos = (t0 + seq_len + h).unsqueeze(1)
                pos_seq = torch.cat([pos_seq, next_pos], dim=1)
                pos_seq = pos_seq[:, -seq_len:]
                pos_cursor += 1

            loss_latent = loss_latent / horizon
            loss_x = loss_x / horizon

            loss = loss_latent + x_weight * loss_x
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            optimizer.step()
            train_loss += loss.item() * x_in.size(0)

        train_loss /= len(train_loader.dataset)

        # -------------------
        # Validation (full teacher forcing)
        # -------------------
        dyn_model.eval()
        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_in  = batch["x_in"].to(device)
                x_out = batch["x_out"].to(device)
                t0    = batch["t0"].to(device)
                B, H, _ = x_out.shape

                z_seq = encoder(x_in)
                z_out_all = encoder(x_out.view(-1, x_out.size(-1)))
                z_out_all = z_out_all.view(x_out.size(0), x_out.size(1), -1)

                pos_seq = t0.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)

                loss_latent = 0.0
                loss_x = 0.0
                pos_cursor = z_seq.size(1) - 1

                for h in range(horizon):
                    start_pos = max(0, pos_cursor - z_seq.size(1) + 1)
                    position_ids = pos_seq[:, -z_seq.size(1):]
                    z_next_pred = dyn_model(z_seq, pos_offset=start_pos, position_ids=position_ids)
                    z_next_true = z_out_all[:, h, :]
                    loss_latent = loss_latent + mse(z_next_pred, z_next_true)

                    x_next_pred = decoder(z_next_pred)
                    x_next_true = x_out[:, h, :]
                    loss_x = loss_x + mse(x_next_pred, x_next_true)

                    z_seq = torch.cat([z_seq, z_next_pred.unsqueeze(1)], dim=1)
                    z_seq = z_seq[:, -seq_len:, :]
                    next_pos = (t0 + seq_len + h).unsqueeze(1)
                    pos_seq = torch.cat([pos_seq, next_pos], dim=1)
                    pos_seq = pos_seq[:, -seq_len:]
                    pos_cursor += 1

                loss_latent = loss_latent / horizon
                loss_x = loss_x / horizon

                loss = loss_latent + x_weight * loss_x
                val_loss += loss.item() * x_in.size(0)

        val_loss /= len(val_loader.dataset)

        rollout_val_loss = long_rollout_val_mse(
            val_norm, encoder, decoder, dyn_model, device, seq_len, rollout_steps
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["rollout_val"].append(rollout_val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": dyn_model.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "rollout_val_loss": rollout_val_loss,
                },
                os.path.join(out_dir, "transformer_best.pt"),
            )

            # Save encoder snapshot when fine-tuning
            if fine_tune_encoder:
                torch.save(
                    {"encoder_state_dict": encoder.state_dict()},
                    os.path.join(out_dir, "encoder_best.pt"),
                )

        if rollout_val_loss < best_rollout_loss:
            best_rollout_loss = rollout_val_loss
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state_dict": dyn_model.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "rollout_val_loss": rollout_val_loss,
                },
                os.path.join(out_dir, "transformer_best_rollout.pt"),
            )

        if global_epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{ts}] [Transformer] Epoch {global_epoch:03d} | Train: {train_loss:.4e} | "
                f"Val: {val_loss:.4e} | Rollout Val: {rollout_val_loss:.4e} | "
                f"TF prob: {tf_ratio:.2f}"
            )

        if epoch % 2 == 0:
            plot_training_rollout_orbits(
                train_norm=train_norm,
                state_mean=state_mean,
                state_std=state_std,
                encoder=encoder,
                decoder=decoder,
                dyn_model=dyn_model,
                seq_len=seq_len,
                n_future=rollout_steps,
                out_dir=out_dir,
                device=device,
                epoch=epoch,
                num_samples=10,
                sample_indices=train_sample_indices,
            )
            plot_rollout_example_2d_orbit(
                test_norm=test_norm,
                state_mean=state_mean,
                state_std=state_std,
                t_eval=t_eval,
                encoder=encoder,
                decoder=decoder,
                dyn_model=dyn_model,
                seq_len=seq_len,
                n_future=rollout_steps,
                out_dir=out_dir,
                device=device,
                epoch=epoch
            )
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}]   -> Saved Transformer rollout orbit plot for epoch {epoch}.")

    return best_val_loss, best_rollout_loss, history

# ============================================================
# Evaluation utilities
# ============================================================
def one_step_mse_xspace(encoder, decoder, dyn_model, data_loader, device, seq_len):
    encoder.eval()
    decoder.eval()
    dyn_model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in data_loader:
            x_in  = batch["x_in"].to(device)
            x_out = batch["x_out"].to(device)
            t0    = batch["t0"].to(device)

            position_ctx = t0.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)

            z_seq = encoder(x_in)
            z_next_true = encoder(x_out[:, 0, :])
            z_next_pred = dyn_model(z_seq, position_ids=position_ctx)

            x_next_true = x_out[:, 0, :]
            x_next_pred = decoder(z_next_pred)

            loss = mse(x_next_pred, x_next_true)
            total_loss += loss.item() * x_in.size(0)
            n += x_in.size(0)

    return total_loss / n


def rollout_transformer(x_init_norm, n_steps, encoder, decoder, dyn_model, device):
    """
    x_init_norm: (T0, x_dim) normalized initial physical window
    returns:
        x_pred_all_norm: (T0 + n_steps, x_dim) normalized
        first T0 rows = given window
        next n_steps = autoregressive transformer predictions
    """
    encoder.eval()
    decoder.eval()
    dyn_model.eval()

    x_init_t = torch.from_numpy(x_init_norm.astype(np.float32)).to(device)  # (T0, x_dim)
    T0, x_dim = x_init_t.shape

    with torch.no_grad():
        z_seq = encoder(x_init_t).unsqueeze(0)  # (1, T0, d)
        x_list = [x_init_t]  # first block: (T0, x_dim)
        pos_seq = torch.arange(T0, device=device).unsqueeze(0)  # (1, T0)
        pos_cursor = z_seq.size(1) - 1

        for step_idx in range(n_steps):
            start_pos = max(0, pos_cursor - z_seq.size(1) + 1)
            position_ids = pos_seq[:, -z_seq.size(1):]
            z_next = dyn_model(z_seq, pos_offset=start_pos, position_ids=position_ids)      # (1,d)
            x_next = decoder(z_next)       # (1,x_dim)
            x_list.append(x_next)          # keep 2D

            # update latent sequence, keep last T0 states
            z_seq = torch.cat([z_seq, z_next.unsqueeze(1)], dim=1)  # (1,T0+1,d)
            z_seq = z_seq[:, -T0:, :]                              # (1,T0,d)

            next_pos = torch.tensor([[T0 + step_idx]], device=device)
            pos_seq = torch.cat([pos_seq, next_pos], dim=1)
            pos_seq = pos_seq[:, -T0:]
            pos_cursor += 1

        x_pred_all = torch.cat(x_list, dim=0)  # (T0 + n_steps, x_dim)

    return x_pred_all.cpu().numpy()


def long_rollout_val_mse(val_norm, encoder, decoder, dyn_model, device, seq_len, n_future, max_trajs=3):
    """
    Compute autoregressive rollout MSE on a small subset of validation
    trajectories to surface long-horizon drift during training.
    """
    encoder.eval()
    decoder.eval()
    dyn_model.eval()

    mse = nn.MSELoss()
    n_used = 0
    total_loss = 0.0
    with torch.no_grad():
        for i in range(min(max_trajs, val_norm.shape[0])):
            x_traj = val_norm[i]
            if x_traj.shape[0] < seq_len + 1:
                continue
            n_steps = min(n_future, max(1, x_traj.shape[0] - seq_len - 1))
            x_init_norm = x_traj[:seq_len]
            x_target = x_traj[:seq_len + n_steps]

            x_pred_all_norm = rollout_transformer(
                x_init_norm, n_steps, encoder, decoder, dyn_model, device
            )

            x_pred = torch.from_numpy(x_pred_all_norm).to(device)
            x_true = torch.from_numpy(x_target.astype(np.float32)).to(device)
            loss = mse(x_pred, x_true)
            total_loss += loss.item()
            n_used += 1

    return float("inf") if n_used == 0 else total_loss / n_used


def _fixed_sample_indices(train_norm, num_samples=10, seed=0):
    if train_norm.shape[0] == 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(seed=seed)
    return rng.choice(
        train_norm.shape[0],
        size=min(num_samples, train_norm.shape[0]),
        replace=False,
    )


def plot_training_rollout_orbits(train_norm, state_mean, state_std, encoder, decoder,
                                 dyn_model, seq_len, n_future, out_dir, device,
                                 epoch, num_samples=10, sample_indices=None):
    """
    Plot rollout orbits for several random training trajectories every
    few epochs to monitor in-distribution behavior. Saved under
    out_dir/training_samples/{epoch}epochs/.
    """
    if train_norm.shape[0] == 0:
        return

    if sample_indices is None:
        sample_indices = _fixed_sample_indices(
            train_norm, num_samples=num_samples, seed=0
        )

    epoch_dir = os.path.join(out_dir, "training_samples", f"{epoch}epochs")
    os.makedirs(epoch_dir, exist_ok=True)

    for idx in sample_indices:
        x_traj_norm = train_norm[idx]
        if x_traj_norm.shape[0] < seq_len + 1:
            continue

        steps = min(n_future, max(1, x_traj_norm.shape[0] - seq_len - 1))
        x_init_norm = x_traj_norm[:seq_len]

        x_pred_all_norm = rollout_transformer(
            x_init_norm, steps, encoder, decoder, dyn_model, device
        )
        x_true_seg_norm = x_traj_norm[:seq_len + steps]

        x_pred_all = x_pred_all_norm * state_std + state_mean
        x_true_seg = x_true_seg_norm * state_std + state_mean
        x_dim = x_true_seg.shape[1]

        plt.figure(figsize=(7, 7))
        idx_boundary = seq_len - 1

        if x_dim == 4:
            x1_true, y1_true = x_true_seg[:, 0], x_true_seg[:, 1]
            x2_true, y2_true = x_true_seg[:, 2], x_true_seg[:, 3]
            x1_pred, y1_pred = x_pred_all[:, 0], x_pred_all[:, 1]
            x2_pred, y2_pred = x_pred_all[:, 2], x_pred_all[:, 3]

            plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
            plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

            plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (pred)", linewidth=1.5, color="C0")
            plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (pred)", linewidth=1.5, color="C1")

            plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                        color="C0", marker="o", s=40, label="Start pred M1")
            plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                        color="C1", marker="o", s=40, label="Start pred M2")
        else:
            plt.plot(x_true_seg[:, 0], x_true_seg[:, 1], label="True", linewidth=1.5, color="C0")
            plt.plot(x_pred_all[:, 0], x_pred_all[:, 1], "--", label="Pred", linewidth=1.5, color="C1")
            plt.scatter(x_true_seg[idx_boundary, 0], x_true_seg[idx_boundary, 1],
                        color="C0", marker="o", s=40, label="Prediction start")

        plt.title(f"Transformer train sample {idx} (epoch {epoch:03d})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(epoch_dir, f"transformer_train_sample_{idx}.png")
        plt.savefig(fname)
        plt.close()


def plot_rollout_example_2d_orbit(test_norm, state_mean, state_std, t_eval,
                                  encoder, decoder, dyn_model,
                                  seq_len, n_future, out_dir, device, epoch):
    """
    Use first test trajectory:
      - take an initial window of length seq_len
      - autoregressively roll out n_future steps with Transformer
      - plot 2D orbit (x-y for each mass), true vs rollout
    """
    if test_norm.shape[0] == 0:
        return

    x_traj_norm = test_norm[0]  # (T_total, 4)
    T_total = x_traj_norm.shape[0]
    if T_total < seq_len + n_future + 1:
        n_future = max(1, T_total - seq_len - 1)

    start_idx = 0
    x_init_norm = x_traj_norm[start_idx:start_idx + seq_len]

    # rollout in normalized space
    x_pred_all_norm = rollout_transformer(
        x_init_norm, n_future, encoder, decoder, dyn_model, device
    )  # (seq_len + n_future, 4)

    x_true_seg_norm = x_traj_norm[start_idx:start_idx + seq_len + n_future]

    # denormalize
    x_pred_all = x_pred_all_norm * state_std + state_mean
    x_true_seg = x_true_seg_norm * state_std + state_mean
    x_dim = x_true_seg.shape[1]

    # --- 2D orbit plot ---
    plt.figure(figsize=(7, 7))
    idx_boundary = seq_len - 1

    if x_dim == 4:
        x1_true, y1_true = x_true_seg[:, 0], x_true_seg[:, 1]
        x2_true, y2_true = x_true_seg[:, 2], x_true_seg[:, 3]

        x1_pred, y1_pred = x_pred_all[:, 0], x_pred_all[:, 1]
        x2_pred, y2_pred = x_pred_all[:, 2], x_pred_all[:, 3]

        plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
        plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

        plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (pred)", linewidth=1.5, color="C0")
        plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (pred)", linewidth=1.5, color="C1")

        plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                    color="C0", marker="o", s=40, label="Start pred M1")
        plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                    color="C1", marker="o", s=40, label="Start pred M2")
    else:
        plt.plot(x_true_seg[:, 0], x_true_seg[:, 1], label="True", linewidth=1.5, color="C0")
        plt.plot(x_pred_all[:, 0], x_pred_all[:, 1], "--", label="Pred", linewidth=1.5, color="C1")
        plt.scatter(x_true_seg[idx_boundary, 0], x_true_seg[idx_boundary, 1],
                    color="C0", marker="o", s=40, label="Prediction start")

    plt.title("Transformer rollout: 2D orbit, true vs pred (test example)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"transformer_rollout_orbit_{epoch}.png"))
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
        "--transformer_mode",
        choices=["reuse", "continue", "train"],
        help=(
            "How to handle an existing Transformer: reuse the latest checkpoint, "
            "continue training it, or train a fresh model."
        ),
    )
    parser.add_argument(
        "--reuse_transformer",
        action="store_true",
        help="DEPRECATED: Equivalent to --transformer_mode=reuse.",
    )
    args = parser.parse_args()

    tf_mode = args.transformer_mode
    if tf_mode is None and args.reuse_transformer:
        tf_mode = "reuse"

    cfg     = get_experiment_config(args.experiment)
    ds_cfg  = cfg.dataset
    k_cfg   = cfg.koopman
    tf_cfg  = cfg.transformer

    EXP_DATA_ROOT = f"{cfg.name}/{cfg.DATA_ROOT}"
    EXP_OUTPUT_ROOT = f"{cfg.name}/outputs"
    
    SEQ_LEN    = ds_cfg.SEQ_LEN
    HORIZON    = ds_cfg.HORIZON

    LATENT_DIM = k_cfg.LATENT_DIM
    HIDDEN_DIM = k_cfg.HIDDEN_DIM

    tf_latent_dim = tf_cfg.LATENT_DIM
    if tf_latent_dim != LATENT_DIM:
        print(
            f"Warning: Transformer latent_dim {tf_latent_dim} does not match "
            f"Koopman latent_dim {LATENT_DIM}. Using {LATENT_DIM} for both."
        )
        tf_latent_dim = LATENT_DIM

    NHEAD      = tf_cfg.NHEAD
    NUM_LAYERS = tf_cfg.NUM_LAYERS
    DIM_FEEDFORWARD = tf_cfg.DIM_FEEDFORWARD
    DROPOUT    = tf_cfg.DROPOUT
    BATCH_SIZE = tf_cfg.BATCH_SIZE
    EPOCHS     = tf_cfg.EPOCHS
    LR         = tf_cfg.LR
    X_WEIGHT   = tf_cfg.X_WEIGHT
    TEACHER_FORCING_START = tf_cfg.TEACHER_FORCING_START
    TEACHER_FORCING_END   = tf_cfg.TEACHER_FORCING_END
    LATENT_NOISE_STD      = tf_cfg.LATENT_NOISE_STD

    ROLLOUT_STEPS = tf_cfg.ROLLOUT_STEPS
    
    # 1) Load dataset
    dataset_pattern = f"{cfg.simulation.PROBLEM.replace('-', '_')}_dataset_*.npz"
    ds_file = find_latest_dataset(pattern=dataset_pattern, data_dir=EXP_DATA_ROOT)
    print(f"Loading dataset from: {ds_file}")
    data = np.load(ds_file)

    t_eval     = data["t_eval"]
    state_mean = data["state_mean"]
    state_std  = data["state_std"]

    train_norm = data["train_norm"]   # (N_train, T, 4)
    val_norm   = data["val_norm"]     # (N_val,   T, 4)
    test_norm  = data["test_norm"]    # (N_test,  T, 4)

    N_train, T_total, x_dim = train_norm.shape
    print(f"Train_norm shape: {train_norm.shape}")
    print(f"Val_norm shape:   {val_norm.shape}")
    print(f"Test_norm shape:  {test_norm.shape}")

    max_len_tf = max(T_total + ROLLOUT_STEPS, SEQ_LEN + ROLLOUT_STEPS) + tf_cfg.MAX_LEN_EXTRA

    # 2) Load latest KoopmanAE (from train_koopman_ae.py)
    koopman_dir, koopman_ckpt_path = find_latest_koopman(output_dir=EXP_OUTPUT_ROOT)
    print(f"Loading KoopmanAE from: {koopman_ckpt_path}")
    ckpt = torch.load(koopman_ckpt_path, map_location=DEVICE)

    koopman_model = KoopmanAE(x_dim=x_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
    koopman_model.load_state_dict(ckpt["model_state_dict"])
    koopman_model.to(DEVICE)

    encoder = koopman_model.encoder
    decoder = koopman_model.decoder

    # 3) Build windowed datasets/loaders for Transformer
    train_ds = WindowedSequenceDataset(train_norm, seq_len=SEQ_LEN, horizon=HORIZON)
    val_ds   = WindowedSequenceDataset(val_norm,   seq_len=SEQ_LEN, horizon=HORIZON)
    test_ds  = WindowedSequenceDataset(test_norm,  seq_len=SEQ_LEN, horizon=HORIZON)

    if len(train_ds) == 0:
        raise ValueError(
            f"No Transformer training windows available: trajectory length {T_total} < "
            f"SEQ_LEN+HORIZON ({SEQ_LEN + HORIZON}). "
            "Increase trajectory length or reduce SEQ_LEN/HORIZON in the experiment config."
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Transformer train samples: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Fixed subset of training trajectories for reproducible plotting and evaluation
    train_sample_indices = _fixed_sample_indices(train_norm, num_samples=10, seed=0)

    # 4) Prepare output dir name (creation deferred so reuse scan ignores this run)
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(EXP_OUTPUT_ROOT, f"transformer_{time_tag}")

    def write_info(mode_note: str = ""):
        info_path = os.path.join(out_dir, "info.txt")
        with open(info_path, "w") as f:
            f.write(f"Dataset file: {ds_file}\n")
            f.write(f"KoopmanAE dir: {koopman_dir}\n")
            f.write(f"SEQ_LEN: {SEQ_LEN}\n")
            f.write(f"HORIZON: {HORIZON}\n")
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"LR: {LR}\n")
            f.write(f"LATENT_DIM: {tf_latent_dim}\n")
            f.write(f"HIDDEN_DIM: {HIDDEN_DIM}\n")
            f.write(f"NHEAD: {NHEAD}\n")
            f.write(f"NUM_LAYERS: {NUM_LAYERS}\n")
            f.write(f"DIM_FEEDFORWARD: {DIM_FEEDFORWARD}\n")
            f.write(f"DROPOUT: {DROPOUT}\n")
            f.write(f"ROLLOUT_STEPS: {ROLLOUT_STEPS}\n")
            f.write(f"X_WEIGHT: {X_WEIGHT}\n")
            f.write(f"TEACHER_FORCING_START: {TEACHER_FORCING_START}\n")
            f.write(f"TEACHER_FORCING_END: {TEACHER_FORCING_END}\n")
            f.write(f"LATENT_NOISE_STD: {LATENT_NOISE_STD}\n")
            if mode_note:
                f.write(f"MODE: {mode_note}\n")

    # 5) Initialize (or reuse) Transformer
    best_rollout_loss = float("nan")
    final_epoch = "final"
    reuse_transformer = False
    continue_transformer = False
    resume_optimizer_state_tf = None
    start_epoch_tf = 0
    best_val_from_ckpt_tf = None
    best_rollout_from_ckpt_tf = None
    latest_tf_dir = None
    latest_tf_ckpt_path = None
    latest_tf_err = ""
    ckpt_tf = None
    ckpt_latent_dim = None

    try:
        latest_tf_dir, latest_tf_ckpt_path = find_latest_transformer(output_dir=EXP_OUTPUT_ROOT)
    except FileNotFoundError as e:
        latest_tf_err = str(e)

    if tf_mode is None:
        if latest_tf_ckpt_path:
            print("\nA previously trained Transformer was found:")
            print(f"  {latest_tf_ckpt_path}")
            choice = input("Choose action [r]euse / [c]ontinue / [n]ew (default=n): ").strip().lower()
            if choice == "r":
                tf_mode = "reuse"
            elif choice == "c":
                tf_mode = "continue"
            else:
                tf_mode = "train"
        else:
            tf_mode = "train"

    if tf_mode in {"reuse", "continue"} and not latest_tf_ckpt_path:
        print(f"\n{latest_tf_err or 'No previous Transformer checkpoint found.'} Training a new model instead.\n")
        tf_mode = "train"

    reuse_ckpt_path = None

    if tf_mode in {"reuse", "continue"} and latest_tf_ckpt_path:
        ckpt_tf = torch.load(latest_tf_ckpt_path, map_location=DEVICE)
        ckpt_latent_dim = ckpt_tf.get("model_state_dict", {}).get("input_norm.weight")
        ckpt_latent_dim = ckpt_latent_dim.shape[0] if ckpt_latent_dim is not None else None
        if ckpt_latent_dim is not None and ckpt_latent_dim != tf_latent_dim:
            print(
                f"\nTransformer checkpoint latent_dim {ckpt_latent_dim} "
                f"does not match Koopman latent_dim {tf_latent_dim}. "
                "Training a new model instead.\n"
            )
            tf_mode = "train"
            ckpt_tf = None
            ckpt_latent_dim = None
        elif ckpt_latent_dim is None:
            print("Could not infer latent_dim from Transformer checkpoint; assuming compatibility.")

    os.makedirs(out_dir, exist_ok=True)

    if tf_mode == "reuse" and ckpt_tf is not None:
        print(f"\nReusing existing Transformer checkpoint from: {latest_tf_ckpt_path}")
        shutil.copytree(latest_tf_dir, out_dir, dirs_exist_ok=True)
        reuse_ckpt_path = os.path.join(out_dir, "transformer_best.pt")
        shutil.copy(latest_tf_ckpt_path, reuse_ckpt_path)
        reuse_transformer = True
        write_info(mode_note="reuse")
    elif tf_mode == "continue" and ckpt_tf is not None:
        print(f"\nContinuing Transformer training from: {latest_tf_ckpt_path}")
        shutil.copytree(latest_tf_dir, out_dir, dirs_exist_ok=True)
        reuse_ckpt_path = os.path.join(out_dir, "transformer_best.pt")
        shutil.copy(latest_tf_ckpt_path, reuse_ckpt_path)
        start_epoch_tf = int(ckpt_tf.get("epoch", 0))
        resume_optimizer_state_tf = ckpt_tf.get("optimizer_state_dict")
        best_val_from_ckpt_tf = ckpt_tf.get("val_loss")
        best_rollout_from_ckpt_tf = ckpt_tf.get("rollout_val_loss")
        continue_transformer = True
        write_info(mode_note="continue")
    else:
        tf_mode = "train"
        write_info(mode_note="train")

    if reuse_transformer:
        print(f"Loading Transformer from: {reuse_ckpt_path}")

        ckpt_pe = ckpt_tf["model_state_dict"].get("pos_encoder.pe")
        ckpt_max_len = ckpt_pe.shape[1] if ckpt_pe is not None else 0
        model_max_len = max(max_len_tf, ckpt_max_len)

        dyn_model = LatentTransformer(
            latent_dim=tf_latent_dim,
            nhead=tf_cfg.NHEAD,
            num_layers=tf_cfg.NUM_LAYERS,
            dim_feedforward=tf_cfg.DIM_FEEDFORWARD,
            dropout=tf_cfg.DROPOUT,
            max_len=model_max_len,
        ).to(DEVICE)

        dyn_model.load_state_dict(ckpt_tf["model_state_dict"], strict=False)
        if "encoder_state_dict" in ckpt_tf:
            encoder.load_state_dict(ckpt_tf["encoder_state_dict"], strict=False)
        dyn_model.eval()

        best_val_loss = float(ckpt_tf.get("val_loss", float("nan")))
        best_rollout_loss = float(ckpt_tf.get("rollout_val_loss", float("nan")))
        final_epoch = ckpt_tf.get("epoch", "reused")
        print(f"Reusing Transformer model — skipping training. "
              f"(stored val_loss = {best_val_loss:.4e})\n")
    else:
        print("\nTraining new Transformer model...\n")

        ckpt_pe = ckpt_tf["model_state_dict"].get("pos_encoder.pe") if continue_transformer and ckpt_tf else None
        ckpt_max_len = ckpt_pe.shape[1] if ckpt_pe is not None else 0
        model_max_len = max(max_len_tf, ckpt_max_len)

        dyn_model = LatentTransformer(
            latent_dim=tf_latent_dim,
            nhead=tf_cfg.NHEAD,
            num_layers=tf_cfg.NUM_LAYERS,
            dim_feedforward=tf_cfg.DIM_FEEDFORWARD,
            dropout=tf_cfg.DROPOUT,
            max_len=model_max_len,
        ).to(DEVICE)

        if continue_transformer and ckpt_tf is not None:
            dyn_model.load_state_dict(ckpt_tf["model_state_dict"], strict=False)
            if "encoder_state_dict" in ckpt_tf:
                encoder.load_state_dict(ckpt_tf["encoder_state_dict"], strict=False)

        best_val_loss, best_rollout_loss, history = train_transformer(
            dyn_model, encoder, decoder,
            train_loader, val_loader,
            num_epochs=EPOCHS, lr=LR,
            device=DEVICE, out_dir=out_dir,
            test_norm=test_norm,
            val_norm=val_norm,
            train_norm=train_norm,
            state_mean=state_mean,
            state_std=state_std,
            t_eval=t_eval,
            seq_len=SEQ_LEN,
            rollout_steps=ROLLOUT_STEPS,
            horizon=HORIZON,
            x_weight=X_WEIGHT,
            teacher_forcing_start=TEACHER_FORCING_START,
            teacher_forcing_end=TEACHER_FORCING_END,
            latent_noise_std=LATENT_NOISE_STD,
            grad_clip=tf_cfg.GRAD_CLIP,
            fine_tune_encoder=tf_cfg.FINE_TUNE_ENCODER,
            train_sample_indices=train_sample_indices,
            start_epoch=start_epoch_tf if continue_transformer else 0,
            optimizer_state_dict=resume_optimizer_state_tf if continue_transformer else None,
            best_val_loss_init=best_val_from_ckpt_tf if continue_transformer else None,
            best_rollout_loss_init=best_rollout_from_ckpt_tf if continue_transformer else None,
        )

        tf_ckpt = torch.load(os.path.join(out_dir, "transformer_best.pt"),
                             map_location=DEVICE)
        dyn_model.load_state_dict(tf_ckpt["model_state_dict"])
        if "encoder_state_dict" in tf_ckpt:
            encoder.load_state_dict(tf_ckpt["encoder_state_dict"], strict=False)
        best_rollout_loss = float(tf_ckpt.get("rollout_val_loss", best_rollout_loss))
        dyn_model.to(DEVICE)
        final_epoch = tf_ckpt.get("epoch", "final")

    plot_training_rollout_orbits(
        train_norm=train_norm,
        state_mean=state_mean,
        state_std=state_std,
        encoder=encoder,
        decoder=decoder,
        dyn_model=dyn_model,
        seq_len=SEQ_LEN,
        n_future=ROLLOUT_STEPS,
        out_dir=out_dir,
        device=DEVICE,
        epoch=final_epoch,
        num_samples=10,
        sample_indices=train_sample_indices,
    )

    print(f"Best Transformer val loss: {best_val_loss:.4e}")
    if not math.isnan(best_rollout_loss):
        print(f"Best Transformer long-rollout val loss: {best_rollout_loss:.4e}")

    # 7) One-step MSE in x-space on test set
    test_one_step_mse = one_step_mse_xspace(
        encoder, decoder, dyn_model, test_loader, DEVICE, SEQ_LEN
    )
    print(f"One-step MSE in x-space (test): {test_one_step_mse:.4e}")
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"best_val_loss: {best_val_loss:.6e}\n")
        if not math.isnan(best_rollout_loss):
            f.write(f"best_rollout_val_loss: {best_rollout_loss:.6e}\n")
        f.write(f"test_one_step_mse_xspace: {test_one_step_mse:.6e}\n")

    # 8) Long rollout example on first test trajectory – 2D orbit only
    plot_rollout_example_2d_orbit(
        test_norm=test_norm,
        state_mean=state_mean,
        state_std=state_std,
        t_eval=t_eval,
        encoder=encoder,
        decoder=decoder,
        dyn_model=dyn_model,
        seq_len=SEQ_LEN,
        n_future=ROLLOUT_STEPS,
        out_dir=out_dir,
        device=DEVICE,
        epoch="final"
    )

    print(f"Transformer training and evaluation finished. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
