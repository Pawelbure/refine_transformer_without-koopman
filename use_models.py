#!/usr/bin/env python
# use_models.py
#
# Script 4:
# - loads latest dataset, KoopmanAE, and Transformer
# - lets the user pick:
#     * a trajectory from train / val / test data
#     * OR a freshly generated trajectory from user-chosen initial values
# - runs Transformer rollout on the chosen time series
# - plots 2D orbits (true vs predicted) and saves them in outputs/use_models_*

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

# Import model classes + rollout helper from the other modules
from train_koopman_ae import KoopmanAE
from train_transformer import LatentTransformer, rollout_transformer

from utils import *

# ============================================================
# Global config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Dynamics for NEW trajectories
# ============================================================
def two_body_rhs(t, y, G, m1, m2):
    """
    y = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y
    rx = x2 - x1
    ry = y2 - y1
    r2 = rx * rx + ry * ry
    r = math.sqrt(r2)
    eps = 1e-9
    r3 = (r2 + eps) ** 1.5

    ax1 = G * m2 * rx / r3
    ay1 = G * m2 * ry / r3
    ax2 = -G * m1 * rx / r3
    ay2 = -G * m1 * ry / r3

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]


def simulate_two_body(m1, m2,
                      x1_0, y1_0, vx1_0, vy1_0,
                      x2_0, y2_0, vx2_0, vy2_0,
                      t_span, num_steps, G):
    from scipy.integrate import solve_ivp

    y0 = [x1_0, y1_0, vx1_0, vy1_0,
          x2_0, y2_0, vx2_0, vy2_0]
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    sol = solve_ivp(
        fun=lambda t, y: two_body_rhs(t, y, G, m1, m2),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    x1 = sol.y[0]
    y1 = sol.y[1]
    x2 = sol.y[4]
    y2 = sol.y[5]

    states = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)  # (T, 4)
    return t_eval, states


def simulate_throw(v0, angle_rad, g, t_span, num_steps, y0=0.0, x0=0.0):
    """Ballistic 2D throw under constant gravity. Returns (T,2) positions."""
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)

    x = x0 + vx * t_eval
    y = y0 + vy * t_eval - 0.5 * g * t_eval ** 2
    states = np.stack([x, y], axis=1).astype(np.float32)
    return t_eval, states


# ============================================================
# Small helpers for user input
# ============================================================
def ask_int(prompt, default):
    s = input(f"{prompt} [default={default}]: ").strip()
    if s == "":
        return default
    return int(s)


def ask_float(prompt, default):
    s = input(f"{prompt} [default={default}]: ").strip()
    if s == "":
        return float(default)
    return float(s)


# ============================================================
# Plotting utilities
# ============================================================
def plot_2d_orbit(true_states, pred_states, seq_len, out_path, title_prefix=""):
    """
    Supports both two-body (4-D) and single projectile (2-D) trajectories.
    seq_len: length of initial window (prediction starts after seq_len-1)
    """
    plt.figure(figsize=(7, 7))

    idx_boundary = seq_len - 1
    if true_states.shape[1] == 4:
        x1_true, y1_true = true_states[:, 0], true_states[:, 1]
        x2_true, y2_true = true_states[:, 2], true_states[:, 3]

        x1_pred, y1_pred = pred_states[:, 0], pred_states[:, 1]
        x2_pred, y2_pred = pred_states[:, 2], pred_states[:, 3]

        plt.plot(x1_true, y1_true, label="Mass 1 (true)", linewidth=1.5, color="C0")
        plt.plot(x2_true, y2_true, label="Mass 2 (true)", linewidth=1.5, color="C1")

        plt.plot(x1_pred, y1_pred, "--", label="Mass 1 (pred)", linewidth=1.5, color="C0")
        plt.plot(x2_pred, y2_pred, "--", label="Mass 2 (pred)", linewidth=1.5, color="C1")

        plt.scatter(x1_true[idx_boundary], y1_true[idx_boundary],
                    color="C0", marker="o", s=40, label="Start pred M1")
        plt.scatter(x2_true[idx_boundary], y2_true[idx_boundary],
                    color="C1", marker="o", s=40, label="Start pred M2")
    else:
        x_true, y_true = true_states[:, 0], true_states[:, 1]
        x_pred, y_pred = pred_states[:, 0], pred_states[:, 1]

        plt.plot(x_true, y_true, label="True", linewidth=1.5, color="C0")
        plt.plot(x_pred, y_pred, "--", label="Predicted", linewidth=1.5, color="C1")
        plt.scatter(x_true[idx_boundary], y_true[idx_boundary],
                    color="C0", marker="o", s=40, label="Prediction start")

    plt.title(f"{title_prefix}2D orbit: true vs Transformer rollout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# Main interaction
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help="Name of experiment configuration to use.",
    )
    args = parser.parse_args()

    cfg    = get_experiment_config(args.experiment)
    ds_cfg = cfg.dataset
    tf_cfg = cfg.transformer
    k_cfg  = cfg.koopman
    sim_cfg = cfg.simulation
    
    EXP_DATA_ROOT = f"{cfg.name}/{cfg.DATA_ROOT}"
    EXP_OUTPUT_ROOT = f"{cfg.name}/outputs"
    
    G              = sim_cfg.G
    T_SPAN         = sim_cfg.T_SPAN
    NUM_STEPS      = sim_cfg.NUM_STEPS

    LATENT_DIM = k_cfg.LATENT_DIM
    HIDDEN_DIM = k_cfg.HIDDEN_DIM

    tf_latent_dim = tf_cfg.LATENT_DIM
    if tf_latent_dim != LATENT_DIM:
        print(
            f"Warning: Transformer latent_dim {tf_latent_dim} does not match "
            f"Koopman latent_dim {LATENT_DIM}. Using {LATENT_DIM} for inference."
        )
        tf_latent_dim = LATENT_DIM

    SEQ_LEN     = ds_cfg.SEQ_LEN
    HORIZON     = ds_cfg.HORIZON
    BATCH_SIZE  = k_cfg.BATCH_SIZE
    EPOCHS      = k_cfg.EPOCHS
    LR          = k_cfg.LR
    K_MAX       = k_cfg.K_MAX
    KOOPMAN_LAMBDA = k_cfg.KOOPMAN_LAMBDA
    
    DEFAULT_SEQ_LEN   = ds_cfg.SEQ_LEN
    DEFAULT_N_FUTURE  = tf_cfg.ROLLOUT_STEPS
    
    # --------------------------------------------------------
    # 1) Load dataset
    # --------------------------------------------------------
    ds_file = find_latest_dataset(data_dir=EXP_DATA_ROOT)
    print(f"Loading dataset from: {ds_file}")
    data = np.load(ds_file)

    t_eval     = data["t_eval"]
    state_mean = data["state_mean"]
    state_std  = data["state_std"]
    problem = str(data.get("problem", "two_body_problem"))

    train_norm = data["train_norm"]
    val_norm   = data["val_norm"]
    test_norm  = data["test_norm"]

    print(f"Train_norm shape: {train_norm.shape}")
    print(f"Val_norm shape:   {val_norm.shape}")
    print(f"Test_norm shape:  {test_norm.shape}")

    _, T_total, x_dim = train_norm.shape
    max_len_tf = max(T_total + tf_cfg.ROLLOUT_STEPS, DEFAULT_SEQ_LEN + DEFAULT_N_FUTURE) + tf_cfg.MAX_LEN_EXTRA

    # --------------------------------------------------------
    # 2) Load KoopmanAE
    # --------------------------------------------------------
    koopman_dir, koopman_ckpt_path = find_latest_koopman(output_dir=EXP_OUTPUT_ROOT)
    print(f"Loading KoopmanAE from: {koopman_ckpt_path}")
    ckpt_ae = torch.load(koopman_ckpt_path, map_location=DEVICE)

    koopman_model = KoopmanAE(x_dim=x_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
    koopman_model.load_state_dict(ckpt_ae["model_state_dict"])
    koopman_model.to(DEVICE)
    koopman_model.eval()
    encoder = koopman_model.encoder
    decoder = koopman_model.decoder

    # --------------------------------------------------------
    # 3) Load Transformer
    # --------------------------------------------------------
    tf_dir, tf_ckpt_path = find_latest_transformer(output_dir=EXP_OUTPUT_ROOT)
    print(f"Loading Transformer from: {tf_ckpt_path}")
    ckpt_tf = torch.load(tf_ckpt_path, map_location=DEVICE)

    ckpt_pe = ckpt_tf["model_state_dict"].get("pos_encoder.pe")
    ckpt_max_len = ckpt_pe.shape[1] if ckpt_pe is not None else 0
    model_max_len = max(max_len_tf, ckpt_max_len)

    ckpt_latent_dim = ckpt_tf.get("model_state_dict", {}).get("input_norm.weight")
    ckpt_latent_dim = ckpt_latent_dim.shape[0] if ckpt_latent_dim is not None else None
    if ckpt_latent_dim is not None and ckpt_latent_dim != tf_latent_dim:
        raise RuntimeError(
            f"Transformer checkpoint latent_dim {ckpt_latent_dim} does not match "
            f"Koopman latent_dim {tf_latent_dim}. Retrain the transformer with a "
            "matching latent dimension."
        )

    dyn_model = LatentTransformer(
        latent_dim=tf_latent_dim,
        nhead=tf_cfg.NHEAD,
        num_layers=tf_cfg.NUM_LAYERS,
        dim_feedforward=tf_cfg.DIM_FEEDFORWARD,
        dropout=tf_cfg.DROPOUT,
        max_len=model_max_len,
    ).to(DEVICE)
    dyn_model.load_state_dict(ckpt_tf["model_state_dict"], strict=False)
    dyn_model.eval()

    # --------------------------------------------------------
    # 4) Prepare output dir for this script
    # --------------------------------------------------------
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(EXP_OUTPUT_ROOT, f"use_models_{time_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # 5) User interaction loop
    # --------------------------------------------------------
    print("\n=== Use Models ===")
    print("Choose source of time series:")
    print("  1) Training trajectory")
    print("  2) Validation trajectory")
    print("  3) Test trajectory")
    print("  4) New simulated trajectory (user-defined initial conditions)")

    choice = input("Your choice [1/2/3/4]: ").strip()
    if choice not in {"1", "2", "3", "4"}:
        print("Invalid choice. Exiting.")
        return

    # Set defaults
    seq_len = ask_int("Initial window length (seq_len)", DEFAULT_SEQ_LEN)
    n_future = ask_int("Number of rollout steps (n_future)", DEFAULT_N_FUTURE)

    if seq_len + n_future > max_len_tf:
        n_future = max_len_tf - seq_len
        print(f"Clipping n_future to {n_future} to respect model max_len={max_len_tf}.")
        
    # ------------------ Case 1–3: existing dataset ------------------
    if choice in {"1", "2", "3"}:
        if choice == "1":
            src_name = "train"
            data_split = train_norm
        elif choice == "2":
            src_name = "val"
            data_split = val_norm
        else:
            src_name = "test"
            data_split = test_norm

        N_split, T_total, _ = data_split.shape
        print(f"\nSelected split: {src_name}, #trajectories = {N_split}, T = {T_total}")

        traj_idx = ask_int("Trajectory index", 0)
        traj_idx = max(0, min(traj_idx, N_split - 1))

        x_traj_norm = data_split[traj_idx]  # (T, 4)

        # seq_len / n_future adjustments
        seq_len = ask_int("Initial window length (seq_len)", DEFAULT_SEQ_LEN)
        n_future = ask_int("Number of rollout steps (n_future)", DEFAULT_N_FUTURE)

        T_total = x_traj_norm.shape[0]
        if T_total < seq_len + n_future + 1:
            n_future = max(1, T_total - seq_len - 1)
            print(f"Adjusted n_future to {n_future} due to limited T.")

        start_idx = ask_int("Start index in trajectory", 0)
        if start_idx + seq_len + n_future > T_total:
            start_idx = 0
            print("Start index too large, resetting to 0.")

        x_init_norm = x_traj_norm[start_idx:start_idx + seq_len]
        x_true_seg_norm = x_traj_norm[start_idx:start_idx + seq_len + n_future]

    # ------------------ Case 4: new simulated trajectory ------------------
    else:
        if problem == "two_body_problem":
            print("\nSimulating NEW two-body trajectory with user-defined initial values.")
            print("Press Enter to accept default in brackets.")

            # masses
            m1 = ask_float("Mass m1", 1.0)
            m2 = ask_float("Mass m2", 1.0)

            # positions
            r_default = 1.0
            r = ask_float("Orbital distance scale r (positions at ±r/2 on x-axis)", r_default)
            x1_0 = -r / 2.0
            x2_0 =  r / 2.0
            y1_0 = 0.0
            y2_0 = 0.0

            # velocities: use circular velocity as baseline
            v_circ = math.sqrt(G * (m1 + m2) / (4.0 * r))
            factor = ask_float("Velocity factor (1.0 = near circular orbit)", 1.0)
            vy1_0 =  v_circ * factor
            vy2_0 = -v_circ * factor
            vx1_0 = 0.0
            vx2_0 = 0.0

            print(f"\nUsing initial conditions:")
            print(f"  m1={m1}, m2={m2}")
            print(f"  (x1,y1)=({x1_0},{y1_0}), (x2,y2)=({x2_0},{y2_0})")
            print(f"  (vx1,vy1)=({vx1_0},{vy1_0}), (vx2,vy2)=({vx2_0},{vy2_0})")

            t_eval_new, states_new = simulate_two_body(
                m1, m2,
                x1_0, y1_0, vx1_0, vy1_0,
                x2_0, y2_0, vx2_0, vy2_0,
                t_span=T_SPAN,
                num_steps=NUM_STEPS,
                G=G,
            )  # states_new: (T_new, 4)
        else:
            print("\nSimulating NEW 2D throw trajectory with user-defined initial values.")
            print("Press Enter to accept default in brackets.")
            v0 = ask_float("Launch speed", 8.0)
            angle_deg = ask_float("Launch angle (deg)", 55.0)
            height0 = ask_float("Initial height", 0.5)

            t_eval_new, states_new = simulate_throw(
                v0=v0,
                angle_rad=math.radians(angle_deg),
                g=G,
                t_span=T_SPAN,
                num_steps=NUM_STEPS,
                y0=height0,
                x0=0.0,
            )  # (T_new, 2)

        # normalize with training stats
        x_traj_norm = (states_new - state_mean) / state_std

        T_total = x_traj_norm.shape[0]
        print(f"New trajectory length T_new = {T_total}")

        seq_len = ask_int("Initial window length (seq_len)", DEFAULT_SEQ_LEN)
        n_future = ask_int("Number of rollout steps (n_future)", DEFAULT_N_FUTURE)

        if T_total < seq_len + n_future + 1:
            n_future = max(1, T_total - seq_len - 1)
            print(f"Adjusted n_future to {n_future} due to limited T_new.")

        start_idx = ask_int("Start index in trajectory", 0)
        if start_idx + seq_len + n_future > T_total:
            start_idx = 0
            print("Start index too large, resetting to 0.")

        x_init_norm = x_traj_norm[start_idx:start_idx + seq_len]
        x_true_seg_norm = x_traj_norm[start_idx:start_idx + seq_len + n_future]

        # override t_eval for plotting (use new one)
        t_eval = t_eval_new

    # --------------------------------------------------------
    # 6) Run Transformer rollout on the chosen normalized series
    # --------------------------------------------------------
    x_pred_all_norm = rollout_transformer(
        x_init_norm, n_future, encoder, decoder, dyn_model, DEVICE
    )  # (seq_len + n_future, 4)

    # denormalize for plotting
    x_pred_all = x_pred_all_norm * state_std + state_mean
    x_true_seg = x_true_seg_norm * state_std + state_mean

    # --------------------------------------------------------
    # 7) Compute simple RMSE on predicted part (positions)
    # --------------------------------------------------------
    pred_pos_future = x_pred_all[seq_len:, :]
    true_pos_future = x_true_seg[seq_len:, :]
    rmse = np.sqrt(np.mean((pred_pos_future - true_pos_future) ** 2))
    print(f"\nRMSE over rollout (all 4 components, predicted part only): {rmse:.4e}")

    # --------------------------------------------------------
    # 8) Plot 2D orbit true vs predicted
    # --------------------------------------------------------
    if choice in {"1", "2", "3"}:
        src_label = { "1": "train", "2": "val", "3": "test" }[choice]
        title_prefix = f"{src_label} traj #{traj_idx}, "
        orbit_path = os.path.join(out_dir, f"orbit_{src_label}_traj{traj_idx}.png")
    else:
        title_prefix = "new simulation, "
        orbit_path = os.path.join(out_dir, "orbit_new_simulation.png")

    plot_2d_orbit(
        true_states=x_true_seg,
        pred_states=x_pred_all,
        seq_len=seq_len,
        out_path=orbit_path,
        title_prefix=title_prefix,
    )

    print(f"Saved 2D orbit plot to: {orbit_path}")
    print(f"Session outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
