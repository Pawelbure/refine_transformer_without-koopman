#!/usr/bin/env python
# generate_two_body_datasets.py
#
# Pre-analysis script:
# - simulate 2-body trajectories
# - build train/val/test splits
# - normalize data
# - save everything into "data/" for later loading
# - create 2D orbit plots for sample train/val/test trajectories

import os
import math
import json
from datetime import datetime

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dataclasses import asdict

from experiment_configs import get_experiment_config, DEFAULT_EXPERIMENT

# ============================================================
# Global config
# ============================================================
SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment",
    type=str,
    default=DEFAULT_EXPERIMENT,
    help="Name of experiment configuration to use.",
)
args = parser.parse_args()

cfg = get_experiment_config(args.experiment)

EXP_DATA_ROOT = f"{cfg.name}/{cfg.DATA_ROOT}"

# Path to persist the full experiment configuration for reproducibility
EXPERIMENT_CFG_PATH = os.path.join(cfg.name, "experiment_config.json")

sim_cfg = cfg.simulation
ds_cfg  = cfg.dataset

PROBLEM         = sim_cfg.PROBLEM
G               = sim_cfg.G
T_SPAN          = sim_cfg.T_SPAN
NUM_STEPS       = sim_cfg.NUM_STEPS
NUM_TRAJECTORIES = sim_cfg.NUM_TRAJECTORIES
NUM_TRAJ_OOD    = sim_cfg.NUM_TRAJ_OOD
PERTURBATION    = sim_cfg.PERTURBATION

SEQ_LEN   = ds_cfg.SEQ_LEN
HORIZON   = ds_cfg.HORIZON
TRAIN_FRAC = ds_cfg.TRAIN_FRAC
VAL_FRAC   = ds_cfg.VAL_FRAC

# ============================================================
# Dynamics and simulation
# ============================================================
def two_body_rhs(t, y, G, m1, m2):
    """
    State y = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
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


def simulate_two_body(m1, m2, x1_0, y1_0, x2_0, y2_0,
                      vx1_0, vy1_0, vx2_0, vy2_0,
                      t_span, num_steps):
    """
    Returns:
        t_eval: (T,)
        states: (T, 4) with [x1, y1, x2, y2]
    """
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

    # state: [x1, y1, x2, y2]
    states = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return t_eval, states


def simulate_throw(v0, angle_rad, g, t_span, num_steps, y0=0.0, x0=0.0):
    """
    Simple ballistic motion with constant gravity.
    Returns positions only: (T, 2) -> [x, y].
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)

    x = x0 + vx * t_eval
    y = y0 + vy * t_eval - 0.5 * g * t_eval ** 2
    states = np.stack([x, y], axis=1).astype(np.float32)
    return t_eval, states


def generate_trajectories(num_trajectories, t_span, num_steps, rng):
    """
    Generate multiple trajectories for the selected PROBLEM.
    Returns:
        t_eval: (T,)
        trajectories: list of length num_trajectories, each (T, x_dim)
    """
    if PROBLEM == "two_body_problem":
        trajectories = []
        m1, m2 = 1.0, 1.0  # fixed masses

        for _ in range(num_trajectories):
            # symmetric initial positions
            r = 1.0
            x1_0, y1_0 = -r / 2.0, 0.0
            x2_0, y2_0 =  r / 2.0, 0.0

            # circular orbit baseline speed
            v_base = math.sqrt(G * (m1 + m2) / (4 * r))
            # small random perturbation
            eps_v = PERTURBATION * v_base

            vx1_0 = 0.0
            vy1_0 =  v_base + rng.uniform(-eps_v, eps_v)

            vx2_0 = 0.0
            vy2_0 = -v_base + rng.uniform(-eps_v, eps_v)

            t_eval, states = simulate_two_body(
                m1, m2,
                x1_0, y1_0, x2_0, y2_0,
                vx1_0, vy1_0, vx2_0, vy2_0,
                t_span=t_span,
                num_steps=num_steps,
            )
            trajectories.append(states)  # (T, 4)
        return t_eval, trajectories

    if PROBLEM == "2d-throw":
        trajectories = []
        # Base initial speed and launch angle (radians)
        v_base = 8.0
        angle_base = math.radians(55.0)

        for _ in range(num_trajectories):
            dv = rng.uniform(-PERTURBATION, PERTURBATION) * v_base
            dtheta = rng.uniform(-PERTURBATION, PERTURBATION) * angle_base
            v0 = max(0.5, v_base + dv)
            angle = angle_base + dtheta

            t_eval, states = simulate_throw(
                v0=v0,
                angle_rad=angle,
                g=G,
                t_span=t_span,
                num_steps=num_steps,
                y0=0.5,
                x0=0.0,
            )
            trajectories.append(states)  # (T, 2)
        return t_eval, trajectories

    raise ValueError(f"Unsupported PROBLEM '{PROBLEM}'")


# ============================================================
# Splitting & plotting
# ============================================================
def split_indices(n_total, train_frac, val_frac):
    n_train = int(train_frac * n_total)
    n_val   = int(val_frac * n_total)
    n_test  = n_total - n_train - n_val

    indices = np.arange(n_total)
    # (Optional) could shuffle here, but keep deterministic ordering for now
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def plot_orbit_2d(states, title, out_path):
    """
    2D orbit plot:
      - two_body_problem: (x1, y1) and (x2, y2)
      - 2d-throw: (x, y) for a single projectile
    states: (T, 2 or 4) in physical space
    """
    plt.figure(figsize=(6, 6))

    if states.shape[1] == 4:
        x1 = states[:, 0]
        y1 = states[:, 1]
        x2 = states[:, 2]
        y2 = states[:, 3]

        plt.plot(x1, y1, label="Mass 1", linewidth=1.5)
        plt.plot(x2, y2, label="Mass 2", linewidth=1.5)
        plt.scatter(x1[0], y1[0], color="C0", marker="o", s=40, label="Start M1")
        plt.scatter(x2[0], y2[0], color="C1", marker="o", s=40, label="Start M2")
    else:
        x = states[:, 0]
        y = states[:, 1]
        plt.plot(x, y, label="Projectile", linewidth=1.5, color="C0")
        plt.scatter(x[0], y[0], color="C0", marker="o", s=40, label="Start")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(EXP_DATA_ROOT, exist_ok=True)

    # Persist the full experiment configuration for this run
    cfg_dict = asdict(cfg)
    os.makedirs(os.path.dirname(EXPERIMENT_CFG_PATH), exist_ok=True)
    with open(EXPERIMENT_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    rng = np.random.default_rng(SEED)

    # 1) Simulate trajectories
    print(f"Simulating trajectories for {PROBLEM}...")
    t_eval, trajectories = generate_trajectories(
        NUM_TRAJECTORIES, T_SPAN, NUM_STEPS, rng
    )
    dt = float(t_eval[1] - t_eval[0])
    print(f"Generated {len(trajectories)} trajectories, each of length {len(t_eval)}, dt={dt:.4f}")

    # Stack into array: (N_traj, T, x_dim)
    trajectories_raw = np.stack(trajectories, axis=0).astype(np.float32)
    x_dim = trajectories_raw.shape[-1]

    # 2) Compute global mean/std over all trajectories (over time & trajs)
    all_states = trajectories_raw.reshape(-1, x_dim)  # (N_traj * T, x_dim)
    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std  = all_states.std(axis=0).astype(np.float32) + 1e-8

    print("State mean:", state_mean)
    print("State std: ", state_std)

    # 3) Normalize trajectories
    trajectories_norm = (trajectories_raw - state_mean) / state_std  # (N, T, x_dim)

    # 4) Split over trajectories into train/val/test
    n_total = trajectories_raw.shape[0]
    train_idx, val_idx, test_idx = split_indices(
        n_total, TRAIN_FRAC, VAL_FRAC
    )
    print(f"Traj split -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_raw = trajectories_raw[train_idx]  # (N_train, T, x_dim)
    val_raw   = trajectories_raw[val_idx]
    test_raw  = trajectories_raw[test_idx]

    train_norm = trajectories_norm[train_idx]
    val_norm   = trajectories_norm[val_idx]
    test_norm  = trajectories_norm[test_idx]

    # 5) Save everything into data/ as a single npz
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    problem_tag = PROBLEM.replace("-", "_")
    ds_path = os.path.join(EXP_DATA_ROOT, f"{problem_tag}_dataset_{time_tag}.npz")

    np.savez(
        ds_path,
        t_eval=t_eval,
        state_mean=state_mean,
        state_std=state_std,
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        train_norm=train_norm,
        val_norm=val_norm,
        test_norm=test_norm,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        problem=PROBLEM,
    )

    print(f"Saved dataset to: {ds_path}")

    # 6) 2D orbit plots for one sample from train/val/test (using raw / denormalized)
    plots_dir = os.path.join(EXP_DATA_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # helper: pick trajectory if split non-empty
    if train_raw.shape[0] > 0:
        states = train_raw[0]  # (T,4), already physical units
        out_path = os.path.join(plots_dir, "train_sample_orbit.png")
        plot_orbit_2d(states, "Sample TRAIN orbit", out_path)

    if val_raw.shape[0] > 0:
        states = val_raw[0]
        out_path = os.path.join(plots_dir, "val_sample_orbit.png")
        plot_orbit_2d(states, "Sample VAL orbit", out_path)

    if test_raw.shape[0] > 0:
        states = test_raw[0]
        out_path = os.path.join(plots_dir, "test_sample_orbit.png")
        plot_orbit_2d(states, "Sample TEST orbit", out_path)

    print(f"Saved orbit plots to: {plots_dir}")


if __name__ == "__main__":
    main()