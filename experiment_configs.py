"""
Experiment configuration registry.
Now focused purely on the seq2seq Transformer (no Koopman autoencoder).
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class SimulationConfig:
    PROBLEM: str        # "two_body_problem" or "2d-throw"
    G: float
    T_SPAN: Tuple[float, float]
    NUM_STEPS: int
    NUM_TRAJECTORIES: int
    NUM_TRAJ_OOD: int
    PERTURBATION: float


@dataclass
class DatasetConfig:
    SEQ_LEN: int        # a.k.a. LOOKBACK / window length
    HORIZON: int        # prediction horizon for training
    TRAIN_FRAC: float
    VAL_FRAC: float


@dataclass
class TransformerConfig:
    MODEL_DIM: int
    NHEAD: int
    NUM_LAYERS: int
    DIM_FEEDFORWARD: int
    DROPOUT: float
    LR: float
    BATCH_SIZE: int
    EPOCHS: int
    ROLLOUT_STEPS: int  # long rollout horizon used in eval
    MAX_LEN_EXTRA: int  # extra margin for positional encoding length
    X_WEIGHT: float
    TEACHER_FORCING_START: float
    TEACHER_FORCING_END: float
    INPUT_NOISE_STD: float
    GRAD_CLIP: float = 0.0


@dataclass
class EvalConfig:
    OOD_ROLLOUT_STEPS: int


@dataclass
class ExperimentConfig:
    name: str
    DATA_ROOT: str
    simulation: SimulationConfig
    dataset: DatasetConfig
    transformer: TransformerConfig
    eval: EvalConfig


# ----------------------------------------------------------------
# Define experiments here
# ----------------------------------------------------------------
EXPERIMENTS: Dict[str, ExperimentConfig] = {}

EXPERIMENTS["experiment1_2025-11-28"] = ExperimentConfig(
    name="experiment1_2025-11-28",
    DATA_ROOT="data",
    simulation=SimulationConfig(
        PROBLEM="two_body_problem",
        G=1.0,
        T_SPAN=(0.0, 10.0),
        NUM_STEPS=2000,
        NUM_TRAJECTORIES=200,
        NUM_TRAJ_OOD=3,
        PERTURBATION=0.05,
    ),
    dataset=DatasetConfig(
        SEQ_LEN=500,
        HORIZON=20,
        TRAIN_FRAC=0.7,
        VAL_FRAC=0.15,
    ),
    transformer=TransformerConfig(
        MODEL_DIM=128,
        NHEAD=4,
        NUM_LAYERS=6,
        DIM_FEEDFORWARD=256,
        DROPOUT=0.1,
        LR=1e-3,
        BATCH_SIZE=32,
        EPOCHS=40,
        ROLLOUT_STEPS=40,
        MAX_LEN_EXTRA=20,
        X_WEIGHT=1.5,
        TEACHER_FORCING_START=0.9,
        TEACHER_FORCING_END=0.0,
        INPUT_NOISE_STD=0.01,
    ),
    eval=EvalConfig(
        OOD_ROLLOUT_STEPS=400,
    ),
)

EXPERIMENTS["experiment2_2025-11-28_high-variance"] = ExperimentConfig(
    name="experiment2_2025-11-28_high-variance",
    DATA_ROOT="data",
    simulation=SimulationConfig(
        PROBLEM="two_body_problem",
        G=1.0,
        T_SPAN=(0.0, 5.0),
        NUM_STEPS=2000,
        NUM_TRAJECTORIES=3000,
        NUM_TRAJ_OOD=3,
        PERTURBATION=0.4,
    ),
    dataset=DatasetConfig(
        SEQ_LEN=400,
        HORIZON=15,
        TRAIN_FRAC=0.7,
        VAL_FRAC=0.15,
    ),
    transformer=TransformerConfig(
        MODEL_DIM=128,
        NHEAD=4,
        NUM_LAYERS=6,
        DIM_FEEDFORWARD=256,
        DROPOUT=0.1,
        LR=1e-3,
        BATCH_SIZE=64,
        EPOCHS=12,
        ROLLOUT_STEPS=120,
        MAX_LEN_EXTRA=20,
        X_WEIGHT=1.5,
        TEACHER_FORCING_START=0.9,
        TEACHER_FORCING_END=0.0,
        INPUT_NOISE_STD=0.015,
    ),
    eval=EvalConfig(
        OOD_ROLLOUT_STEPS=400,
    ),
)

EXPERIMENTS["test_experiment"] = ExperimentConfig(
    name="test_experiment",
    DATA_ROOT="data",
    simulation=SimulationConfig(
        PROBLEM="2d-throw",
        G=9.81,
        T_SPAN=(0.0, 2.0),
        NUM_STEPS=500,
        NUM_TRAJECTORIES=200,
        NUM_TRAJ_OOD=6,
        PERTURBATION=0.4,
    ),
    dataset=DatasetConfig(
        SEQ_LEN=20,
        HORIZON=5,
        TRAIN_FRAC=0.7,
        VAL_FRAC=0.15,
    ),
    transformer=TransformerConfig(
        MODEL_DIM=64,
        NHEAD=4,
        NUM_LAYERS=2,
        DIM_FEEDFORWARD=128,
        DROPOUT=0.0,
        LR=1e-3,
        BATCH_SIZE=16,
        EPOCHS=10,
        ROLLOUT_STEPS=20,
        MAX_LEN_EXTRA=10,
        X_WEIGHT=1.0,
        TEACHER_FORCING_START=1.0,
        TEACHER_FORCING_END=0.5,
        INPUT_NOISE_STD=0.0,
    ),
    eval=EvalConfig(
        OOD_ROLLOUT_STEPS=50,
    ),
)

DEFAULT_EXPERIMENT = "experiment1_2025-11-28"


def get_experiment_config(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment config '{name}'")
    return EXPERIMENTS[name]


def get_experiments() -> Dict[str, ExperimentConfig]:
    return EXPERIMENTS


if __name__ == "__main__":
    import json
    configs_as_dict = {name: cfg.__dict__ for name, cfg in EXPERIMENTS.items()}
    print(json.dumps(configs_as_dict, indent=2))
