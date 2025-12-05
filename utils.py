import os
import glob

# ============================================================
# Helper: find latest dataset / KoopmanAE
# ============================================================
def find_latest_dataset(pattern="*_dataset_*.npz", data_dir=""):
    search_pattern = os.path.join(data_dir, pattern)
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(
            f"No dataset files matching {search_pattern}. "
            f"Run your data-generation script first."
        )
    files = sorted(files)
    return files[-1]

def find_latest_koopman(pattern="koopman_ae_*", ckpt_name="koopman_ae_best.pt", output_dir=""):
    search_pattern = os.path.join(output_dir, pattern)
    dirs = glob.glob(search_pattern)
    if not dirs:
        raise FileNotFoundError(
            f"No KoopmanAE directories matching {search_pattern}. "
            f"Run train_koopman_ae.py first."
        )
    dirs = sorted(dirs)
    last_dir = dirs[-1]
    ckpt_path = os.path.join(last_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Latest KoopmanAE dir found ({last_dir}) "
            f"but checkpoint {ckpt_name} missing."
        )
    return last_dir, ckpt_path

def find_latest_transformer(pattern="transformer_*", ckpt_name="transformer_best.pt", output_dir=""):
    search_pattern = os.path.join(output_dir, pattern)
    dirs = sorted(glob.glob(search_pattern))
    completed_dirs = [d for d in dirs if os.path.exists(os.path.join(d, ckpt_name))]

    if not completed_dirs:
        skipped = [d for d in dirs if not os.path.exists(os.path.join(d, ckpt_name))]
        skipped_msg = f" Skipped dirs without {ckpt_name}: {skipped}" if skipped else ""
        raise FileNotFoundError(
            f"No Transformer checkpoints found under {search_pattern}.{skipped_msg} "
            "Run train_transformer.py to create one."
        )

    last_dir = completed_dirs[-1]
    ckpt_path = os.path.join(last_dir, ckpt_name)
    return last_dir, ckpt_path
