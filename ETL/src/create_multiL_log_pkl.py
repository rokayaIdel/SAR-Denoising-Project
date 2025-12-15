# create_multiL_log_pkl.py

import os
import numpy as np
from PIL import Image
import pickle

EPS = 1e-6

def load_image(path):
    img = Image.open(path).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0
    return img

def to_log_domain(img):
    log_img = np.log(img + EPS)
    log_img = (log_img - log_img.mean()) / (img.std() + EPS)
    return log_img

def list_all_images(root):
    """Return full paths of all .png/.jpg files inside root (recursively)."""
    imgs = []
    exts = (".png", ".jpg", ".jpeg")
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                imgs.append(os.path.join(subdir, f))
    return sorted(imgs)

def build_multiL_pickle(
    patches_root="../data/patches",
    save_path="../data/pickles/patches_multiL_log.pkl"
):

    DATA = {}
    levels = ["L1", "L2", "L4", "L8"]

    for L in levels:
        print(f"[INFO] Processing level {L}...")

        clean_dir = os.path.join(patches_root, L, "clean")
        noisy_dir = os.path.join(patches_root, L, "noisy")

        clean_files = list_all_images(clean_dir)
        noisy_files = list_all_images(noisy_dir)

        if len(clean_files) != len(noisy_files):
            raise ValueError(f"Mismatch in L={L}: {len(clean_files)} clean vs {len(noisy_files)} noisy")

        DATA[L] = {"clean": [], "noisy": []}

        for cf, nf in zip(clean_files, noisy_files):
            clean = load_image(cf)
            noisy = load_image(nf)

            clean_log = to_log_domain(clean)
            noisy_log = to_log_domain(noisy)

            DATA[L]["clean"].append(clean_log)
            DATA[L]["noisy"].append(noisy_log)

        print(f"[OK] Stored {len(clean_files)} log-domain pairs for {L}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(DATA, f)

    print(f"\n[SAVED] Log-domain multi-level pickle → {save_path}")

if __name__ == "__main__":
    build_multiL_pickle()
