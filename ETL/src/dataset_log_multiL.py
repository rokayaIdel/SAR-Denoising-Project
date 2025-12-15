import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class SARDenoiseMultiL_Log(Dataset):
    """
    Loads the log-domain multi-level noise dataset from patches_multiL_log.pkl
    and returns (noisy, clean).
    """

    def __init__(self, pkl_path, levels=None):
        with open(pkl_path, "rb") as f:
            DATA = pickle.load(f)

        # if no specific levels → use all
        if levels is None:
            levels = list(DATA.keys())

        pairs = []
        for L in levels:
            clean_list = DATA[L]["clean"]
            noisy_list = DATA[L]["noisy"]

            for c, n in zip(clean_list, noisy_list):
                c = np.expand_dims(c.astype("float32"), 0)  # (1,H,W)
                n = np.expand_dims(n.astype("float32"), 0)
                pairs.append((n, c))

        self.pairs = pairs
        print(f"[INFO] Loaded {len(self.pairs)} log-domain pairs from {levels}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.pairs[idx][0]),
            torch.tensor(self.pairs[idx][1])
        )
