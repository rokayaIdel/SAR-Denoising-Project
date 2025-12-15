# train_finetune.py
# Fine-tuning du modèle DnCNN-S (blind denoiser) sur vos patches log-domain

import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Pour importer le modèle depuis /Models
sys.path.append(os.path.abspath("../../Models"))

from dncnn import DnCNN, load_pretrained_dncnn_s, freeze_first_layers
from dataset_log_multiL import SARDenoiseMultiL_Log


# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "batch_size": 16,
    "epochs": 5,                       # safe pour CPU
    "learning_rate": 1e-4,
    "num_workers": 0,                  # mettre >0 si GPU
    "freeze_layers": 12,               # on fine-tune seulement les couches finales
    "checkpoint_path": "../checkpoints/dncnn_finetuned.pth",
    "pkl_path": "../data/pickles/patches_multiL_log.pkl"
}


# ----------------------------------------------------------------------
# 2. TRAINING FUNCTION
# ----------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for i, (noisy, clean) in enumerate(dataloader):
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f"  Batch {i}/{len(dataloader)} — Loss: {loss.item():.6f}")

    avg_loss = running_loss / len(dataloader)
    print(f"[EPOCH {epoch}] Avg Loss = {avg_loss:.6f}")
    return avg_loss


# ----------------------------------------------------------------------
# 3. MAIN TRAINING LOOP
# ----------------------------------------------------------------------
def finetune_dncnn():

    print("\n==============================")
    print("   FINE-TUNING DnCNN-S")
    print("==============================\n")

    # --- Load dataset ---
    print("[INFO] Loading dataset...")
    dataset = SARDenoiseMultiL_Log(CONFIG["pkl_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    # --- Load pretrained model ---
    model = DnCNN(depth=17, n_channels=64, image_channels=1).to(DEVICE)
    model = load_pretrained_dncnn_s(model)

    # --- Freeze early layers, fine-tune last ones ---
    freeze_first_layers(model, CONFIG["freeze_layers"])

    # Loss + optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )

    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    # --- Create checkpoint folder if not exists ---
    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)

    # --- Training loop ---
    best_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            epoch
        )

        scheduler.step()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"[SAVE] New best model saved at epoch {epoch}\n")

    print("\n==============================")
    print("   TRAINING FINISHED 🎉")
    print("==============================")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint saved to: {CONFIG['checkpoint_path']}")


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    finetune_dncnn()
