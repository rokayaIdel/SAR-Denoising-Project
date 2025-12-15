"""
Training script for the LISTA-based unrolled neural network.
This script is independent from other models in the repository.
"""

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Models.lunrolled_lista import UnrolledLISTA


class SARPatchDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        self.noisy = data["noisy"]
        self.clean = data["clean"]

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        y = torch.tensor(self.noisy[idx], dtype=torch.float32).view(-1)
        x = torch.tensor(self.clean[idx], dtype=torch.float32).view(-1)
        return y, x


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SARPatchDataset("data/pickles/patches.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UnrolledLISTA(patch_size=64, num_layers=10).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0.0
        for y, x_gt in loader:
            y, x_gt = y.to(device), x_gt.to(device)

            optimizer.zero_grad()
            x_pred = model(y)
            loss = criterion(x_pred, x_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "lista_L3.pth")


if __name__ == "__main__":
    train()
