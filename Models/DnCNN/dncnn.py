# dncnn.py
# Architecture DnCNN + chargement des poids préentraînés + outils de freeze

import torch
import torch.nn as nn
from collections import OrderedDict


class DnCNN(nn.Module):
    """
    Implémentation PyTorch du DnCNN (17 couches par défaut).
    Le réseau apprend à prédire le bruit → clean = input - noise.
    """

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True):
        super(DnCNN, self).__init__()

        layers = []

        # 1️⃣ Première couche (Conv + ReLU)
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 2️⃣ Couches 2..(depth-1) : Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # 3️⃣ Dernière couche : reconstruit le bruit
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Le réseau apprend à prédire le bruit → clean = x - noise
        """
        noise = self.dncnn(x)
        return x - noise



# ---------------------------------------------------------------------
# CHARGEMENT DES POIDS PRÉENTRAÎNÉS
# ---------------------------------------------------------------------

def load_pretrained_dncnn_s(model):
    """
    Charge les poids préentraînés du DnCNN-S.
    Ton fichier de poids téléchargé : Models/pretrained/net.pth
    """

    weight_path = "../pretrained/net.pth"   # <-- CORRECTION ICI

    try:
        state_dict = torch.load(weight_path, map_location="cpu")

        # Certains poids ont un prefixe 'module.' → on le retire
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        print("[OK] Poids préentraînés DnCNN-S chargés.")
    except Exception as e:
        print("[WARNING] Impossible de charger les poids :", e)
        print("→ Le modèle démarre avec une initialisation aléatoire.")

    return model



# ---------------------------------------------------------------------
# FONCTIONS POUR LE FINE-TUNING
# ---------------------------------------------------------------------

def freeze_first_layers(model, num_layers_to_freeze):
    """
    Gèle les premières couches du réseau.
    Exemple : freeze_first_layers(model, 12)
    """

    count = 0
    for layer in model.dncnn.children():
        if count < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        count += 1

    print(f"[INFO] {num_layers_to_freeze} premières couches gelées.")



def count_trainable(model):
    """
    Affiche le nombre de paramètres totaux et entraînables.
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[INFO] Paramètres totaux     : {total:,}")
    print(f"[INFO] Paramètres entraînables : {trainable:,}")

    return total, trainable



# ---------------------------------------------------------------------
# TEST RAPIDE
# ---------------------------------------------------------------------

if __name__ == "__main__":
    model = DnCNN()
    x = torch.randn(1, 1, 64, 64)
    y = model(x)
    print("Output shape:", y.shape)
