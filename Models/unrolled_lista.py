import torch
import torch.nn as nn


class UnrolledLISTA(nn.Module):
    """
    Unrolled neural network (LISTA) for image denoising.
    Each layer corresponds to one ISTA iteration with learned parameters.
    """

    def __init__(self, patch_size=64, num_layers=10):
        super(UnrolledLISTA, self).__init__()

        self.patch_size = patch_size
        self.dim = patch_size * patch_size
        self.num_layers = num_layers

        # Learned linear transformation
        self.W = nn.Linear(self.dim, self.dim, bias=False)

        # Learned thresholds (one per layer)
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_layers)
        ])

    def soft_threshold(self, x, threshold):
        """
        Soft-thresholding operator (proximal operator of l1 norm)
        """
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)

    def forward(self, y):
        """
        Forward pass of the unrolled network
        y : noisy input patch (batch_size, dim)
        """
        # ISTA initialization
        x = y

        # Deep unfolding: one layer = one ISTA iteration
        for k in range(self.num_layers):
            x = self.W(x)
            x = self.soft_threshold(x, self.thresholds[k])

        return x
