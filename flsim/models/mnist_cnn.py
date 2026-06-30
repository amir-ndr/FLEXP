"""
models/mnist_cnn.py: Lightweight CNN for MNIST classification.

Architecture matches the standard FL benchmark model from McMahan et al. (2017):
  Conv(1→32, 5×5) → ReLU → MaxPool(2)
  Conv(32→64, 5×5) → ReLU → MaxPool(2)
  Flatten → Linear(1024→512) → ReLU → Linear(512→10)

Input: (B, 1, 28, 28) — single-channel 28×28 MNIST images
Output: (B, 10) — logits for 10 digit classes
"""

import torch.nn as nn


class MnistCNN(nn.Module):
    """
    Lightweight 2-layer CNN for MNIST.

    This class does NOT:
    - Manage training loops.
    - Load data or apply normalisation.
    - Hold optimizer state.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),    # (B,1,28,28) → (B,32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → (B,32,12,12)
            nn.Conv2d(32, 64, kernel_size=5),    # → (B,64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → (B,64,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # → (B, 1024)
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, 1, 28, 28).

        Returns:
            tensor of shape (B, 10) — unnormalized logits.
        """
        return self.classifier(self.features(x))
