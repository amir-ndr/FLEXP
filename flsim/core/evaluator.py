"""
core/evaluator.py: Global model evaluator on the held-out test set.

Computes test loss and test accuracy for the current global model.
Called by the Simulator at the end of every evaluate_every rounds.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    """Evaluation metrics for one snapshot of the global model."""
    test_loss: float
    test_accuracy: float   # fraction correct, in [0, 1]


class Evaluator:
    """
    Evaluates the global model on a fixed test dataset.

    This class does NOT:
    - Perform training or gradient computation.
    - Modify the model (runs in torch.no_grad()).
    - Know about communication rounds or simulated time.
    """

    def __init__(self, test_dataset, batch_size: int = 256):
        """
        Args:
            test_dataset: torchvision dataset for evaluation.
            batch_size (int): batch size for evaluation DataLoader.
        """
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

    def evaluate(self, model: nn.Module, device: torch.device = None) -> EvalResult:
        """
        Evaluate model on the full test set.

        Args:
            model (nn.Module): global model to evaluate.
            device (torch.device): device for inference. Defaults to cpu.

        Returns:
            EvalResult with test_loss and test_accuracy.
        """
        if device is None:
            device = torch.device("cpu")

        model = model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += len(y)

        mean_loss = total_loss / max(total_samples, 1)
        accuracy   = total_correct / max(total_samples, 1)
        return EvalResult(test_loss=mean_loss, test_accuracy=accuracy)
