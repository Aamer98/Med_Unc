import torch
import torch.nn.functional as F
from torchmetrics import Metric


def brier_score(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Brier Score for multi-class classification.

    Args:
        probs: (batch_size, num_classes) predicted probabilities.
        y: (batch_size,) integer labels.

    Returns:
        A scalar tensor representing the mean Brier Score over the batch.
    """
    # Convert integer labels to one-hot vectors
    y_onehot = F.one_hot(y, num_classes=probs.size(-1)).float()   # (batch_size, num_classes)

    # Compute the squared difference (p_i - y_i)^2
    diff = probs - y_onehot
    squared_diff = diff.pow(2)  # (batch_size, num_classes)

    # Sum over classes, then average over batch
    return squared_diff.sum(dim=-1).mean()


class BrierScore(Metric):
    """Computes the Brier Score for multi-class classification.
    
    Brier Score = 1/N * sum((p_i - y_i)^2) over all samples (and summed across classes).
    """
    is_differentiable = False
    higher_is_better = False

    def __init__(self, num_classes: int, dist_sync_on_step: bool = False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.num_classes = num_classes
        
        # We will accumulate the total squared error and total number of samples
        self.add_state("sum_squared_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: Probabilities or logits of shape (batch_size, num_classes)
            target: Ground-truth integer labels of shape (batch_size,)
        """
        # Make sure preds are probabilities.
        # If preds are logits, apply softmax first.
        if preds.dim() == 2 and preds.size(1) == self.num_classes:
            # If these are logits, convert to probabilities with softmax.
            # Otherwise, if already probabilities, skip this step.
            preds = F.softmax(preds, dim=-1)

        # Convert target to one-hot
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()  # (batch_size, num_classes)

        # Compute (p_i - y_i)^2 for each sample and each class, then sum over classes
        squared_diff = (preds - target_onehot).pow(2).sum(dim=-1)  # shape: (batch_size,)

        # Accumulate the results
        self.sum_squared_diff += squared_diff.sum()
        self.total_samples += target.numel()

    def compute(self):
        # Brier Score is the mean of these squared differences
        return self.sum_squared_diff / self.total_samples