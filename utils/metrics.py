import scipy
import numpy as np
from torchmetrics.metric import Metric


def brier_scores(labels, probs=None, logits=None):
  """Compute elementwise Brier score.

  Args:
    labels: Tensor of integer labels shape [N1, N2, ...]
    probs: Tensor of categorical probabilities of shape [N1, N2, ..., M].
    logits: If `probs` is None, class probabilities are computed as a softmax
      over these logits, otherwise, this argument is ignored.
  Returns:
    Tensor of shape [N1, N2, ...] consisting of Brier score contribution from
    each element. The full-dataset Brier score is an average of these values.
  """
  assert (probs is None) != (logits is None)
  if probs is None:
    probs = scipy.special.softmax(logits, axis=-1)
  nlabels = probs.shape[-1]
  flat_probs = probs.reshape([-1, nlabels])
  flat_labels = labels.reshape([len(flat_probs)])

  plabel = flat_probs[np.arange(len(flat_labels)), flat_labels]
  out = np.square(flat_probs).sum(axis=-1) - 2 * plabel
  return out.reshape(labels.shape)


class MyAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


class MeanSquaredError(Metric):
    def __init__(
        self,
        squared: bool = True,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs=self.num_outputs)

        self.sum_squared_error += sum_squared_error
        self.total += target.numel()

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)
