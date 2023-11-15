from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .functional import label_smoothed_nll_loss

__all__ = ["SoftCrossEntropyLoss"]


"""
SoftCrossEntropyLoss是一个基于nn.CrossEntropyLoss的损失函数，它支持标签平滑化，
即在目标标签中使用平滑化的标签分布而不是独热编码的标签分布。
这可以防止模型过度自信和过拟合，以及提高模型的泛化性能。
smooth_factor参数控制平滑化的程度，当它为0时不进行平滑化，当它为1时使用均匀分布进行平滑化。
ignore_index参数用于指定在计算损失时要忽略的标签的索引。
reduction参数指定要如何计算损失的总和或平均值。
该损失函数的输入包括模型的输出和目标标签，输出是计算出的损失值。
"""
class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
