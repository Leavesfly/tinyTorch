"""损失函数模块。

Author: TinyAI Team
"""

from tinytorch.ml.losses.loss import Loss
from tinytorch.ml.losses.mse import MSELoss
from tinytorch.ml.losses.crossentropy import CrossEntropyLoss, BCELoss

__all__ = [
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
]
