"""交叉熵损失函数。

Author: TinyAI Team
"""

from tinytorch.ml.losses.loss import Loss
from tinytorch.autograd import Tensor
from tinytorch.autograd.ops.loss import CrossEntropy as CrossEntropyOp
from tinytorch.autograd.ops.loss import BinaryCrossEntropy


class CrossEntropyLoss(Loss):
    """交叉熵损失函数。
    
    用于多分类任务。结合了 Softmax 和负对数似然损失。
    
    公式：
        loss = -sum(target_i * log(softmax(input_i)))
        
    对于单个样本：
        loss = -log(softmax(input)[target_class])
    
    Attributes:
        reduction: 损失聚合方式（'mean' 或 'sum'）
    
    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> # logits: (batch_size, num_classes)
        >>> logits = Tensor(NdArray([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]]))
        >>> # targets: (batch_size,) 类别索引
        >>> targets = Tensor(NdArray([0, 1]))
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """初始化交叉熵损失函数。
        
        Args:
            reduction: 损失聚合方式，'mean' 或 'sum'
        """
        super().__init__()
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction}")
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """计算交叉熵损失。
        
        Args:
            input: 预测 logits，形状 (batch_size, num_classes)
            target: 目标类别索引，形状 (batch_size,)
        
        Returns:
            交叉熵损失值
        """
        return CrossEntropyOp(self.reduction)(input, target)
    
    def __repr__(self) -> str:
        """返回损失函数的字符串表示。"""
        return f"CrossEntropyLoss(reduction={self.reduction})"


class BCELoss(Loss):
    """二元交叉熵损失函数。
    
    用于二分类任务。
    
    公式：
        loss = -[target * log(input) + (1 - target) * log(1 - input)]
    
    Attributes:
        reduction: 损失聚合方式（'mean' 或 'sum'）
    
    Example:
        >>> loss_fn = BCELoss()
        >>> # 预测概率（已经过 sigmoid）
        >>> probs = Tensor(NdArray([0.8, 0.3, 0.9]))
        >>> targets = Tensor(NdArray([1.0, 0.0, 1.0]))
        >>> loss = loss_fn(probs, targets)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """初始化二元交叉熵损失函数。
        
        Args:
            reduction: 损失聚合方式
        """
        super().__init__()
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction}")
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """计算二元交叉熵损失。
        
        Args:
            input: 预测概率，形状 (batch_size,) 或 (batch_size, 1)
            target: 目标标签（0 或 1），形状同 input
        
        Returns:
            BCE 损失值
        """
        return BinaryCrossEntropy(self.reduction)(input, target)
    
    def __repr__(self) -> str:
        """返回损失函数的字符串表示。"""
        return f"BCELoss(reduction={self.reduction})"
