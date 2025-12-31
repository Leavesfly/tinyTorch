"""交叉熵损失函数。

Author: TinyAI Team
"""

import math
from tinytorch.ml.losses.loss import Loss
from tinytorch.autograd import Variable
from tinytorch.tensor import Tensor


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
        >>> logits = Variable(Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]]))
        >>> # targets: (batch_size,) 类别索引
        >>> targets = Variable(Tensor([0, 1]))
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
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """计算交叉熵损失。
        
        Args:
            input: 预测 logits，形状 (batch_size, num_classes)
            target: 目标类别索引，形状 (batch_size,)
        
        Returns:
            交叉熵损失值
        """
        # 获取形状信息
        if len(input.value.shape.dims) == 2:
            batch_size, num_classes = input.value.shape.dims
        else:
            raise ValueError(f"Expected 2D input, got shape {input.value.shape.dims}")
        
        # 计算每个样本的损失
        losses = []
        
        for b in range(batch_size):
            # 提取当前样本的 logits
            logits = []
            for c in range(num_classes):
                idx = b * num_classes + c
                logits.append(input.value.data[idx])
            
            # 计算 softmax
            max_logit = max(logits)
            exp_logits = [math.exp(x - max_logit) for x in logits]
            sum_exp = sum(exp_logits)
            softmax_probs = [x / sum_exp for x in exp_logits]
            
            # 获取目标类别
            target_class = int(target.value.data[b])
            
            # 计算负对数似然
            # loss = -log(softmax_probs[target_class])
            prob = softmax_probs[target_class]
            if prob <= 0:
                prob = 1e-10  # 防止 log(0)
            
            loss = -math.log(prob)
            losses.append(loss)
        
        # 聚合损失
        if self.reduction == 'mean':
            total_loss = sum(losses) / len(losses)
        elif self.reduction == 'sum':
            total_loss = sum(losses)
        else:  # 'none'
            # 返回每个样本的损失
            return Variable(Tensor(losses, input.value.shape[:1], 'float32'), requires_grad=True)
        
        # 返回标量损失
        loss_tensor = Tensor([total_loss], (1,), 'float32')
        return Variable(loss_tensor, requires_grad=True)
    
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
        >>> probs = Variable(Tensor([0.8, 0.3, 0.9]))
        >>> targets = Variable(Tensor([1.0, 0.0, 1.0]))
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
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """计算二元交叉熵损失。
        
        Args:
            input: 预测概率，形状 (batch_size,) 或 (batch_size, 1)
            target: 目标标签（0 或 1），形状同 input
        
        Returns:
            BCE 损失值
        """
        # 展平输入
        input_data = input.value.data
        target_data = target.value.data
        
        if len(input_data) != len(target_data):
            raise ValueError(
                f"input and target must have same number of elements, "
                f"got {len(input_data)} and {len(target_data)}"
            )
        
        # 计算损失
        losses = []
        epsilon = 1e-10  # 防止 log(0)
        
        for i in range(len(input_data)):
            pred = input_data[i]
            tgt = target_data[i]
            
            # 限制概率范围
            pred = max(epsilon, min(1.0 - epsilon, pred))
            
            # BCE: -[target * log(pred) + (1 - target) * log(1 - pred)]
            loss = -(tgt * math.log(pred) + (1.0 - tgt) * math.log(1.0 - pred))
            losses.append(loss)
        
        # 聚合损失
        if self.reduction == 'mean':
            total_loss = sum(losses) / len(losses)
        elif self.reduction == 'sum':
            total_loss = sum(losses)
        else:  # 'none'
            return Variable(Tensor(losses, input.value.shape, 'float32'), requires_grad=True)
        
        loss_tensor = Tensor([total_loss], (1,), 'float32')
        return Variable(loss_tensor, requires_grad=True)
    
    def __repr__(self) -> str:
        """返回损失函数的字符串表示。"""
        return f"BCELoss(reduction={self.reduction})"
