"""评估器基类。

Author: TinyAI Team
"""

from typing import List, Union


def _validate_inputs(predictions: List, targets: List) -> None:
    """验证预测值和目标值的长度是否一致。

    Args:
        predictions: 预测值列表
        targets: 目标值列表

    Raises:
        ValueError: 当长度不匹配时
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must have same length, "
            f"got {len(predictions)} and {len(targets)}"
        )


def _to_class_indices(values: List[Union[List[float], float]]) -> List[int]:
    """将预测值或目标值转换为类别索引列表。

    如果元素是列表/元组（多分类概率/logits），取最大值索引；
    否则直接转换为整数。

    Args:
        values: 预测值或目标值列表

    Returns:
        类别索引列表
    """
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.append(value.index(max(value)))
        else:
            result.append(int(value))
    return result


class Evaluator:
    """评估器基类。
    
    所有评估器都应该继承这个类。子类需要实现 evaluate 方法定义评估逻辑。
    
    Example:
        >>> evaluator = AccuracyEvaluator()
        >>> predictions = [[0.1, 0.9], [0.8, 0.2]]
        >>> targets = [1, 0]
        >>> accuracy = evaluator.evaluate(predictions, targets)
    """
    
    def __init__(self, name: str = None):
        """初始化评估器。
        
        Args:
            name: 评估器名称
        """
        self.name = name or self.__class__.__name__
    
    def evaluate(self, predictions: List, targets: List) -> float:
        """计算评估指标（抽象方法，子类必须实现）。
        
        Args:
            predictions: 预测值列表
            targets: 目标值列表
        
        Returns:
            评估指标值
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement evaluate method")
    
    def __call__(self, predictions: List, targets: List) -> float:
        """调用评估器。
        
        Args:
            predictions: 预测值
            targets: 目标值
        
        Returns:
            评估指标值
        """
        return self.evaluate(predictions, targets)
    
    def __repr__(self) -> str:
        """返回评估器的字符串表示。"""
        return f"{self.__class__.__name__}()"
