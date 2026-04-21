"""评估器实现。

提供分类和回归任务的评估器：
    - AccuracyEvaluator: 准确率评估
    - PrecisionRecallEvaluator: 精确率/召回率/F1 评估
    - RegressionEvaluator: MAE/MSE/RMSE 评估

Author: TinyAI Team
"""

from typing import List, Union, Dict
from tinytorch.ml.evaluators.evaluator import Evaluator, _validate_inputs, _to_class_indices


class AccuracyEvaluator(Evaluator):
    """准确率评估器。

    计算分类任务的准确率。

    公式: accuracy = (正确预测数) / (总样本数)

    Example:
        >>> evaluator = AccuracyEvaluator()
        >>> predictions = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
        >>> targets = [1, 0, 1]
        >>> accuracy = evaluator.evaluate(predictions, targets)
        >>> print(f"Accuracy: {accuracy:.2%}")
        Accuracy: 100.00%
    """

    def evaluate(self, predictions: List[Union[List[float], float]], targets: List[Union[int, float]]) -> float:
        """计算准确率。

        Args:
            predictions: 预测值（多分类概率/logits 或类别索引）
            targets: 目标标签（类别索引）

        Returns:
            准确率（0-1之间的浮点数）
        """
        _validate_inputs(predictions, targets)
        if len(predictions) == 0:
            return 0.0

        pred_classes = _to_class_indices(predictions)
        target_classes = _to_class_indices(targets)
        correct = sum(p == t for p, t in zip(pred_classes, target_classes))
        return correct / len(predictions)


class PrecisionRecallEvaluator(Evaluator):
    """精确率和召回率评估器。

    计算二分类任务的 Precision、Recall 和 F1-Score。

    Attributes:
        average: 多分类时的平均方式，'binary'、'macro' 或 'micro'

    Example:
        >>> evaluator = PrecisionRecallEvaluator(average='binary')
        >>> predictions = [0, 1, 1, 0, 1]
        >>> targets = [0, 1, 0, 0, 1]
        >>> metrics = evaluator.evaluate_all(predictions, targets)
        >>> print(f"F1-Score: {metrics['f1']:.2f}")
    """

    def __init__(self, average: str = 'binary') -> None:
        """初始化精确率召回率评估器。

        Args:
            average: 平均方式（'binary'、'macro' 或 'micro'）
        """
        super().__init__()
        self.average: str = average

    def evaluate(self, predictions: List[Union[List[float], float]], targets: List[Union[int, float]]) -> float:
        """计算 F1-Score。"""
        return self.evaluate_all(predictions, targets)['f1']

    def evaluate_all(self, predictions: List[Union[List[float], float]], targets: List[Union[int, float]]) -> Dict[str, float]:
        """计算 Precision、Recall 和 F1-Score。

        Args:
            predictions: 预测值
            targets: 目标值

        Returns:
            包含 precision、recall、f1 的字典
        """
        _validate_inputs(predictions, targets)

        pred_classes = _to_class_indices(predictions)
        target_classes = _to_class_indices(targets)

        true_positive = sum(p == 1 and t == 1 for p, t in zip(pred_classes, target_classes))
        false_positive = sum(p == 1 and t == 0 for p, t in zip(pred_classes, target_classes))
        false_negative = sum(p == 0 and t == 1 for p, t in zip(pred_classes, target_classes))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1_score}


class RegressionEvaluator(Evaluator):
    """回归评估器。

    计算回归任务的评估指标：MAE、MSE、RMSE。

    Example:
        >>> evaluator = RegressionEvaluator()
        >>> predictions = [2.5, 3.0, 4.5]
        >>> targets = [2.0, 3.5, 4.0]
        >>> metrics = evaluator.evaluate_all(predictions, targets)
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """

    def evaluate(self, predictions: List[Union[List[float], float]], targets: List[Union[int, float]]) -> float:
        """计算 MSE（默认指标）。"""
        return self.evaluate_all(predictions, targets)['mse']

    def evaluate_all(self, predictions: List[Union[List[float], float]], targets: List[Union[int, float]]) -> Dict[str, float]:
        """计算 MAE、MSE 和 RMSE。

        Args:
            predictions: 预测值
            targets: 目标值

        Returns:
            包含 mae、mse、rmse 的字典
        """
        _validate_inputs(predictions, targets)
        if len(predictions) == 0:
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}

        pred_values = [float(p[0]) if isinstance(p, (list, tuple)) else float(p) for p in predictions]
        target_values = [float(t[0]) if isinstance(t, (list, tuple)) else float(t) for t in targets]

        mae = sum(abs(p - t) for p, t in zip(pred_values, target_values)) / len(pred_values)
        mse = sum((p - t) ** 2 for p, t in zip(pred_values, target_values)) / len(pred_values)
        rmse = mse ** 0.5

        return {'mae': mae, 'mse': mse, 'rmse': rmse}
