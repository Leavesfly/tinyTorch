"""准确率评估器。

Author: TinyAI Team
"""

from typing import List
from tinytorch.ml.evaluators.evaluator import Evaluator


class AccuracyEvaluator(Evaluator):
    """准确率评估器。
    
    计算分类任务的准确率。
    
    公式: accuracy = (正确预测数) / (总样本数)
    
    Example:
        >>> evaluator = AccuracyEvaluator()
        >>> # 预测值（logits 或概率）
        >>> predictions = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
        >>> # 真实标签
        >>> targets = [1, 0, 1]
        >>> accuracy = evaluator.evaluate(predictions, targets)
        >>> print(f"Accuracy: {accuracy:.2%}")
        Accuracy: 100.00%
    """
    
    def __init__(self):
        """初始化准确率评估器。"""
        super().__init__()
    
    def evaluate(self, predictions: List, targets: List) -> float:
        """计算准确率。
        
        Args:
            predictions: 预测值，可以是：
                - 二维列表：每个样本的类别概率或logits，形状 (num_samples, num_classes)
                - 一维列表：每个样本的预测类别索引
            targets: 目标标签，一维列表，每个元素是类别索引
        
        Returns:
            准确率（0-1之间的浮点数）
        
        Raises:
            ValueError: 当预测值和目标值数量不匹配时
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"predictions and targets must have same length, "
                f"got {len(predictions)} and {len(targets)}"
            )
        
        if len(predictions) == 0:
            return 0.0
        
        correct = 0
        total = len(predictions)
        
        for pred, target in zip(predictions, targets):
            # 如果预测值是列表（多分类概率/logits）
            if isinstance(pred, (list, tuple)):
                # 取最大值的索引作为预测类别
                pred_class = pred.index(max(pred))
            else:
                # 如果已经是类别索引
                pred_class = int(pred)
            
            # 目标值转换为整数
            target_class = int(target) if not isinstance(target, int) else target
            
            if pred_class == target_class:
                correct += 1
        
        accuracy = correct / total
        return accuracy


class PrecisionRecallEvaluator(Evaluator):
    """精确率和召回率评估器。
    
    计算二分类或多分类任务的 Precision、Recall 和 F1-Score。
    
    Attributes:
        average: 多分类时的平均方式，'binary'、'macro' 或 'micro'
    
    Example:
        >>> evaluator = PrecisionRecallEvaluator(average='binary')
        >>> predictions = [0, 1, 1, 0, 1]
        >>> targets = [0, 1, 0, 0, 1]
        >>> metrics = evaluator.evaluate_all(predictions, targets)
        >>> print(f"Precision: {metrics['precision']:.2f}")
        >>> print(f"Recall: {metrics['recall']:.2f}")
        >>> print(f"F1-Score: {metrics['f1']:.2f}")
    """
    
    def __init__(self, average: str = 'binary'):
        """初始化精确率召回率评估器。
        
        Args:
            average: 平均方式
                - 'binary': 二分类（默认）
                - 'macro': 宏平均（各类别指标的算术平均）
                - 'micro': 微平均（所有样本的总体指标）
        """
        super().__init__()
        self.average = average
    
    def evaluate(self, predictions: List, targets: List) -> float:
        """计算 F1-Score。
        
        Args:
            predictions: 预测值
            targets: 目标值
        
        Returns:
            F1-Score
        """
        metrics = self.evaluate_all(predictions, targets)
        return metrics['f1']
    
    def evaluate_all(self, predictions: List, targets: List) -> dict:
        """计算所有指标。
        
        Args:
            predictions: 预测值
            targets: 目标值
        
        Returns:
            包含 precision、recall、f1 的字典
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"predictions and targets must have same length, "
                f"got {len(predictions)} and {len(targets)}"
            )
        
        # 转换预测值
        pred_classes = []
        for pred in predictions:
            if isinstance(pred, (list, tuple)):
                pred_classes.append(pred.index(max(pred)))
            else:
                pred_classes.append(int(pred))
        
        # 转换目标值
        target_classes = [int(t) if not isinstance(t, int) else t for t in targets]
        
        # 计算混淆矩阵元素
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        for pred, target in zip(pred_classes, target_classes):
            if pred == 1 and target == 1:
                true_positive += 1
            elif pred == 1 and target == 0:
                false_positive += 1
            elif pred == 0 and target == 1:
                false_negative += 1
        
        # 计算 Precision
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0.0
        
        # 计算 Recall
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0.0
        
        # 计算 F1-Score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class RegressionEvaluator(Evaluator):
    """回归评估器。
    
    计算回归任务的评估指标：MAE、MSE、RMSE。
    
    Example:
        >>> evaluator = RegressionEvaluator()
        >>> predictions = [2.5, 3.0, 4.5]
        >>> targets = [2.0, 3.5, 4.0]
        >>> metrics = evaluator.evaluate_all(predictions, targets)
        >>> print(f"MAE: {metrics['mae']:.4f}")
        >>> print(f"MSE: {metrics['mse']:.4f}")
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    
    def __init__(self):
        """初始化回归评估器。"""
        super().__init__()
    
    def evaluate(self, predictions: List, targets: List) -> float:
        """计算 MSE（默认指标）。
        
        Args:
            predictions: 预测值
            targets: 目标值
        
        Returns:
            MSE 值
        """
        metrics = self.evaluate_all(predictions, targets)
        return metrics['mse']
    
    def evaluate_all(self, predictions: List, targets: List) -> dict:
        """计算所有回归指标。
        
        Args:
            predictions: 预测值
            targets: 目标值
        
        Returns:
            包含 mae、mse、rmse 的字典
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"predictions and targets must have same length, "
                f"got {len(predictions)} and {len(targets)}"
            )
        
        if len(predictions) == 0:
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        
        # 提取数值
        pred_values = []
        target_values = []
        
        for pred, target in zip(predictions, targets):
            # 如果是列表，取第一个元素
            if isinstance(pred, (list, tuple)):
                pred_values.append(float(pred[0]))
            else:
                pred_values.append(float(pred))
            
            if isinstance(target, (list, tuple)):
                target_values.append(float(target[0]))
            else:
                target_values.append(float(target))
        
        # 计算 MAE (Mean Absolute Error)
        mae = sum(abs(p - t) for p, t in zip(pred_values, target_values)) / len(pred_values)
        
        # 计算 MSE (Mean Squared Error)
        mse = sum((p - t) ** 2 for p, t in zip(pred_values, target_values)) / len(pred_values)
        
        # 计算 RMSE (Root Mean Squared Error)
        rmse = mse ** 0.5
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
