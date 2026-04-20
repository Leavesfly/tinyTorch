"""机器学习模块 - 训练和评估框架。

该模块提供完整的模型训练、评估和管理功能。

核心类：
    Model: 模型生命周期管理（保存、加载、参数管理）
    Trainer: 训练循环控制器
    DataSet: 数据集抽象和批处理（旧接口，推荐使用 tinytorch.utils.data.DataLoader）

子模块：
    optimizers: 优化器（SGD, Adam）
    losses: 损失函数（MSELoss, CrossEntropyLoss, BCELoss）
    evaluators: 评估器（AccuracyEvaluator 等）
"""

# --- 核心类 ---
from tinytorch.ml.model import Model
from tinytorch.ml.trainer import Trainer
from tinytorch.ml.dataset import DataSet
from tinytorch.ml.monitor import Monitor, EarlyStopping
from tinytorch.ml.visualizer import TrainingVisualizer

# --- 子模块 ---
from tinytorch.ml import optimizers
from tinytorch.ml import losses
from tinytorch.ml import evaluators

# --- 常用类的便捷导入 ---
from tinytorch.ml.optimizers import Optimizer, SGD, Adam
from tinytorch.ml.losses import Loss, MSELoss, CrossEntropyLoss, BCELoss
from tinytorch.ml.evaluators import (
    Evaluator,
    AccuracyEvaluator,
    PrecisionRecallEvaluator,
    RegressionEvaluator,
)

__all__ = [
    # 核心类
    'Model',
    'Trainer',
    'DataSet',
    'Monitor',
    'EarlyStopping',
    'TrainingVisualizer',
    # 优化器
    'Optimizer',
    'SGD',
    'Adam',
    # 损失函数
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    # 评估器
    'Evaluator',
    'AccuracyEvaluator',
    'PrecisionRecallEvaluator',
    'RegressionEvaluator',
    # 子模块
    'optimizers',
    'losses',
    'evaluators',
]
