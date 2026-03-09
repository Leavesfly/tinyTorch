"""机器学习模块 - 训练和评估框架。

该模块提供完整的模型训练、评估和管理功能。

类：
    Model: 模型生命周期管理
    Trainer: 训练循环控制器
    DataSet: 数据集抽象和批处理
"""

from tinytorch.ml.model import Model
from tinytorch.ml.trainer import Trainer
from tinytorch.ml.dataset import DataSet
from tinytorch.ml.monitor import Monitor, EarlyStopping
from tinytorch.ml import optimizers
from tinytorch.ml import losses
from tinytorch.ml import evaluators

# 导入常用类
from tinytorch.ml.optimizers import Optimizer, SGD, Adam
from tinytorch.ml.losses import Loss, MSELoss, CrossEntropyLoss, BCELoss
from tinytorch.ml.evaluators import (
    Evaluator, 
    AccuracyEvaluator, 
    PrecisionRecallEvaluator, 
    RegressionEvaluator
)

__all__ = [
    'Model', 
    'Trainer', 
    'DataSet',
    'Monitor',
    'EarlyStopping',
    'Optimizer',
    'SGD',
    'Adam',
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'Evaluator',
    'AccuracyEvaluator',
    'PrecisionRecallEvaluator',
    'RegressionEvaluator',
    'optimizers',
    'losses',
    'evaluators',
]
