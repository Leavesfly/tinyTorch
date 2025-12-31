"""Machine learning module - Training and evaluation framework.

This module provides complete model training, evaluation, and management functionality.

Classes:
    Model: Model lifecycle management
    Trainer: Training loop controller
    DataSet: Dataset abstraction and batching
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
