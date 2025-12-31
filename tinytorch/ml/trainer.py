"""训练器类。

提供完整的模型训练流程控制。

Author: TinyAI Team
"""

from typing import Optional
from tinytorch.ml.model import Model
from tinytorch.ml.dataset import DataSet
from tinytorch.ml.optimizers.optimizer import Optimizer
from tinytorch.ml.losses.loss import Loss
from tinytorch.autograd.variable import Variable


class Trainer:
    """训练器类。
    
    封装完整的模型训练流程，包括训练循环、损失计算、参数更新等。
    
    Attributes:
        model: 待训练模型
        dataset: 训练数据集
        optimizer: 优化器
        loss_fn: 损失函数
        max_epochs: 最大训练轮次
        print_interval: 打印间隔（多少个 batch 打印一次）
    
    Example:
        >>> from tinytorch.nn import Sequential, Linear, ReLU
        >>> from tinytorch.ml import Model, Trainer, DataSet
        >>> from tinytorch.ml.optimizers import SGD
        >>> from tinytorch.ml.losses import MSELoss
        >>> 
        >>> # 构建模型
        >>> net = Sequential(Linear(10, 20), ReLU(), Linear(20, 1))
        >>> model = Model('MyModel', net)
        >>> 
        >>> # 准备数据
        >>> dataset = DataSet(train_data, train_labels, batch_size=32)
        >>> 
        >>> # 创建训练器
        >>> trainer = Trainer(
        ...     model=model,
        ...     dataset=dataset,
        ...     optimizer=SGD(model.parameters(), learning_rate=0.01),
        ...     loss_fn=MSELoss(),
        ...     max_epochs=100
        ... )
        >>> 
        >>> # 开始训练
        >>> trainer.train()
    """
    
    def __init__(self, model: Model, dataset: DataSet, 
                 optimizer: Optimizer, loss_fn: Loss,
                 max_epochs: int = 10, print_interval: int = 10,
                 val_dataset: Optional[DataSet] = None):
        """初始化训练器。
        
        Args:
            model: 待训练模型
            dataset: 训练数据集
            optimizer: 优化器
            loss_fn: 损失函数
            max_epochs: 最大训练轮次
            print_interval: 打印间隔
            val_dataset: 验证数据集（可选）
        """
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.print_interval = print_interval
        self.val_dataset = val_dataset
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def train(self) -> None:
        """执行完整的训练循环。"""
        print(f"开始训练模型: {self.model.name}")
        print(f"训练样本数: {len(self.dataset)}, 批次大小: {self.dataset.batch_size}")
        print(f"训练轮次: {self.max_epochs}")
        print("-" * 60)
        
        for epoch in range(self.max_epochs):
            # 训练一个 epoch
            epoch_loss = self.train_epoch(epoch)
            self.train_losses.append(epoch_loss)
            
            # 验证
            if self.val_dataset is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"Epoch [{epoch+1}/{self.max_epochs}] "
                      f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{self.max_epochs}] "
                      f"Train Loss: {epoch_loss:.6f}")
        
        print("-" * 60)
        print("训练完成!")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch。
        
        Args:
            epoch: 当前 epoch 编号
        
        Returns:
            该 epoch 的平均损失
        """
        # 设置为训练模式
        self.model.train()
        
        # 获取所有批次
        batches = self.dataset.get_batches()
        total_loss = 0.0
        num_batches = len(batches)
        
        for batch_idx, (batch_data, batch_labels) in enumerate(batches):
            # 转换为 Variable
            input_var = Variable(batch_data, requires_grad=False)
            target_var = Variable(batch_labels, requires_grad=False)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(input_var)
            
            # 计算损失
            loss = self.loss_fn(output, target_var)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累积损失
            loss_value = loss.value.data[0]
            total_loss += loss_value
            
            # 打印进度
            if (batch_idx + 1) % self.print_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.6f}")
        
        # 返回平均损失
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """在验证集上验证模型。
        
        Returns:
            验证集上的平均损失
        """
        if self.val_dataset is None:
            raise ValueError("No validation dataset provided")
        
        # 设置为评估模式
        self.model.eval()
        
        # 获取所有批次
        batches = self.val_dataset.get_batches()
        total_loss = 0.0
        num_batches = len(batches)
        
        for batch_data, batch_labels in batches:
            # 转换为 Variable
            input_var = Variable(batch_data, requires_grad=False)
            target_var = Variable(batch_labels, requires_grad=False)
            
            # 前向传播（不需要梯度）
            output = self.model(input_var)
            
            # 计算损失
            loss = self.loss_fn(output, target_var)
            
            # 累积损失
            loss_value = loss.value.data[0]
            total_loss += loss_value
        
        # 返回平均损失
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, file_path: str) -> None:
        """保存训练检查点。
        
        Args:
            file_path: 检查点保存路径
        """
        import pickle
        
        checkpoint = {
            'model_name': self.model.name,
            'model_state': self.model.module.to_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"检查点已保存至: {file_path}")
    
    def load_checkpoint(self, file_path: str) -> None:
        """加载训练检查点。
        
        Args:
            file_path: 检查点文件路径
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # 加载训练历史
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"检查点已从 {file_path} 加载")
    
    def __repr__(self) -> str:
        """返回训练器的字符串表示。"""
        return (f"Trainer(model={self.model.name}, "
                f"optimizer={self.optimizer.__class__.__name__}, "
                f"loss={self.loss_fn.__class__.__name__}, "
                f"epochs={self.max_epochs})")
