"""训练监控类。

提供训练过程的监控、记录和可视化功能。

Author: TinyAI Team
"""

from typing import Dict, List, Any
import time


class Monitor:
    """训练监控器。
    
    记录和管理训练过程中的各种指标，如损失、准确率、学习率等。
    
    Attributes:
        metrics: 存储所有指标的字典
        history: 历史记录列表
        start_time: 训练开始时间
    
    Example:
        >>> monitor = Monitor()
        >>> monitor.record('train_loss', 0.5)
        >>> monitor.record('val_loss', 0.6)
        >>> monitor.print_summary()
    """
    
    def __init__(self):
        """初始化监控器。"""
        self.metrics: Dict[str, List[float]] = {}
        self.history: List[Dict[str, Any]] = []
        self.start_time = None
        self.current_epoch = 0
    
    def start(self) -> None:
        """开始训练监控。"""
        self.start_time = time.time()
        print("=" * 60)
        print("训练监控启动")
        print("=" * 60)
    
    def record(self, metric_name: str, value: float, epoch: int = None) -> None:
        """记录一个指标值。
        
        Args:
            metric_name: 指标名称
            value: 指标值
            epoch: epoch 编号（可选）
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        # 记录到历史
        if epoch is not None:
            self.current_epoch = epoch
    
    def record_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """记录一个 epoch 的所有指标。
        
        Args:
            epoch: epoch 编号
            metrics: 指标字典，如 {'train_loss': 0.5, 'val_acc': 0.9}
        """
        epoch_record = {
            'epoch': epoch,
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            **metrics
        }
        self.history.append(epoch_record)
        
        # 更新指标历史
        for name, value in metrics.items():
            self.record(name, value, epoch)
        
        self.current_epoch = epoch
    
    def get_metric(self, metric_name: str) -> List[float]:
        """获取某个指标的所有历史值。
        
        Args:
            metric_name: 指标名称
        
        Returns:
            指标值列表
        """
        return self.metrics.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> float:
        """获取某个指标的最新值。
        
        Args:
            metric_name: 指标名称
        
        Returns:
            最新的指标值，如果不存在则返回 None
        """
        values = self.metrics.get(metric_name, [])
        return values[-1] if values else None
    
    def get_best(self, metric_name: str, mode: str = 'min') -> float:
        """获取某个指标的最佳值。
        
        Args:
            metric_name: 指标名称
            mode: 'min' 表示越小越好，'max' 表示越大越好
        
        Returns:
            最佳指标值
        """
        values = self.metrics.get(metric_name, [])
        if not values:
            return None
        
        if mode == 'min':
            return min(values)
        elif mode == 'max':
            return max(values)
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def print_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """打印 epoch 信息。
        
        Args:
            epoch: epoch 编号
            metrics: 指标字典
        """
        # 计算已用时间
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # 构建输出字符串
        metrics_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        
        print(f"Epoch {epoch} [{elapsed:.1f}s] - {metrics_str}")
    
    def print_summary(self) -> None:
        """打印训练摘要。"""
        print("\n" + "=" * 60)
        print("训练摘要")
        print("=" * 60)
        
        if not self.metrics:
            print("无记录的指标")
            return
        
        # 打印各指标的统计信息
        for metric_name, values in self.metrics.items():
            if not values:
                continue
            
            latest = values[-1]
            best_min = min(values)
            best_max = max(values)
            avg = sum(values) / len(values)
            
            print(f"\n{metric_name}:")
            print(f"  最新值: {latest:.6f}")
            print(f"  最小值: {best_min:.6f}")
            print(f"  最大值: {best_max:.6f}")
            print(f"  平均值: {avg:.6f}")
        
        # 打印总时间
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n总训练时间: {total_time:.2f} 秒")
        
        print("=" * 60)
    
    def save_history(self, file_path: str) -> None:
        """保存训练历史到文件。
        
        Args:
            file_path: 保存路径
        """
        import pickle
        
        data = {
            'metrics': self.metrics,
            'history': self.history
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"训练历史已保存至: {file_path}")
    
    def load_history(self, file_path: str) -> None:
        """从文件加载训练历史。
        
        Args:
            file_path: 文件路径
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.metrics = data.get('metrics', {})
        self.history = data.get('history', [])
        
        print(f"训练历史已从 {file_path} 加载")
    
    def __repr__(self) -> str:
        """返回监控器的字符串表示。"""
        num_metrics = len(self.metrics)
        num_records = len(self.history)
        return f"Monitor(metrics={num_metrics}, records={num_records})"


class EarlyStopping:
    """早停机制。
    
    监控验证指标，当指标在一定 epoch 内没有改善时停止训练。
    
    Attributes:
        patience: 容忍的 epoch 数
        mode: 'min' 表示指标越小越好，'max' 表示越大越好
        min_delta: 最小改善阈值
        best_score: 最佳分数
        counter: 计数器
        should_stop: 是否应该停止训练
    
    Example:
        >>> early_stopping = EarlyStopping(patience=5, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_one_epoch()
        ...     if early_stopping.step(val_loss):
        ...         print("早停触发")
        ...         break
    """
    
    def __init__(self, patience: int = 10, mode: str = 'min', min_delta: float = 0.0):
        """初始化早停机制。
        
        Args:
            patience: 容忍的 epoch 数
            mode: 'min' 或 'max'
            min_delta: 最小改善阈值
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def step(self, score: float) -> bool:
        """更新早停状态。
        
        Args:
            score: 当前指标值
        
        Returns:
            True 表示应该停止训练，False 表示继续
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # 检查是否改善
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def reset(self) -> None:
        """重置早停状态。"""
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __repr__(self) -> str:
        """返回早停机制的字符串表示。"""
        return (f"EarlyStopping(patience={self.patience}, mode={self.mode}, "
                f"counter={self.counter}/{self.patience})")
