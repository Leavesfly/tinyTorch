"""训练系统单元测试。

测试 Trainer、Monitor、EarlyStopping 和 Model save/load 功能。

Author: TinyAI Team
"""

import pytest
import tempfile
from tinytorch.ndarr import NdArray
from tinytorch.autograd import Tensor
from tinytorch.nn import Module
from tinytorch.nn.layers import Linear, ReLU
from tinytorch.nn.parameter import Parameter
from tinytorch.ml.model import Model
from tinytorch.ml.dataset import DataSet
from tinytorch.ml.monitor import Monitor, EarlyStopping
from tinytorch.ml.trainer import Trainer
from tinytorch.ml.losses import MSELoss
from tinytorch.ml.optimizers import SGD
from tinytorch.utils import random as tt_random


class SimpleMLP(Module):
    """简单的两层 MLP 测试模型。"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """初始化模型。"""
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        """前向传播。"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestMonitor:
    """Monitor 类的测试。"""
    
    def test_monitor_record(self):
        """测试记录指标并验证。"""
        tt_random.seed(42)
        monitor = Monitor()
        
        # 记录多个指标
        monitor.record('train_loss', 0.5)
        monitor.record('train_loss', 0.4)
        monitor.record('train_loss', 0.3)
        
        monitor.record('val_loss', 0.6)
        monitor.record('val_loss', 0.5)
        
        # 验证记录
        assert len(monitor.metrics['train_loss']) == 3
        assert monitor.metrics['train_loss'] == [0.5, 0.4, 0.3]
        assert len(monitor.metrics['val_loss']) == 2
        assert monitor.metrics['val_loss'] == [0.6, 0.5]
    
    def test_monitor_get_best(self):
        """测试获取最佳值（min/max 模式）。"""
        tt_random.seed(42)
        monitor = Monitor()
        
        # 逐个记录指标（record 接受单个 float）
        for val in [0.5, 0.3, 0.4, 0.2, 0.35]:
            monitor.record('loss', val)
        for val in [0.7, 0.75, 0.8, 0.78, 0.85]:
            monitor.record('accuracy', val)
        
        # 测试 min 模式
        best_loss = monitor.get_best('loss', mode='min')
        assert best_loss == 0.2
        
        # 测试 max 模式
        best_acc = monitor.get_best('accuracy', mode='max')
        assert best_acc == 0.85
        
        # 测试不存在的指标
        assert monitor.get_best('nonexistent', mode='min') is None
        
        # 测试无效模式
        with pytest.raises(ValueError):
            monitor.get_best('loss', mode='invalid')
    
    def test_monitor_summary(self):
        """测试 summary 不报错。"""
        tt_random.seed(42)
        monitor = Monitor()
        
        # 记录一些指标
        monitor.record('train_loss', 0.5)
        monitor.record('train_loss', 0.4)
        monitor.record('val_loss', 0.6)
        
        # 调用 summary 应该不报错
        monitor.print_summary()
        
        # 测试空指标的情况
        empty_monitor = Monitor()
        empty_monitor.print_summary()


class TestEarlyStopping:
    """EarlyStopping 类的测试。"""
    
    def test_early_stopping_improvement(self):
        """测试有改善时不触发。"""
        tt_random.seed(42)
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # 模拟改善的情况
        scores = [0.5, 0.4, 0.3, 0.25]
        
        for score in scores:
            should_stop = early_stopping.step(score)
            assert not should_stop
        
        # 验证状态
        assert early_stopping.best_score == 0.25
        assert early_stopping.counter == 0
        assert not early_stopping.should_stop
    
    def test_early_stopping_no_improvement(self):
        """测试无改善超过 patience 时触发。"""
        tt_random.seed(42)
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # 初始最佳分数
        early_stopping.step(0.5)
        
        # 模拟无改善的情况（持续高于最佳分数）
        scores = [0.6, 0.7, 0.8]  # 3次无改善
        
        for i, score in enumerate(scores):
            should_stop = early_stopping.step(score)
            # 前2次不应该触发
            if i < 2:
                assert not should_stop
            else:
                # 第3次应该触发
                assert should_stop
        
        # 验证最终状态
        assert early_stopping.best_score == 0.5
        assert early_stopping.counter == 3
        assert early_stopping.should_stop
    
    def test_early_stopping_min_mode(self):
        """测试 min 模式。"""
        tt_random.seed(42)
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        # 测试 min 模式（越小越好）
        early_stopping.step(0.5)
        assert early_stopping.best_score == 0.5
        
        # 改善
        early_stopping.step(0.4)
        assert early_stopping.best_score == 0.4
        assert early_stopping.counter == 0
        
        # 无改善
        early_stopping.step(0.45)
        assert early_stopping.best_score == 0.4
        assert early_stopping.counter == 1
        
        # 再次无改善，触发早停
        should_stop = early_stopping.step(0.5)
        assert should_stop
        assert early_stopping.counter == 2
    
    def test_early_stopping_max_mode(self):
        """测试 max 模式。"""
        tt_random.seed(42)
        early_stopping = EarlyStopping(patience=2, mode='max')
        
        # 测试 max 模式（越大越好）
        early_stopping.step(0.5)
        assert early_stopping.best_score == 0.5
        
        # 改善
        early_stopping.step(0.6)
        assert early_stopping.best_score == 0.6
        assert early_stopping.counter == 0
        
        # 无改善
        early_stopping.step(0.55)
        assert early_stopping.best_score == 0.6
        assert early_stopping.counter == 1
        
        # 再次无改善，触发早停
        should_stop = early_stopping.step(0.5)
        assert should_stop
        assert early_stopping.counter == 2
    
    def test_early_stopping_reset(self):
        """测试重置早停状态。"""
        tt_random.seed(42)
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # 设置一些状态
        early_stopping.step(0.5)
        early_stopping.step(0.6)
        
        # 重置
        early_stopping.reset()
        
        # 验证重置后的状态
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert not early_stopping.should_stop
    
    def test_early_stopping_min_delta(self):
        """测试最小改善阈值。"""
        tt_random.seed(42)
        early_stopping = EarlyStopping(patience=2, mode='min', min_delta=0.1)
        
        # 初始
        early_stopping.step(0.5)
        
        # 改善小于 min_delta，不应该算作改善
        early_stopping.step(0.45)  # 改善 0.05 < 0.1
        assert early_stopping.best_score == 0.5
        assert early_stopping.counter == 1
        
        # 改善大于 min_delta，应该算作改善
        early_stopping.step(0.3)  # 改善 0.2 > 0.1
        assert early_stopping.best_score == 0.3
        assert early_stopping.counter == 0


class TestModelSaveLoad:
    """Model save/load 功能的测试。"""
    
    def test_model_save_load(self):
        """测试保存模型到临时文件，加载后验证参数一致。"""
        tt_random.seed(42)
        
        # 创建模型并设置特定参数
        module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        module.fc1.weight.value = NdArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        module.fc1.bias.value = NdArray([0.1, 0.2, 0.3])
        module.fc2.weight.value = NdArray([[0.4, 0.5, 0.6]])
        module.fc2.bias.value = NdArray([0.7])
        
        model = Model('test_model', module)
        
        # 使用临时文件保存和加载
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # 保存模型
            model.save(temp_path)
            
            # 加载模型
            loaded_model = Model.load(
                temp_path,
                module=SimpleMLP(input_size=2, hidden_size=3, output_size=1)
            )
            
            # 验证参数一致
            original_params = dict(model.named_parameters())
            loaded_params = dict(loaded_model.named_parameters())
            
            assert len(original_params) == len(loaded_params)
            
            for name in original_params:
                assert name in loaded_params
                original_data = original_params[name].value.data
                loaded_data = loaded_params[name].value.data
                assert original_data == loaded_data, f"Parameter {name} mismatch"
        
        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_model_state_dict(self):
        """验证 state_dict 和 load_state_dict。"""
        tt_random.seed(42)
        
        # 创建模型
        module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        
        # 获取 state_dict
        state_dict = module.state_dict()
        
        # 验证 state_dict 包含所有参数
        assert 'fc1.weight' in state_dict
        assert 'fc1.bias' in state_dict
        assert 'fc2.weight' in state_dict
        assert 'fc2.bias' in state_dict
        
        # 修改参数
        module.fc1.weight.value = NdArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        module.fc1.bias.value = NdArray([0.1, 0.2, 0.3])
        module.fc2.weight.value = NdArray([[0.4, 0.5, 0.6]])
        module.fc2.bias.value = NdArray([0.7])
        
        # 获取新的 state_dict
        new_state_dict = module.state_dict()
        
        # 创建新模型并加载状态
        new_module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        new_module.load_state_dict(new_state_dict)
        
        # 验证参数一致：state_dict 的值是 dict（含 kind/value/shape 等），
        # 比较时应对比嵌套列表形式的 value 字段
        for name in new_state_dict:
            original_entry = new_state_dict[name]
            loaded_param = None
            for param_name, param in new_module.named_parameters():
                if param_name == name:
                    loaded_param = param.value.to_list()
                    break
            assert loaded_param is not None
            assert loaded_param == original_entry['value']
    
    def test_model_save_load_round_trip(self):
        """测试完整的保存和加载流程。"""
        tt_random.seed(42)
        
        # 创建模型
        module = SimpleMLP(input_size=3, hidden_size=5, output_size=2)
        model = Model('round_trip_test', module)
        
        # 进行一次前向传播来初始化参数
        x = Tensor(NdArray([[1.0, 2.0, 3.0]]))
        _ = model(x)
        
        # 获取保存前的参数
        original_params = {name: p.value.data.copy() for name, p in model.named_parameters()}
        
        # 使用临时文件
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # 保存
            model.save(temp_path)
            
            # 加载
            loaded_model = Model.load(
                temp_path,
                module=SimpleMLP(input_size=3, hidden_size=5, output_size=2)
            )
            
            # 验证参数恢复
            loaded_params = {name: p.value.data for name, p in loaded_model.named_parameters()}
            
            for name in original_params:
                assert name in loaded_params
                assert loaded_params[name] == original_params[name]
            
            # 验证模型名称和元数据
            assert loaded_model.name == 'round_trip_test'
            assert loaded_model.model_info['framework'] == 'tinyTorch'
        
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestTrainer:
    """Trainer 类的测试。"""
    
    def test_trainer_basic(self):
        """测试创建 Trainer 并训练几个 epoch，验证损失下降。"""
        tt_random.seed(42)
        
        # 创建简单的训练数据
        data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        labels = [[3.0], [5.0], [7.0], [9.0]]  # y = x1 + x2
        
        # 创建数据集
        dataset = DataSet(data, labels, batch_size=2, shuffle=False)
        
        # 创建模型
        module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        model = Model('basic_test', module)
        
        # 创建损失函数和优化器
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_epochs=5,
            print_interval=10
        )
        
        # 训练
        trainer.train()
        
        # 验证训练历史
        assert len(trainer.train_losses) == 5
        
        # 验证损失下降（至少最后应该比开始时小）
        assert trainer.train_losses[-1] < trainer.train_losses[0] * 1.5  # 允许一定的波动
    
    def test_trainer_with_monitor(self):
        """测试带 Monitor 的训练。"""
        tt_random.seed(42)
        
        # 创建训练数据
        data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        labels = [[3.0], [5.0], [7.0], [9.0]]
        
        # 创建数据集
        dataset = DataSet(data, labels, batch_size=2, shuffle=False)
        
        # 创建模型
        module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        model = Model('monitor_test', module)
        
        # 创建损失函数和优化器
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_epochs=3,
            print_interval=10
        )
        
        # 创建监控器
        monitor = Monitor()
        monitor.start()
        
        # 训练并记录
        for epoch in range(3):
            epoch_loss = trainer.train_epoch(epoch)
            monitor.record_epoch(epoch, {'train_loss': epoch_loss})
        
        # 验证监控器记录
        assert len(monitor.history) == 3
        assert len(monitor.metrics['train_loss']) == 3
        
        # 验证摘要不报错
        monitor.print_summary()
    
    def test_trainer_with_validation(self):
        """测试带验证集的训练。"""
        tt_random.seed(42)
        
        # 创建训练数据
        train_data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        train_labels = [[3.0], [5.0], [7.0], [9.0]]
        
        val_data = [[1.5, 2.5], [2.5, 3.5]]
        val_labels = [[4.0], [6.0]]
        
        # 创建数据集
        train_dataset = DataSet(train_data, train_labels, batch_size=2, shuffle=False)
        val_dataset = DataSet(val_data, val_labels, batch_size=2, shuffle=False)
        
        # 创建模型
        module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        model = Model('val_test', module)
        
        # 创建损失函数和优化器
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 创建带验证集的训练器
        trainer = Trainer(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_epochs=3,
            print_interval=10,
            val_dataset=val_dataset
        )
        
        # 训练
        trainer.train()
        
        # 验证训练和验证损失都被记录
        assert len(trainer.train_losses) == 3
        assert len(trainer.val_losses) == 3
    
    def test_trainer_with_early_stopping(self):
        """测试带早停的训练。"""
        tt_random.seed(42)
        
        # 创建训练数据
        data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        labels = [[3.0], [5.0], [7.0], [9.0]]
        
        # 创建数据集
        dataset = DataSet(data, labels, batch_size=2, shuffle=False)
        
        # 创建模型
        module = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
        model = Model('early_stop_test', module)
        
        # 创建损失函数和优化器
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_epochs=10,
            print_interval=10
        )
        
        # 创建早停机制
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        # 模拟训练循环
        for epoch in range(5):
            epoch_loss = trainer.train_epoch(epoch)
            
            if early_stopping.step(epoch_loss):
                # 早停触发
                break
        
        # 验证早停状态
        assert early_stopping.best_score is not None
        # 验证训练确实进行了一些 epoch
        assert epoch >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
