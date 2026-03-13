"""端到端集成测试。

测试完整的训练流程，从数据加载到模型训练和评估。

Author: TinyAI Team
"""

import pytest
from tinytorch.ndarr import NdArray
from tinytorch.autograd import Tensor
from tinytorch.nn import Module
from tinytorch.nn.layers import Linear, ReLU
from tinytorch.nn.parameter import Parameter
from tinytorch.ml.model import Model
from tinytorch.ml.losses import MSELoss
from tinytorch.ml.optimizers import SGD


class SimpleModel(Module):
    """简单的测试模型。"""
    
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


class TestEndToEnd:
    """端到端集成测试。"""
    
    def test_simple_training_loop(self):
        """测试简单的训练循环。"""
        # 创建模型
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        
        # 创建损失函数和优化器
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 创建训练数据
        x_train = Tensor(NdArray([[1.0, 2.0, 3.0]]))
        y_train = Tensor(NdArray([[0.5, 0.5]]))
        
        # 训练一步
        optimizer.zero_grad()
        initial_params = [p.value.data.copy() for p in model.parameters()]
        
        # 前向传播
        y_pred = model(x_train)
        
        # 计算损失
        loss = loss_fn(y_pred, y_train)
        
        # 反向传播 + 参数更新
        loss.backward()
        optimizer.step()

        # 验证损失是标量，且参数发生更新
        assert loss.value.shape.size == 1
        updated_params = [p.value.data for p in model.parameters()]
        changed = any(initial_params[i] != updated_params[i] for i in range(len(initial_params)))
        assert changed
        
    def test_multiple_training_steps(self):
        """测试多步训练。"""
        # 创建模型
        model = SimpleModel(input_size=2, hidden_size=3, output_size=1)
        
        # 创建损失函数和优化器
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 创建训练数据
        x_train = Tensor(NdArray([[1.0, 2.0]]))
        y_train = Tensor(NdArray([[3.0]]))
        
        # 训练多步
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            losses.append(loss.value.data[0])
            loss.backward()
            optimizer.step()
        
        # 验证损失被记录
        assert len(losses) == 3
        
    def test_model_prediction(self):
        """测试模型预测。"""
        # 创建模型
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        
        # 设置为评估模式
        model.eval()
        
        # 进行预测
        x = Tensor(NdArray([[1.0, 2.0, 3.0]]))
        y_pred = model(x)
        
        # 验证输出形状
        assert y_pred.value.shape.dims[0] == 1
        assert y_pred.value.shape.dims[1] == 2


class TestModelSaving:
    """模型保存和加载的测试。"""
    
    def test_get_parameters(self):
        """测试获取模型参数。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        params = list(model.parameters())
        
        # 应该有权重和偏置参数
        assert len(params) > 0
        
    def test_parameter_update(self):
        """测试参数更新。"""
        model = SimpleModel(input_size=2, hidden_size=3, output_size=1)
        
        # 获取初始参数
        initial_params = [p.data.copy() for p in model.parameters()]
        
        # 训练一步
        optimizer = SGD(model.parameters(), lr=0.1)
        x = Tensor(NdArray([[1.0, 2.0]]))
        y = Tensor(NdArray([[1.0]]))
        
        y_pred = model(x)
        loss_fn = MSELoss()
        loss = loss_fn(y_pred, y)
        
        # 手动设置梯度（简化）
        for p in model.parameters():
            p.grad = NdArray([0.1] * len(p.data))
        
        optimizer.step()
        
        # 验证参数已更新
        updated_params = [p.data for p in model.parameters()]
        # 至少有一个参数应该改变
        changed = any(
            initial_params[i] != updated_params[i] 
            for i in range(len(initial_params))
        )
        assert changed

    def test_model_save_and_load_round_trip(self, tmp_path):
        """测试 Model.save/load 可恢复模块参数。"""
        module = SimpleModel(input_size=2, hidden_size=3, output_size=1)
        module.fc1.weight.value = NdArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        module.fc1.bias.value = NdArray([0.1, 0.2, 0.3])
        module.fc2.weight.value = NdArray([[0.4, 0.5, 0.6]])
        module.fc2.bias.value = NdArray([0.7])

        file_path = tmp_path / 'model.pkl'
        model = Model('simple', module)
        model.save(str(file_path))

        loaded = Model.load(
            str(file_path),
            module=SimpleModel(input_size=2, hidden_size=3, output_size=1),
        )

        loaded_params = dict(loaded.named_parameters())
        assert loaded_params['fc1.weight'].value.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert loaded_params['fc1.bias'].value.data == [0.1, 0.2, 0.3]
        assert loaded_params['fc2.weight'].value.data == [0.4, 0.5, 0.6]
        assert loaded_params['fc2.bias'].value.data == [0.7]


class TestBatchProcessing:
    """批处理的测试。"""
    
    def test_single_sample(self):
        """测试单样本处理。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        x = Tensor(NdArray([[1.0, 2.0, 3.0]]))
        y = model(x)
        assert y.value.shape.dims[0] == 1
        
    def test_small_batch(self):
        """测试小批量处理。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        # 批量大小为 2
        x = Tensor(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        y = model(x)
        assert y.value.shape.dims[0] == 2


class TestModelModes:
    """模型模式的测试。"""
    
    def test_train_mode(self):
        """测试训练模式。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        model.train()
        assert model.training == True
        
    def test_eval_mode(self):
        """测试评估模式。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        model.eval()
        assert model.training == False
        
    def test_mode_switch(self):
        """测试模式切换。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        
        # 默认是训练模式
        assert model.training == True
        
        # 切换到评估模式
        model.eval()
        assert model.training == False
        
        # 切换回训练模式
        model.train()
        assert model.training == True


class TestGradientFlow:
    """梯度流的测试。"""
    
    def test_gradient_computation(self):
        """测试梯度计算。"""
        # 创建简单的计算图
        x = Tensor(NdArray([2.0]), name="x")
        y = x * x  # y = x^2
        
        # 反向传播
        y.backward()
        
        # 验证梯度
        assert x.grad is not None
        
    def test_gradient_accumulation(self):
        """测试梯度累积。"""
        x = Tensor(NdArray([1.0]), name="x")
        
        # 第一次计算
        y1 = x * x
        y1.backward()
        grad1 = x.grad.data[0]
        
        # 第二次计算（不清除梯度）
        y2 = x * x * x
        y2.backward()
        
        # 梯度应该累积
        assert x.grad.data[0] != grad1


class TestErrorHandling:
    """错误处理的测试。"""
    
    def test_invalid_input_shape(self):
        """测试无效输入形状。"""
        model = SimpleModel(input_size=3, hidden_size=5, output_size=2)
        # 输入维度不匹配：Linear 期望 (batch, 3)，传入 (1, 2) 会导致 matmul 形状错误
        with pytest.raises(ValueError):
            x = Tensor(NdArray([[1.0, 2.0]]))  # 只有2维，应该是3维
            model(x)
            
    def test_empty_parameters(self):
        """测试空参数列表。"""
        optimizer = SGD([], lr=0.01)
        # 应该可以正常创建
        assert len(optimizer.params) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
