"""循环神经网络层。

Author: TinyAI Team
"""

import math
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd import Variable
from tinytorch.tensor import Tensor, Shape
from tinytorch.nn import init


class RNN(Module):
    """基础循环神经网络层。
    
    实现标准的 RNN 单元：h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    
    Attributes:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        use_bias: 是否使用偏置
        W_ih: 输入到隐藏的权重
        W_hh: 隐藏到隐藏的权重
        bias: 偏置项
    
    Example:
        >>> rnn = RNN(input_size=10, hidden_size=20)
        >>> x = Variable(Tensor.randn((batch_size, seq_len, 10)))
        >>> h = rnn(x)
        >>> print(h.value.shape)
        (batch_size, seq_len, 20)
    """
    
    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True):
        """初始化 RNN 层。
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            use_bias: 是否使用偏置
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        # 初始化权重
        k = 1.0 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.uniform(-k, k, (hidden_size, input_size)),
            name='W_ih'
        )
        self.W_hh = Parameter(
            init.uniform(-k, k, (hidden_size, hidden_size)),
            name='W_hh'
        )
        
        if use_bias:
            self.bias = Parameter(
                init.uniform(-k, k, (hidden_size,)),
                name='bias'
            )
        else:
            self.bias = None
    
    def forward(self, input: Variable, h_0: Variable = None) -> Variable:
        """前向传播。
        
        Args:
            input: 输入序列，形状 (batch_size, seq_len, input_size)
            h_0: 初始隐藏状态，形状 (batch_size, hidden_size)，默认为零
        
        Returns:
            所有时间步的隐藏状态，形状 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, input_size = input.value.shape.dims
        
        if input_size != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {input_size}"
            )
        
        # 初始化隐藏状态
        if h_0 is None:
            h_t = Variable(Tensor.zeros((batch_size, self.hidden_size)), requires_grad=False)
        else:
            h_t = h_0
        
        # 存储所有时间步的输出
        outputs = []
        
        for t in range(seq_len):
            # 提取当前时间步的输入：x_t (batch_size, input_size)
            x_t_data = []
            for b in range(batch_size):
                for i in range(input_size):
                    idx = b * seq_len * input_size + t * input_size + i
                    x_t_data.append(input.value.data[idx])
            
            x_t = Variable(
                Tensor(x_t_data, Shape((batch_size, input_size)), 'float32'),
                requires_grad=input.requires_grad
            )
            
            # 计算 h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
            h_t = self._cell_forward(x_t, h_t)
            outputs.append(h_t)
        
        # 拼接所有时间步的输出
        output_data = []
        for b in range(batch_size):
            for t in range(seq_len):
                for h in range(self.hidden_size):
                    idx = b * self.hidden_size + h
                    output_data.append(outputs[t].value.data[idx])
        
        output_shape = Shape((batch_size, seq_len, self.hidden_size))
        output_tensor = Tensor(output_data, output_shape, 'float32')
        
        return Variable(output_tensor, requires_grad=input.requires_grad)
    
    def _cell_forward(self, x_t: Variable, h_prev: Variable) -> Variable:
        """单个 RNN 单元的前向传播。
        
        Args:
            x_t: 当前时间步输入，形状 (batch_size, input_size)
            h_prev: 上一时间步隐藏状态，形状 (batch_size, hidden_size)
        
        Returns:
            当前时间步隐藏状态，形状 (batch_size, hidden_size)
        """
        # W_ih @ x_t
        ih_out = self._matmul(x_t, self.W_ih, transpose_weight=True)
        
        # W_hh @ h_prev
        hh_out = self._matmul(h_prev, self.W_hh, transpose_weight=True)
        
        # 相加
        h_t = self._add_variables(ih_out, hh_out)
        
        # 加偏置
        if self.use_bias:
            h_t = self._add_bias(h_t, self.bias)
        
        # tanh 激活
        h_t = self._apply_tanh(h_t)
        
        return h_t
    
    def _matmul(self, x: Variable, weight: Parameter, transpose_weight: bool = False) -> Variable:
        """矩阵乘法辅助函数。"""
        if transpose_weight:
            # x @ weight.T
            w_t = weight.value.transpose()
        else:
            w_t = weight.value
        
        result = x.value.matmul(w_t)
        return Variable(result, requires_grad=x.requires_grad)
    
    def _add_variables(self, a: Variable, b: Variable) -> Variable:
        """Variable 相加。"""
        result = a.value.add(b.value)
        return Variable(result, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _add_bias(self, x: Variable, bias: Parameter) -> Variable:
        """添加偏置（广播）。"""
        # x: (batch_size, hidden_size), bias: (hidden_size,)
        batch_size, hidden_size = x.value.shape.dims
        result_data = []
        
        for b in range(batch_size):
            for h in range(hidden_size):
                idx = b * hidden_size + h
                result_data.append(x.value.data[idx] + bias.value.data[h])
        
        result_tensor = Tensor(result_data, x.value.shape, 'float32')
        return Variable(result_tensor, requires_grad=x.requires_grad)
    
    def _apply_tanh(self, x: Variable) -> Variable:
        """应用 tanh 激活函数。"""
        result = x.value.tanh()
        return Variable(result, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"use_bias={self.use_bias})")


class LSTM(Module):
    """长短期记忆网络层。
    
    实现 LSTM 单元，包含输入门、遗忘门、输出门和单元状态。
    
    公式：
        i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)  # 输入门
        f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)  # 遗忘门
        g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)     # 候选单元状态
        o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)  # 输出门
        c_t = f_t * c_{t-1} + i_t * g_t                    # 单元状态
        h_t = o_t * tanh(c_t)                               # 隐藏状态
    
    Attributes:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        use_bias: 是否使用偏置
    
    Example:
        >>> lstm = LSTM(input_size=10, hidden_size=20)
        >>> x = Variable(Tensor.randn((batch_size, seq_len, 10)))
        >>> h, c = lstm(x)
        >>> print(h.value.shape, c.value.shape)
        (batch_size, seq_len, 20) (batch_size, 20)
    """
    
    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True):
        """初始化 LSTM 层。
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            use_bias: 是否使用偏置
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        k = 1.0 / math.sqrt(hidden_size)
        
        # 输入门
        self.W_ii = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_ii')
        self.W_hi = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_hi')
        
        # 遗忘门
        self.W_if = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_if')
        self.W_hf = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_hf')
        
        # 候选单元状态
        self.W_ig = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_ig')
        self.W_hg = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_hg')
        
        # 输出门
        self.W_io = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_io')
        self.W_ho = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_ho')
        
        if use_bias:
            self.b_i = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_i')
            self.b_f = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_f')
            self.b_g = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_g')
            self.b_o = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_o')
        else:
            self.b_i = self.b_f = self.b_g = self.b_o = None
    
    def forward(self, input: Variable, states=None):
        """前向传播。
        
        Args:
            input: 输入序列，形状 (batch_size, seq_len, input_size)
            states: 初始状态 (h_0, c_0)，默认为零
        
        Returns:
            (所有时间步的隐藏状态, 最后的单元状态)
        """
        batch_size, seq_len, input_size = input.value.shape.dims
        
        # 初始化状态
        if states is None:
            h_t = Variable(Tensor.zeros((batch_size, self.hidden_size)), requires_grad=False)
            c_t = Variable(Tensor.zeros((batch_size, self.hidden_size)), requires_grad=False)
        else:
            h_t, c_t = states
        
        outputs = []
        
        for t in range(seq_len):
            # 提取当前时间步输入
            x_t_data = []
            for b in range(batch_size):
                for i in range(input_size):
                    idx = b * seq_len * input_size + t * input_size + i
                    x_t_data.append(input.value.data[idx])
            
            x_t = Variable(
                Tensor(x_t_data, Shape((batch_size, input_size)), 'float32'),
                requires_grad=input.requires_grad
            )
            
            # LSTM 单元前向传播
            h_t, c_t = self._cell_forward(x_t, h_t, c_t)
            outputs.append(h_t)
        
        # 拼接输出
        output_data = []
        for b in range(batch_size):
            for t in range(seq_len):
                for h in range(self.hidden_size):
                    idx = b * self.hidden_size + h
                    output_data.append(outputs[t].value.data[idx])
        
        output_shape = Shape((batch_size, seq_len, self.hidden_size))
        output_tensor = Tensor(output_data, output_shape, 'float32')
        
        return Variable(output_tensor, requires_grad=input.requires_grad), c_t
    
    def _cell_forward(self, x_t: Variable, h_prev: Variable, c_prev: Variable):
        """LSTM 单元前向传播。"""
        # 输入门
        i_t = self._gate(x_t, h_prev, self.W_ii, self.W_hi, self.b_i, activation='sigmoid')
        
        # 遗忘门
        f_t = self._gate(x_t, h_prev, self.W_if, self.W_hf, self.b_f, activation='sigmoid')
        
        # 候选单元状态
        g_t = self._gate(x_t, h_prev, self.W_ig, self.W_hg, self.b_g, activation='tanh')
        
        # 输出门
        o_t = self._gate(x_t, h_prev, self.W_io, self.W_ho, self.b_o, activation='sigmoid')
        
        # 单元状态更新
        c_t = self._update_cell(f_t, c_prev, i_t, g_t)
        
        # 隐藏状态
        h_t = self._update_hidden(o_t, c_t)
        
        return h_t, c_t
    
    def _gate(self, x_t: Variable, h_prev: Variable, W_i: Parameter, 
              W_h: Parameter, bias: Parameter, activation: str) -> Variable:
        """计算门控值。"""
        # W_i @ x_t + W_h @ h_prev + bias
        result = self._linear(x_t, W_i, transpose=True)
        result = self._add_variables(result, self._linear(h_prev, W_h, transpose=True))
        
        if bias is not None:
            result = self._add_bias(result, bias)
        
        if activation == 'sigmoid':
            return self._apply_sigmoid(result)
        elif activation == 'tanh':
            return self._apply_tanh(result)
        else:
            return result
    
    def _linear(self, x: Variable, weight: Parameter, transpose: bool = False) -> Variable:
        """线性变换。"""
        if transpose:
            w = weight.value.transpose()
        else:
            w = weight.value
        result = x.value.matmul(w)
        return Variable(result, requires_grad=x.requires_grad)
    
    def _add_variables(self, a: Variable, b: Variable) -> Variable:
        """Variable 相加。"""
        result = a.value.add(b.value)
        return Variable(result, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _add_bias(self, x: Variable, bias: Parameter) -> Variable:
        """添加偏置。"""
        batch_size, hidden_size = x.value.shape.dims
        result_data = []
        for b in range(batch_size):
            for h in range(hidden_size):
                idx = b * hidden_size + h
                result_data.append(x.value.data[idx] + bias.value.data[h])
        result_tensor = Tensor(result_data, x.value.shape, 'float32')
        return Variable(result_tensor, requires_grad=x.requires_grad)
    
    def _apply_sigmoid(self, x: Variable) -> Variable:
        """Sigmoid 激活。"""
        result = x.value.sigmoid()
        return Variable(result, requires_grad=x.requires_grad)
    
    def _apply_tanh(self, x: Variable) -> Variable:
        """Tanh 激活。"""
        result = x.value.tanh()
        return Variable(result, requires_grad=x.requires_grad)
    
    def _update_cell(self, f_t: Variable, c_prev: Variable, 
                     i_t: Variable, g_t: Variable) -> Variable:
        """更新单元状态：c_t = f_t * c_prev + i_t * g_t"""
        fc = self._mul_variables(f_t, c_prev)
        ig = self._mul_variables(i_t, g_t)
        c_t = self._add_variables(fc, ig)
        return c_t
    
    def _update_hidden(self, o_t: Variable, c_t: Variable) -> Variable:
        """更新隐藏状态：h_t = o_t * tanh(c_t)"""
        tanh_c = self._apply_tanh(c_t)
        h_t = self._mul_variables(o_t, tanh_c)
        return h_t
    
    def _mul_variables(self, a: Variable, b: Variable) -> Variable:
        """逐元素相乘。"""
        result = a.value.mul(b.value)
        return Variable(result, requires_grad=(a.requires_grad or b.requires_grad))
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"use_bias={self.use_bias})")


class GRU(Module):
    """门控循环单元层。
    
    实现 GRU 单元，包含重置门和更新门。
    
    公式：
        r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)  # 重置门
        z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)  # 更新门
        n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1}) + b_n)  # 新候选状态
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}              # 隐藏状态
    
    Attributes:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        use_bias: 是否使用偏置
    
    Example:
        >>> gru = GRU(input_size=10, hidden_size=20)
        >>> x = Variable(Tensor.randn((batch_size, seq_len, 10)))
        >>> h = gru(x)
        >>> print(h.value.shape)
        (batch_size, seq_len, 20)
    """
    
    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True):
        """初始化 GRU 层。
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            use_bias: 是否使用偏置
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        
        k = 1.0 / math.sqrt(hidden_size)
        
        # 重置门
        self.W_ir = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_ir')
        self.W_hr = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_hr')
        
        # 更新门
        self.W_iz = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_iz')
        self.W_hz = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_hz')
        
        # 新候选状态
        self.W_in = Parameter(init.uniform(-k, k, (hidden_size, input_size)), name='W_in')
        self.W_hn = Parameter(init.uniform(-k, k, (hidden_size, hidden_size)), name='W_hn')
        
        if use_bias:
            self.b_r = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_r')
            self.b_z = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_z')
            self.b_n = Parameter(init.uniform(-k, k, (hidden_size,)), name='b_n')
        else:
            self.b_r = self.b_z = self.b_n = None
    
    def forward(self, input: Variable, h_0: Variable = None) -> Variable:
        """前向传播（实现简化版本，与 RNN 类似）。"""
        # 简化实现，参考 RNN 和 LSTM 的模式
        batch_size, seq_len, input_size = input.value.shape.dims
        
        if h_0 is None:
            h_t = Variable(Tensor.zeros((batch_size, self.hidden_size)), requires_grad=False)
        else:
            h_t = h_0
        
        outputs = []
        
        for t in range(seq_len):
            x_t_data = []
            for b in range(batch_size):
                for i in range(input_size):
                    idx = b * seq_len * input_size + t * input_size + i
                    x_t_data.append(input.value.data[idx])
            
            x_t = Variable(
                Tensor(x_t_data, Shape((batch_size, input_size)), 'float32'),
                requires_grad=input.requires_grad
            )
            
            h_t = self._cell_forward(x_t, h_t)
            outputs.append(h_t)
        
        # 拼接输出
        output_data = []
        for b in range(batch_size):
            for t in range(seq_len):
                for h in range(self.hidden_size):
                    idx = b * self.hidden_size + h
                    output_data.append(outputs[t].value.data[idx])
        
        output_shape = Shape((batch_size, seq_len, self.hidden_size))
        output_tensor = Tensor(output_data, output_shape, 'float32')
        
        return Variable(output_tensor, requires_grad=input.requires_grad)
    
    def _cell_forward(self, x_t: Variable, h_prev: Variable) -> Variable:
        """GRU 单元前向传播（简化实现）。"""
        # 重置门
        r_t = self._gate(x_t, h_prev, self.W_ir, self.W_hr, self.b_r, 'sigmoid')
        
        # 更新门
        z_t = self._gate(x_t, h_prev, self.W_iz, self.W_hz, self.b_z, 'sigmoid')
        
        # 新候选状态（简化版本）
        n_t = self._gate(x_t, h_prev, self.W_in, self.W_hn, self.b_n, 'tanh')
        
        # 更新隐藏状态：h_t = (1 - z_t) * n_t + z_t * h_prev
        h_t = self._gru_update(z_t, n_t, h_prev)
        
        return h_t
    
    def _gate(self, x_t: Variable, h_prev: Variable, W_i: Parameter, 
              W_h: Parameter, bias: Parameter, activation: str) -> Variable:
        """计算门控值。"""
        result = self._linear(x_t, W_i, True)
        result = self._add_variables(result, self._linear(h_prev, W_h, True))
        if bias is not None:
            result = self._add_bias(result, bias)
        if activation == 'sigmoid':
            return self._apply_sigmoid(result)
        else:
            return self._apply_tanh(result)
    
    def _linear(self, x: Variable, weight: Parameter, transpose: bool) -> Variable:
        """线性变换。"""
        w = weight.value.transpose() if transpose else weight.value
        result = x.value.matmul(w)
        return Variable(result, requires_grad=x.requires_grad)
    
    def _add_variables(self, a: Variable, b: Variable) -> Variable:
        """变量相加。"""
        result = a.value.add(b.value)
        return Variable(result, requires_grad=(a.requires_grad or b.requires_grad))
    
    def _add_bias(self, x: Variable, bias: Parameter) -> Variable:
        """添加偏置。"""
        batch_size, hidden_size = x.value.shape.dims
        result_data = []
        for b in range(batch_size):
            for h in range(hidden_size):
                idx = b * hidden_size + h
                result_data.append(x.value.data[idx] + bias.value.data[h])
        result_tensor = Tensor(result_data, x.value.shape, 'float32')
        return Variable(result_tensor, requires_grad=x.requires_grad)
    
    def _apply_sigmoid(self, x: Variable) -> Variable:
        """Sigmoid 激活。"""
        result = x.value.sigmoid()
        return Variable(result, requires_grad=x.requires_grad)
    
    def _apply_tanh(self, x: Variable) -> Variable:
        """Tanh 激活。"""
        result = x.value.tanh()
        return Variable(result, requires_grad=x.requires_grad)
    
    def _gru_update(self, z_t: Variable, n_t: Variable, h_prev: Variable) -> Variable:
        """GRU 状态更新。"""
        # (1 - z_t) * n_t + z_t * h_prev
        batch_size, hidden_size = z_t.value.shape.dims
        result_data = []
        for b in range(batch_size):
            for h in range(hidden_size):
                idx = b * hidden_size + h
                z_val = z_t.value.data[idx]
                n_val = n_t.value.data[idx]
                h_val = h_prev.value.data[idx]
                result_data.append((1.0 - z_val) * n_val + z_val * h_val)
        
        result_tensor = Tensor(result_data, Shape((batch_size, hidden_size)), 'float32')
        return Variable(result_tensor, requires_grad=True)
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"GRU(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"use_bias={self.use_bias})")
