"""循环神经网络层。

Author: TinyAI Team
"""

import math
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd import Tensor
from tinytorch.autograd.ops.nn import StackTime as _StackTime
from tinytorch.autograd.ops.nn import TimeSlice as _TimeSlice
from tinytorch.ndarr import NdArray
from tinytorch.nn import init


class RNNBase(Module):
    """RNN 系列层的公共基类。

    提供门控权重创建、门控计算和通用的时间步循环逻辑。
    子类只需实现 ``_cell_forward`` 即可定义具体的循环单元行为，
    可选覆写 ``_init_states`` / ``_extract_hidden`` / ``_pack_output``
    来定制状态初始化和输出格式。
    """

    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True) -> None:
        """初始化 RNN 基类的公共属性。

        参数:
            input_size: 输入特征维度。
            hidden_size: 隐藏状态维度。
            use_bias: 是否使用偏置。
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    # ------------------------------------------------------------------
    # 权重创建与门控计算
    # ------------------------------------------------------------------

    def _create_gate_weights(self, gate_name: str, input_size: int,
                             hidden_size: int, bound: float,
                             create_bias: bool) -> None:
        """为一个门创建输入权重、隐藏权重和可选偏置。

        创建的参数会以 ``W_i{gate_name}``、``W_h{gate_name}``、
        ``b_{gate_name}`` 的命名注册到模块上。

        参数:
            gate_name: 门的后缀标识（如 'i', 'f', 'r'）
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            bound: 均匀分布初始化的范围 [-bound, bound]
            create_bias: 是否创建偏置参数
        """
        setattr(self, f'W_i{gate_name}',
                Parameter(init.uniform(-bound, bound, (hidden_size, input_size)),
                          name=f'W_i{gate_name}'))
        setattr(self, f'W_h{gate_name}',
                Parameter(init.uniform(-bound, bound, (hidden_size, hidden_size)),
                          name=f'W_h{gate_name}'))
        if create_bias:
            setattr(self, f'b_{gate_name}',
                    Parameter(init.uniform(-bound, bound, (hidden_size,)),
                              name=f'b_{gate_name}'))
        else:
            setattr(self, f'b_{gate_name}', None)

    def _gate(self, x_t: Tensor, h_prev: Tensor, W_i: Parameter,
              W_h: Parameter, bias: 'Parameter | None',
              activation: str) -> Tensor:
        """计算门控值：activation(x_t @ W_i.T + h_prev @ W_h.T + bias)"""
        result = x_t.matmul(W_i.transpose()) + h_prev.matmul(W_h.transpose())
        if bias is not None:
            result = result + bias
        if activation == 'sigmoid':
            return result.sigmoid()
        return result.tanh()

    # ------------------------------------------------------------------
    # 通用前向传播（模板方法模式）
    # ------------------------------------------------------------------

    def _validate_input(self, input: Tensor) -> 'tuple[int, int, int]':
        """校验输入形状并返回 (batch_size, seq_len, input_size)。"""
        batch_size, seq_len, input_size = input.value.shape.dims
        if input_size != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {input_size}"
            )
        return batch_size, seq_len, input_size

    def _init_hidden(self, batch_size: int) -> Tensor:
        """创建全零的隐藏状态张量。"""
        return Tensor(
            NdArray.zeros((batch_size, self.hidden_size)),
            requires_grad=False,
        )

    def forward(self, input: Tensor, initial_states=None):
        """通用的 RNN 前向传播：校验 → 初始化 → 时间步循环 → 组装输出。

        子类通过覆写钩子方法来定制行为：
        - ``_init_states``: 初始化循环状态
        - ``_cell_forward``: 单个时间步的计算
        - ``_extract_hidden``: 从状态中提取 h_t 用于输出收集
        - ``_pack_output``: 组装最终返回值

        参数:
            input: 输入序列，形状 (batch_size, seq_len, input_size)。
            initial_states: 初始隐藏状态，格式由子类决定。

        返回:
            由 ``_pack_output`` 决定的输出格式。
        """
        batch_size, seq_len, input_size = self._validate_input(input)
        states = self._init_states(batch_size, initial_states)

        hidden_outputs = []
        for time_step in range(seq_len):
            x_t = _TimeSlice(time_step, seq_len, input_size)(input)
            states = self._cell_forward(x_t, states)
            hidden_outputs.append(self._extract_hidden(states))

        stacked_output = _StackTime(self.hidden_size)(*hidden_outputs)
        return self._pack_output(stacked_output, states)

    def _init_states(self, batch_size: int, initial_states):
        """初始化循环状态。子类可覆写以支持多状态（如 LSTM 的 h, c）。"""
        if initial_states is not None:
            return initial_states
        return self._init_hidden(batch_size)

    def _cell_forward(self, x_t: Tensor, states):
        """单个时间步的前向传播，子类必须实现。"""
        raise NotImplementedError

    def _extract_hidden(self, states) -> Tensor:
        """从状态中提取隐藏状态 h_t，用于收集每个时间步的输出。"""
        return states

    def _pack_output(self, stacked_output: Tensor, final_states):
        """组装最终输出，子类可覆写以返回额外状态（如 LSTM 的 cell state）。"""
        return stacked_output


class RNN(RNNBase):
    """基础循环神经网络层。

    实现标准的 RNN 单元：h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)

    Example:
        >>> rnn = RNN(input_size=10, hidden_size=20)
        >>> x = Tensor(NdArray.randn((batch_size, seq_len, 10)))
        >>> h = rnn(x)
        >>> print(h.value.shape)
        (batch_size, seq_len, 20)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, use_bias)

        bound = 1.0 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.uniform(-bound, bound, (hidden_size, input_size)), name='W_ih')
        self.W_hh = Parameter(
            init.uniform(-bound, bound, (hidden_size, hidden_size)), name='W_hh')
        self.bias = (
            Parameter(init.uniform(-bound, bound, (hidden_size,)), name='bias')
            if use_bias else None
        )

    def _cell_forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """单个 RNN 单元的前向传播。

        参数:
            x_t: 当前时间步输入，形状 (batch_size, input_size)。
            h_prev: 上一时间步隐藏状态，形状 (batch_size, hidden_size)。

        返回:
            当前时间步隐藏状态，形状 (batch_size, hidden_size)。
        """
        h_t = x_t.matmul(self.W_ih.transpose()) + h_prev.matmul(self.W_hh.transpose())
        if self.use_bias:
            h_t = h_t + self.bias
        return h_t.tanh()

    def __repr__(self) -> str:
        return (f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"use_bias={self.use_bias})")


class LSTM(RNNBase):
    """长短期记忆网络层。

    实现 LSTM 单元，包含输入门、遗忘门、输出门和单元状态。

    公式：
        i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
        f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
        g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
        o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    Example:
        >>> lstm = LSTM(input_size=10, hidden_size=20)
        >>> x = Tensor(NdArray.randn((batch_size, seq_len, 10)))
        >>> h, c = lstm(x)
        >>> print(h.value.shape, c.value.shape)
        (batch_size, seq_len, 20) (batch_size, 20)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, use_bias)

        bound = 1.0 / math.sqrt(hidden_size)
        for gate_name in ('i', 'f', 'g', 'o'):
            self._create_gate_weights(gate_name, input_size, hidden_size, bound, use_bias)

    def _init_states(self, batch_size: int, initial_states):
        """初始化 (h_0, c_0) 状态对。"""
        if initial_states is not None:
            return initial_states
        return self._init_hidden(batch_size), self._init_hidden(batch_size)

    def _cell_forward(self, x_t: Tensor, states: tuple) -> tuple:
        """LSTM 单元前向传播。"""
        h_prev, c_prev = states

        input_gate = self._gate(x_t, h_prev, self.W_ii, self.W_hi, self.b_i, 'sigmoid')
        forget_gate = self._gate(x_t, h_prev, self.W_if, self.W_hf, self.b_f, 'sigmoid')
        candidate = self._gate(x_t, h_prev, self.W_ig, self.W_hg, self.b_g, 'tanh')
        output_gate = self._gate(x_t, h_prev, self.W_io, self.W_ho, self.b_o, 'sigmoid')

        cell_state = forget_gate * c_prev + input_gate * candidate
        hidden_state = output_gate * cell_state.tanh()
        return hidden_state, cell_state

    def _extract_hidden(self, states: tuple) -> Tensor:
        """从 (h_t, c_t) 中提取 h_t。"""
        return states[0]

    def _pack_output(self, stacked_output: Tensor, final_states: tuple):
        """返回 (所有时间步的隐藏状态, 最后的单元状态)。"""
        return stacked_output, final_states[1]

    def __repr__(self) -> str:
        return (f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"use_bias={self.use_bias})")


class GRU(RNNBase):
    """门控循环单元层。

    实现 GRU 单元，包含重置门和更新门。

    公式：
        r_t = sigmoid(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
        z_t = sigmoid(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
        n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1}) + b_n)
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}

    Example:
        >>> gru = GRU(input_size=10, hidden_size=20)
        >>> x = Tensor(NdArray.randn((batch_size, seq_len, 10)))
        >>> h = gru(x)
        >>> print(h.value.shape)
        (batch_size, seq_len, 20)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True) -> None:
        super().__init__(input_size, hidden_size, use_bias)

        bound = 1.0 / math.sqrt(hidden_size)
        for gate_name in ('r', 'z', 'n'):
            self._create_gate_weights(gate_name, input_size, hidden_size, bound, use_bias)

    def _cell_forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """GRU 单元前向传播。"""
        reset_gate = self._gate(x_t, h_prev, self.W_ir, self.W_hr, self.b_r, 'sigmoid')
        update_gate = self._gate(x_t, h_prev, self.W_iz, self.W_hz, self.b_z, 'sigmoid')

        # 新候选状态：n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_prev) + b_n)
        candidate = x_t.matmul(self.W_in.transpose()) + reset_gate * h_prev.matmul(self.W_hn.transpose())
        if self.b_n is not None:
            candidate = candidate + self.b_n
        candidate = candidate.tanh()

        # 更新隐藏状态：h_t = (1 - z_t) * n_t + z_t * h_prev
        one_minus_update = (update_gate * -1.0) + 1.0
        return one_minus_update * candidate + update_gate * h_prev

    def __repr__(self) -> str:
        return (f"GRU(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"use_bias={self.use_bias})")
