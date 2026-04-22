"""测试辅助工具。

本模块为单元测试集中提供:
    - gradcheck: 基于中心差分的数值梯度校验，对比解析梯度与数值梯度
    - tensor / ndarray 构造快捷方法: 减少测试样板代码
    - assert_ndarray_close: 浮点数组近似相等断言

这些工具仅服务于 `tests/` 目录，不属于公共 API。
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr import NdArray

Number = Union[int, float]


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------

def make_ndarray(data, dtype: str = "float32") -> NdArray:
    """从嵌套 Python 列表快速构造 NdArray。"""
    return NdArray(data, dtype=dtype)


def make_tensor(
    data,
    requires_grad: bool = True,
    name: str = None,
    dtype: str = "float32",
) -> Tensor:
    """从嵌套 Python 列表快速构造 Tensor。"""
    return Tensor(NdArray(data, dtype=dtype), name=name, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# 近似断言
# ---------------------------------------------------------------------------

def assert_close(
    actual: Number,
    expected: Number,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    msg: str = "",
) -> None:
    """单个浮点数近似相等断言。

    判定条件: |actual - expected| <= atol + rtol * |expected|

    Args:
        actual: 实际值
        expected: 期望值
        rtol: 相对容差
        atol: 绝对容差
        msg: 失败时附加的调试信息

    Raises:
        AssertionError: 差异超过容差时抛出。
    """
    diff = abs(actual - expected)
    threshold = atol + rtol * abs(expected)
    if diff > threshold:
        raise AssertionError(
            f"assert_close failed: |{actual} - {expected}| = {diff} > {threshold}. {msg}"
        )


def assert_ndarray_close(
    actual: NdArray,
    expected: NdArray,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """逐元素比较两个 NdArray 是否近似相等。

    Args:
        actual: 实际输出的 NdArray
        expected: 期望输出的 NdArray
        rtol: 相对容差
        atol: 绝对容差

    Raises:
        AssertionError: 形状不匹配或任一元素超出容差时抛出。
    """
    if actual.shape.dims != expected.shape.dims:
        raise AssertionError(
            f"shape mismatch: {actual.shape.dims} vs {expected.shape.dims}"
        )
    for i, (a, e) in enumerate(zip(actual.data, expected.data)):
        diff = abs(a - e)
        threshold = atol + rtol * abs(e)
        if diff > threshold:
            raise AssertionError(
                f"ndarray mismatch at flat index {i}: "
                f"actual={a} expected={e} diff={diff} threshold={threshold}"
            )


# ---------------------------------------------------------------------------
# 数值梯度工具 (gradcheck)
# ---------------------------------------------------------------------------

def _clone_ndarray(arr: NdArray) -> NdArray:
    """深拷贝一个 NdArray，保持数据/形状/ dtype 一致。"""
    return NdArray(list(arr.data), arr.shape.dims, arr.dtype)


def numerical_grad(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-4,
) -> List[NdArray]:
    """使用中心差分计算标量输出函数对每个输入的数值梯度。

    要求:
        - fn(*inputs) 返回一个标量 Tensor（size == 1）。
        - inputs 中每个 Tensor 都会被视作待求导变量；
          函数不会修改原 Tensor.value 的内容（运算结束后恢复）。

    算法:
        对输入 x 的每个元素 x_i 执行:
            (fn(x + eps*e_i) - fn(x - eps*e_i)) / (2 * eps)

    Args:
        fn: 从若干 Tensor 映射到单个标量 Tensor 的函数。
        inputs: 需要数值梯度的输入 Tensor 列表。
        eps: 差分步长，默认 1e-4（float32 下经验值）。

    Returns:
        数值梯度列表，与 `inputs` 一一对应，形状同对应输入。
    """
    grads: List[NdArray] = []
    for k, inp in enumerate(inputs):
        original = _clone_ndarray(inp.value)
        grad_data: List[float] = [0.0] * original.shape.size

        for i in range(original.shape.size):
            # f(x + eps)
            inp.value.data[i] = original.data[i] + eps
            plus = fn(*inputs)
            f_plus = plus.value.data[0] if plus.value.shape.size == 1 else sum(plus.value.data)

            # f(x - eps)
            inp.value.data[i] = original.data[i] - eps
            minus = fn(*inputs)
            f_minus = minus.value.data[0] if minus.value.shape.size == 1 else sum(minus.value.data)

            grad_data[i] = (f_plus - f_minus) / (2.0 * eps)

            # 恢复当前元素
            inp.value.data[i] = original.data[i]

        grads.append(NdArray(grad_data, original.shape.dims, original.dtype))

    return grads


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-4,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> None:
    """对比解析梯度与中心差分数值梯度，验证 backward 实现正确性。

    使用方式:
        >>> x = make_tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = make_tensor([[0.5, 1.5], [2.5, 3.5]])
        >>> gradcheck(lambda a, b: (a * b).sum(), [x, y])

    纯 Python 实现的数值精度有限，建议:
        - eps 取 1e-4 ~ 1e-3
        - rtol 取 1e-2，atol 取 1e-3（float32 的典型组合）

    Args:
        fn: 从若干 Tensor 映射到单个标量 Tensor 的函数。
        inputs: 参与检查的输入 Tensor；requires_grad 将被强制置 True。
        eps: 数值差分步长
        rtol: 相对容差
        atol: 绝对容差

    Raises:
        AssertionError: 解析梯度与数值梯度差异超出容差时抛出。
    """
    # 强制开启梯度追踪并清空历史梯度
    for inp in inputs:
        inp.requires_grad = True
        inp.grad = None

    # 解析梯度
    output = fn(*inputs)
    if output.value.shape.size != 1:
        raise ValueError(
            f"gradcheck requires scalar output, got shape={output.value.shape.dims}"
        )
    output.backward()
    analytic = [
        inp.grad if inp.grad is not None else NdArray(
            [0.0] * inp.value.shape.size, inp.value.shape.dims, inp.value.dtype
        )
        for inp in inputs
    ]

    # 清理梯度再走数值路径，避免数值评估时残留 creator 干扰
    for inp in inputs:
        inp.grad = None
        inp.creator = None

    numeric = numerical_grad(fn, inputs, eps=eps)

    # 逐个输入比较
    for idx, (a, n) in enumerate(zip(analytic, numeric)):
        if a.shape.dims != n.shape.dims:
            raise AssertionError(
                f"input[{idx}] grad shape mismatch: analytic={a.shape.dims} "
                f"numeric={n.shape.dims}"
            )
        for flat_idx, (av, nv) in enumerate(zip(a.data, n.data)):
            diff = abs(av - nv)
            threshold = atol + rtol * abs(nv)
            if diff > threshold:
                raise AssertionError(
                    f"gradcheck failed at input[{idx}] flat[{flat_idx}]: "
                    f"analytic={av} numeric={nv} diff={diff} threshold={threshold}"
                )


__all__ = [
    "make_ndarray",
    "make_tensor",
    "assert_close",
    "assert_ndarray_close",
    "numerical_grad",
    "gradcheck",
]
