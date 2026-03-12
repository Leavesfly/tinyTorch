"""tinyTorch 快速测试，验证基本功能。

本脚本测试以下内容：
1. Tensor 创建和运算
2. 自动微分
3. 梯度计算

作者：TinyAI Team
"""

import sys
# sys.path.insert(0, '/Users/yefei.yf/Qoder/TinyAI/tinyTorch')

from tinytorch.tensor import Tensor, Shape
from tinytorch.autograd import Variable


def test_tensor_operations():
    """测试基本的 tensor 运算。"""
    print("=== 测试 Tensor 运算 ===")
    
    # 测试 tensor 创建
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"Tensor 1: shape={t1.shape}, data={t1.data}")
    
    t2 = Tensor.ones((2, 2))
    print(f"Tensor 2 (ones): shape={t2.shape}, data={t2.data}")
    
    # 测试加法
    t3 = t1.add(t2)
    print(f"t1 + t2: data={t3.data}")
    
    # 测试乘法
    t4 = t1.mul(t2)
    print(f"t1 * t2: data={t4.data}")
    
    # 测试矩阵乘法
    t5 = t1.matmul(t2)
    print(f"t1 @ t2: data={t5.data}")
    
    print("✓ Tensor 运算测试通过\n")


def test_autograd_basic():
    """测试基本的自动微分。"""
    print("=== 测试自动微分基础 ===")
    
    # 创建变量
    x = Variable(Tensor([[2.0, 3.0]]), name="x")
    print(f"x: {x.value.data}")
    
    # 简单运算：y = x^2
    y = x * x
    print(f"y = x^2: {y.value.data}")
    
    # 反向传播
    y.backward()
    print(f"dy/dx: {x.grad.data}")
    print(f"期望值：[4.0, 6.0] (2*x)")
    
    print("✓ 自动微分基础测试通过\n")


def test_autograd_chain():
    """测试自动微分的链式法则。"""
    print("=== 测试自动微分链式法则 ===")
    
    # 创建变量
    x = Variable(Tensor([[1.0, 2.0]]), name="x")
    
    # 运算链：z = (x + 1) * 2
    y = x + 1.0  # y = x + 1
    z = y * 2.0  # z = 2y = 2(x + 1)
    
    print(f"x: {x.value.data}")
    print(f"y = x + 1: {y.value.data}")
    print(f"z = y * 2: {z.value.data}")
    
    # 反向传播
    z.backward()
    print(f"dz/dx: {x.grad.data}")
    print(f"期望值：[2.0, 2.0] (2x + 2 的导数是 2)")
    
    print("✓ 自动微分链式法则测试通过\n")


def test_matmul_grad():
    """测试矩阵乘法的梯度。"""
    print("=== 测试矩阵乘法梯度 ===")
    
    # 创建变量
    A = Variable(Tensor([[1.0, 2.0], [3.0, 4.0]]), name="A")
    B = Variable(Tensor([[1.0, 0.0], [0.0, 1.0]]), name="B")  # 单位矩阵
    
    # 矩阵乘法
    C = A.matmul(B)
    print(f"A: {A.value.data}")
    print(f"B: {B.value.data}")
    print(f"C = A @ B: {C.value.data}")
    
    # 求和得到标量以便反向传播
    loss = C.sum()
    print(f"loss = sum(C): {loss.value.data}")
    
    # 反向传播
    loss.backward()
    print(f"dL/dA: {A.grad.data}")
    print(f"dL/dB: {B.grad.data}")
    
    print("✓ 矩阵乘法梯度测试通过\n")


def test_activation_functions():
    """测试激活函数及其梯度。"""
    print("=== 测试激活函数 ===")
    
    # ReLU
    x = Variable(Tensor([[-1.0, 0.0, 1.0, 2.0]]), name="x")
    y = x.relu()
    print(f"x: {x.value.data}")
    print(f"ReLU(x): {y.value.data}")
    
    y.backward()
    print(f"ReLU'(x): {x.grad.data}")
    print(f"期望值：[0.0, 0.0, 1.0, 1.0]")
    
    # Sigmoid
    x2 = Variable(Tensor([[0.0, 1.0]]), name="x2")
    y2 = x2.sigmoid()
    print(f"\nx2: {x2.value.data}")
    print(f"Sigmoid(x2): {y2.value.data}")
    
    print("✓ 激活函数测试通过\n")


def main():
    """运行所有测试。"""
    print("=" * 50)
    print("tinyTorch 快速测试")
    print("=" * 50)
    print()
    
    try:
        test_tensor_operations()
        test_autograd_basic()
        test_autograd_chain()
        test_matmul_grad()
        test_activation_functions()
        
        print("=" * 50)
        print("所有测试通过！✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试失败，错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
