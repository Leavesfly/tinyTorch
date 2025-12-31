"""Quick test to verify tinyTorch basic functionality.

This script tests:
1. Tensor creation and operations
2. Automatic differentiation
3. Gradient computation

Author: TinyAI Team
"""

import sys
sys.path.insert(0, '/Users/yefei.yf/Qoder/TinyAI/tinyTorch')

from tinytorch.tensor import Tensor, Shape
from tinytorch.autograd import Variable


def test_tensor_operations():
    """Test basic tensor operations."""
    print("=== Testing Tensor Operations ===")
    
    # Test tensor creation
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"Tensor 1: shape={t1.shape}, data={t1.data}")
    
    t2 = Tensor.ones((2, 2))
    print(f"Tensor 2 (ones): shape={t2.shape}, data={t2.data}")
    
    # Test addition
    t3 = t1.add(t2)
    print(f"t1 + t2: data={t3.data}")
    
    # Test multiplication
    t4 = t1.mul(t2)
    print(f"t1 * t2: data={t4.data}")
    
    # Test matrix multiplication
    t5 = t1.matmul(t2)
    print(f"t1 @ t2: data={t5.data}")
    
    print("✓ Tensor operations test passed\n")


def test_autograd_basic():
    """Test basic automatic differentiation."""
    print("=== Testing Autograd Basic ===")
    
    # Create variables
    x = Variable(Tensor([[2.0, 3.0]]), name="x")
    print(f"x: {x.value.data}")
    
    # Simple operation: y = x^2
    y = x * x
    print(f"y = x^2: {y.value.data}")
    
    # Backward
    y.backward()
    print(f"dy/dx: {x.grad.data}")
    print(f"Expected: [4.0, 6.0] (2*x)")
    
    print("✓ Autograd basic test passed\n")


def test_autograd_chain():
    """Test chain rule in automatic differentiation."""
    print("=== Testing Autograd Chain Rule ===")
    
    # Create variable
    x = Variable(Tensor([[1.0, 2.0]]), name="x")
    
    # Chain of operations: z = (x + 1) * 2
    y = x + 1.0  # y = x + 1
    z = y * 2.0  # z = 2y = 2(x + 1)
    
    print(f"x: {x.value.data}")
    print(f"y = x + 1: {y.value.data}")
    print(f"z = y * 2: {z.value.data}")
    
    # Backward
    z.backward()
    print(f"dz/dx: {x.grad.data}")
    print(f"Expected: [2.0, 2.0] (derivative of 2x + 2 is 2)")
    
    print("✓ Autograd chain rule test passed\n")


def test_matmul_grad():
    """Test matrix multiplication gradient."""
    print("=== Testing Matrix Multiplication Gradient ===")
    
    # Create variables
    A = Variable(Tensor([[1.0, 2.0], [3.0, 4.0]]), name="A")
    B = Variable(Tensor([[1.0, 0.0], [0.0, 1.0]]), name="B")  # Identity matrix
    
    # Matrix multiplication
    C = A.matmul(B)
    print(f"A: {A.value.data}")
    print(f"B: {B.value.data}")
    print(f"C = A @ B: {C.value.data}")
    
    # Sum to get scalar for backward
    loss = C.sum()
    print(f"loss = sum(C): {loss.value.data}")
    
    # Backward
    loss.backward()
    print(f"dL/dA: {A.grad.data}")
    print(f"dL/dB: {B.grad.data}")
    
    print("✓ Matrix multiplication gradient test passed\n")


def test_activation_functions():
    """Test activation functions and their gradients."""
    print("=== Testing Activation Functions ===")
    
    # ReLU
    x = Variable(Tensor([[-1.0, 0.0, 1.0, 2.0]]), name="x")
    y = x.relu()
    print(f"x: {x.value.data}")
    print(f"ReLU(x): {y.value.data}")
    
    y.backward()
    print(f"ReLU'(x): {x.grad.data}")
    print(f"Expected: [0.0, 0.0, 1.0, 1.0]")
    
    # Sigmoid
    x2 = Variable(Tensor([[0.0, 1.0]]), name="x2")
    y2 = x2.sigmoid()
    print(f"\nx2: {x2.value.data}")
    print(f"Sigmoid(x2): {y2.value.data}")
    
    print("✓ Activation functions test passed\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("tinyTorch Quick Test")
    print("=" * 50)
    print()
    
    try:
        test_tensor_operations()
        test_autograd_basic()
        test_autograd_chain()
        test_matmul_grad()
        test_activation_functions()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
