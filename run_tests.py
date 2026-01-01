#!/usr/bin/env python
"""运行 tinyTorch 测试套件的脚本。

使用方法:
    python run_tests.py              # 运行所有测试
    python run_tests.py test_tensor  # 运行特定测试文件
    python run_tests.py -v           # 详细输出

Author: TinyAI Team
"""

import sys
import subprocess
import os


def check_pytest():
    """检查 pytest 是否安装。"""
    try:
        import pytest
        return True
    except ImportError:
        return False


def install_pytest():
    """安装测试依赖。"""
    print("正在安装测试依赖...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytest'])
        print("✓ pytest 安装完成")
        return True
    except subprocess.CalledProcessError:
        print("✗ pytest 安装失败")
        return False


def run_tests(args):
    """运行测试。"""
    # 确保 pytest 已安装
    if not check_pytest():
        print("未检测到 pytest，尝试安装...")
        if not install_pytest():
            print("\n请手动安装 pytest:")
            print("  pip install pytest")
            return 1
    
    # 导入 pytest
    import pytest
    
    # 构建测试参数
    test_args = ['tests/']
    
    # 处理命令行参数
    if args:
        # 如果指定了测试文件，运行特定文件
        if any(arg.startswith('test_') for arg in args):
            test_args = [f'tests/{arg}.py' if not arg.endswith('.py') else f'tests/{arg}' 
                        for arg in args if arg.startswith('test_')]
        # 添加其他参数（如 -v）
        test_args.extend([arg for arg in args if not arg.startswith('test_')])
    else:
        # 默认详细输出
        test_args.append('-v')
    
    # 运行测试
    print(f"\n运行测试: {' '.join(test_args)}\n")
    print("=" * 70)
    
    return pytest.main(test_args)


def main():
    """主函数。"""
    print("=" * 70)
    print("tinyTorch 测试套件")
    print("=" * 70)
    
    # 获取命令行参数
    args = sys.argv[1:]
    
    # 显示帮助
    if '-h' in args or '--help' in args:
        print(__doc__)
        return 0
    
    # 运行测试
    result = run_tests(args)
    
    if result == 0:
        print("\n" + "=" * 70)
        print("✓ 所有测试通过！")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ 部分测试失败")
        print("=" * 70)
    
    return result


if __name__ == '__main__':
    sys.exit(main())
