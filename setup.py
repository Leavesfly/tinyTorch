"""tinyTorch 安装配置脚本。"""

from setuptools import setup, find_packages
import os

# 读取 README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "A lightweight deep learning framework implemented in pure Python"

setup(
    name="tinytorch",
    version="0.1.0",
    author="TinyAI Team",
    author_email="",
    description="轻量级深度学习框架 - 纯 Python 实现",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leavesfly/TinyAI",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.7",
    install_requires=[
        # 无外部依赖 - 纯 Python 实现
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "flake8>=3.9",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
