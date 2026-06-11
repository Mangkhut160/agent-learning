"""
HelloAgents 安装配置文件
========================

此文件用于构建和发布 hello-agents Python包。

安装方式:
    pip install hello-agents           # 从PyPI安装
    pip install -e .                  # 开发模式安装
    python setup.py install           # 从源码安装
"""

from setuptools import setup, find_packages

# 读取 README 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hello-agents",
    version="0.1.1",
    author="HelloAgents Team",
    author_email="hello@helloagents.dev",
    description="一个轻量级、可扩展的AI智能体开发框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjyaoao/HelloAgents",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "search": [
            "tavily>=0.2.0",
            "serpapi>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords=[
        "ai",
        "agent",
        "llm",
        "openai",
        "gpt",
        "artificial-intelligence",
        "chatbot",
        "autonomous-agent",
    ],
    project_urls={
        "Bug Reports": "https://github.com/jjyaoao/HelloAgents/issues",
        "Source": "https://github.com/jjyaoao/HelloAgents",
        "Documentation": "https://github.com/jjyaoao/HelloAgents#readme",
    },
)