"""
HelloAgents - 轻量级智能体框架
===============================

一个模块化、可扩展的AI智能体开发框架，支持多种Agent范式和工具系统。

主要特性:
- 统一的LLM接口，支持多提供商（OpenAI、ModelScope、智谱AI、VLLM、Ollama等）
- 标准化消息系统
- 灵活的工具注册与执行机制
- 多种Agent范式：SimpleAgent、ReActAgent、ReflectionAgent、PlanAndSolveAgent

快速开始:
    from hello_agents import HelloAgentsLLM, SimpleAgent

    llm = HelloAgentsLLM()
    agent = SimpleAgent(name="助手", llm=llm, system_prompt="你是一个有用的AI助手")
    response = agent.run("你好！")

版本: 0.1.1
Python版本: >= 3.10
"""

__version__ = "0.1.1"
__author__ = "HelloAgents Team"

# 核心组件导出
from .core.llm import HelloAgentsLLM
from .core.message import Message, MessageRole
from .core.config import Config
from .core.agent import Agent
from .core.exceptions import (
    HelloAgentsError,
    LLMPricingError,
    ToolExecutionError,
    AgentRuntimeError
)

# Agent实现导出
from .agents.simple_agent import SimpleAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanAndSolveAgent

# 工具系统导出
from .tools.base import Tool, ToolParameter
from .tools.registry import ToolRegistry

# 配置管理
from .core.config import Config

__all__ = [
    # 版本信息
    "__version__",

    # 核心组件
    "HelloAgentsLLM",
    "Message",
    "MessageRole",
    "Config",
    "Agent",

    # 异常类
    "HelloAgentsError",
    "LLMPricingError",
    "ToolExecutionError",
    "AgentRuntimeError",

    # Agent实现
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",

    # 工具系统
    "Tool",
    "ToolParameter",
    "ToolRegistry",
]