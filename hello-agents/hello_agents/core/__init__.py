"""
核心模块初始化
==============

核心模块包含框架的基础设施组件：

- llm: 统一的大语言模型接口，支持多提供商自动检测
- message: 标准化消息系统
- config: 配置管理，支持环境变量和默认值
- agent: Agent抽象基类，定义所有Agent的接口规范
- exceptions: 统一的异常体系

这些组件共同构成了HelloAgents框架的技术底座。
"""

from .llm import HelloAgentsLLM
from .message import Message, MessageRole
from .config import Config
from .agent import Agent
from .exceptions import (
    HelloAgentsError,
    LLMPricingError,
    ToolExecutionError,
    AgentRuntimeError
)

__all__ = [
    "HelloAgentsLLM",
    "Message",
    "MessageRole",
    "Config",
    "Agent",
    "HelloAgentsError",
    "LLMPricingError",
    "ToolExecutionError",
    "AgentRuntimeError",
]