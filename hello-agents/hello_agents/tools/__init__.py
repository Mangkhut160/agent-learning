"""
工具系统初始化
==============

工具系统是HelloAgents框架中Agent能力扩展的核心模块。

核心组件:
- base: Tool基类和ToolParameter参数定义
- registry: ToolRegistry工具注册表，管理所有可用工具

设计理念:
1. 统一接口：所有工具都实现Tool接口，确保调用方式一致
2. 自描述能力：工具能够告诉系统自己需要什么参数
3. 灵活注册：支持Tool对象和函数两种注册方式

使用示例:
    from hello_agents.tools import ToolRegistry

    registry = ToolRegistry()

    # 注册Tool对象
    registry.register_tool(my_tool)

    # 或注册函数
    registry.register_function("calc", "计算器", my_calc_func)

    # 执行工具
    result = registry.execute_tool("calc", "2+3")
"""

from .base import Tool, ToolParameter
from .registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolRegistry",
]