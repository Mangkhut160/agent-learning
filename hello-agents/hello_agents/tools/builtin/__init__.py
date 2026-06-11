"""
内置工具集
===========

本模块包含HelloAgents框架内置的工具实现。

已内置的工具:
- CalculatorTool: 数学计算器，支持基本运算和常见数学函数
- SearchTool: 搜索工具，支持Tavily和SerpAPI多源搜索

使用示例:
    from hello_agents.tools.builtin import CalculatorTool, SearchTool

    # 创建计算器工具
    calculator = CalculatorTool()

    # 创建搜索工具
    search = SearchTool(backend="hybrid")
"""

from .calculator import CalculatorTool
from .search import SearchTool

__all__ = [
    "CalculatorTool",
    "SearchTool",
]