"""
工具基类与参数定义
==================

本模块定义了工具系统的核心抽象，包括：
- Tool: 所有工具的基类，定义统一接口
- ToolParameter: 工具参数定义，支持类型检查和文档生成

设计原则:
1. 单一职责：每个Tool只负责一个功能
2. 统一接口：通过run方法统一执行入口
3. 自描述能力：通过get_parameters告知调用者参数需求

使用示例:
    from hello_agents.tools.base import Tool, ToolParameter
    from typing import List

    class CalculatorTool(Tool):
        def __init__(self):
            super().__init__(
                name="calculator",
                description="执行数学计算"
            )

        def run(self, parameters: dict) -> str:
            # 实现计算逻辑
            pass

        def get_parameters(self) -> List[ToolParameter]:
            return [
                ToolParameter(
                    name="expression",
                    type="string",
                    description="数学表达式，如 2+3*4"
                )
            ]
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """
    工具参数定义

    用于描述工具所需的参数信息，包括参数名、类型、描述、是否必需等。
    这些信息用于：
    - 生成工具文档
    - 验证输入参数
    - 构建OpenAI function calling schema

    Attributes:
        name: 参数名称
        type: 参数类型（string, number, boolean, array等）
        description: 参数描述，说明参数的用途和格式
        required: 是否必需，默认为True
        default: 默认值（可选）
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """
    工具基类 - 所有工具的抽象基类

    Tool类是HelloAgents工具系统的核心抽象，定义了所有工具必须
    遵循的接口规范。任何想要被Agent使用的工具都必须继承此类
    并实现其抽象方法。

    设计要点:
    1. 统一接口：通过run方法执行工具，接受字典参数并返回字符串结果
    2. 自描述能力：通过get_parameters方法声明参数需求
    3. 元数据：name和description用于工具发现和文档生成

    Abstract Methods:
        run: 执行工具的核心方法，子类必须实现
        get_parameters: 返回参数定义列表，子类必须实现

    Example:
        class MyTool(Tool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="我的自定义工具"
                )

            def run(self, parameters: Dict[str, Any]) -> str:
                expr = parameters.get("expression", "")
                # 执行逻辑
                return result

            def get_parameters(self) -> List[ToolParameter]:
                return [
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="要计算的表达式"
                    )
                ]
    """

    def __init__(self, name: str, description: str):
        """
        初始化工具

        Args:
            name: 工具名称，用于唯一标识和调用
            description: 工具描述，说明工具的用途和功能
        """
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行工具的核心方法

        这是工具的核心接口，所有工具必须实现此方法。
        方法接受一个字典参数，执行相应的功能，并返回字符串结果。

        Args:
            parameters: 工具执行所需的参数字典。
                       具体需要哪些参数由get_parameters方法定义。

        Returns:
            工具执行结果的字符串表示。
                       即使执行失败，也应返回描述错误的字符串。

        Note:
            子类实现时应该包含适当的错误处理，
            确保即使执行失败也返回一个描述性字符串。
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        获取工具参数定义

        此方法用于声明工具所需的参数信息，包括参数名、类型、描述等。
        这些信息用于：
        - 生成Agent可读的工具文档
        - 验证调用者传入的参数
        - 构建OpenAI function calling schema

        Returns:
            ToolParameter列表，描述工具的所有参数

        Example:
            def get_parameters(self) -> List[ToolParameter]:
                return [
                    ToolParameter(
                        name="query",
                        type="string",
                        description="搜索查询词"
                    ),
                    ToolParameter(
                        name="limit",
                        type="number",
                        description="返回结果数量限制",
                        required=False,
                        default=5
                    )
                ]
        """
        pass

    def __str__(self) -> str:
        """返回工具的可读表示"""
        return f"Tool(name={self.name}, description={self.description})"

    def __repr__(self) -> str:
        """返回工具的技术表示"""
        return f"Tool(name='{self.name}', description='{self.description}')"