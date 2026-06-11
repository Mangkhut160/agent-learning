"""
工具注册表
===========

ToolRegistry是HelloAgents工具系统的管理中枢，提供工具的注册、
发现、执行等功能。它支持两种注册方式：
1. Tool对象注册：适合复杂工具，支持完整的参数定义
2. 函数直接注册：适合简单工具，快速集成现有函数

核心功能:
- register_tool: 注册Tool对象
- register_function: 注册函数为工具
- execute_tool: 执行指定工具
- get_tools_description: 获取所有工具的格式化描述
- to_openai_schema: 转换为OpenAI function calling格式

使用示例:
    from hello_agents.tools import ToolRegistry

    registry = ToolRegistry()

    # 注册Tool对象
    registry.register_tool(my_tool)

    # 注册函数
    def my_calc(expr):
        return str(eval(expr))
    registry.register_function("calc", "计算器", my_calc)

    # 执行工具
    result = registry.execute_tool("calc", "2+3")
    print(result)  # 5
"""

from typing import Dict, Any, List, Callable, Optional
from .base import Tool, ToolParameter


class ToolRegistry:
    """
    HelloAgents工具注册表 - 工具系统的管理中枢

    ToolRegistry负责管理所有可用的工具，提供统一的注册、执行和发现接口。
    它维护两个内部存储：
    - _tools: Tool对象的字典，键为工具名
    - _functions: 函数的字典，键为工具名

    设计特点:
    1. 双重存储：支持Tool对象和函数两种形式
    2. 统一执行：通过execute_tool提供一致的调用接口
    3. 自描述：通过get_tools_description生成可读的工具文档

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register_tool(CalculatorTool())
        >>> result = registry.execute_tool("calculator", "2+3")
    """

    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, Tool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}
        print("🔧 工具注册表初始化完成")

    def register_tool(self, tool: Tool) -> None:
        """
        注册Tool对象

        适合复杂工具，支持完整的参数定义和验证。

        Args:
            tool: Tool子类实例

        Note:
            如果工具名已存在，会打印警告但仍允许覆盖，
            这是为了支持热更新等场景。

        Example:
            calculator = CalculatorTool()
            registry.register_tool(calculator)
        """
        if tool.name in self._tools:
            print(f"⚠️ 警告: 工具 '{tool.name}' 已存在，将被覆盖。")

        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册 (类型: Tool对象)")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[[str], str]
    ) -> None:
        """
        直接注册函数作为工具（简便方式）

        适合简单工具，可以快速将现有函数集成到工具系统中。
        注册的函数签名应为：func(input_str) -> str

        Args:
            name: 工具名称
            description: 工具描述
            func: 函数，接收字符串参数，返回字符串结果

        Example:
            def my_calc(expression):
                return str(eval(expression))

            registry.register_function(
                name="calculator",
                description="数学计算器",
                func=my_calc
            )
        """
        if name in self._functions:
            print(f"⚠️ 警告: 工具 '{name}' 已存在，将被覆盖。")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册 (类型: Function)")

    def execute_tool(self, tool_name: str, input_data: str) -> str:
        """
        执行指定工具

        这是工具系统的核心执行方法，提供统一的工具调用接口。
        首先检查_tool对象，如果不存在则检查_functions。

        Args:
            tool_name: 工具名称
            input_data: 工具输入数据（字符串）

        Returns:
            工具执行结果的字符串

        Raises:
            ValueError: 当工具不存在时抛出
        """
        # 先检查Tool对象
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            try:
                # 将字符串输入转换为字典参数
                parameters = self._parse_input(input_data, tool)
                result = tool.run(parameters)
                return str(result) if result else "执行完成，无返回结果"
            except Exception as e:
                return f"❌ 工具执行错误: {str(e)}"

        # 再检查函数
        if tool_name in self._functions:
            func_info = self._functions[tool_name]
            try:
                result = func_info["func"](input_data)
                return str(result) if result else "执行完成，无返回结果"
            except Exception as e:
                return f"❌ 函数执行错误: {str(e)}"

        # 工具不存在
        available = self.list_tools()
        return f"❌ 错误: 工具 '{tool_name}' 不存在。可用工具: {available}"

    def _parse_input(self, input_data: str, tool: Tool) -> Dict[str, Any]:
        """
        解析输入数据为参数字典

        简单的参数解析实现，支持以下格式：
        - 单参数: "value" -> {"input": "value"}
        - 多参数: "key1=value1,key2=value2" -> {"key1": "value1", "key2": "value2"}

        Args:
            input_data: 输入字符串
            tool: Tool对象，用于获取参数定义

        Returns:
            参数字典
        """
        # 获取工具的参数定义
        params = tool.get_parameters()

        # 如果没有参数定义或只有一个参数，直接返回输入
        if not params or len(params) == 1:
            param_name = params[0].name if params else "input"
            return {param_name: input_data}

        # 多参数情况：尝试解析 key=value 格式
        if '=' in input_data:
            result = {}
            pairs = input_data.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    result[key.strip()] = value.strip()
            if result:
                return result

        # 无法解析，返回默认格式
        return {"input": input_data}

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        获取指定名称的Tool对象

        Args:
            tool_name: 工具名称

        Returns:
            Tool对象，如果不存在则返回None
        """
        return self._tools.get(tool_name)

    def unregister(self, tool_name: str) -> bool:
        """
        注销工具

        从注册表中移除指定工具。

        Args:
            tool_name: 工具名称

        Returns:
            是否成功移除
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            print(f"🗑️ 工具 '{tool_name}' 已注销")
            return True
        if tool_name in self._functions:
            del self._functions[tool_name]
            print(f"🗑️ 工具 '{tool_name}' 已注销")
            return True
        return False

    def list_tools(self) -> List[str]:
        """
        列出所有可用工具

        Returns:
            工具名称列表
        """
        tools = list(self._tools.keys()) + list(self._functions.keys())
        return tools

    def get_tools_description(self) -> str:
        """
        获取所有可用工具的格式化描述字符串

        此方法生成的描述字符串可以直接用于构建Agent的提示词，
        让Agent了解可用的工具及其功能。

        Returns:
            格式化的工具描述字符串
        """
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            params = tool.get_parameters()
            params_str = ", ".join([f"{p.name}({p.type})" for p in params]) if params else "无参数"
            descriptions.append(f"- {tool.name}: {tool.description} [参数: {params_str}]")

        # 函数工具描述
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def to_openai_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        将指定工具转换为OpenAI function calling schema格式

        用于FunctionCallAgent，使工具能够被OpenAI原生function calling使用。

        Args:
            tool_name: 工具名称

        Returns:
            符合OpenAI function calling标准的schema，如果工具不存在返回None
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return None

        parameters = tool.get_parameters()

        # 构建properties
        properties = {}
        required = []

        for param in parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }

            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"

            if param.type == "array":
                prop["items"] = {"type": "string"}

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def __str__(self) -> str:
        """返回注册表的可读表示"""
        return f"ToolRegistry(tools={len(self._tools)}, functions={len(self._functions)})"