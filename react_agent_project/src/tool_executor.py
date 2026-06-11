"""
工具执行器模块
负责管理和调度所有可用工具
"""

from typing import Dict, Any, Callable


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    支持动态注册、获取和列出工具。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable) -> None:
        """
        向工具箱中注册一个新工具。

        Args:
            name: 工具名称（唯一标识符）
            description: 工具描述（供LLM理解何时使用）
            func: 工具执行函数
        """
        if name in self.tools:
            print(f"警告: 工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册。")

    def unregisterTool(self, name: str) -> bool:
        """
        移除一个已注册的工具。

        Args:
            name: 要移除的工具名称

        Returns:
            是否成功移除
        """
        if name in self.tools:
            del self.tools[name]
            print(f"🗑️ 工具 '{name}' 已移除。")
            return True
        print(f"警告: 工具 '{name}' 不存在。")
        return False

    def getTool(self, name: str) -> Callable:
        """
        根据名称获取一个工具的执行函数。

        Args:
            name: 工具名称

        Returns:
            工具执行函数，如果不存在返回None
        """
        return self.tools.get(name, {}).get("func")

    def getToolDescription(self, name: str) -> str:
        """
        根据名称获取工具的描述。

        Args:
            name: 工具名称

        Returns:
            工具描述字符串
        """
        return self.tools.get(name, {}).get("description", "")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。

        Returns:
            格式化的工具列表字符串
        """
        if not self.tools:
            return "当前没有可用工具。"

        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

    def listTools(self) -> list:
        """
        获取所有已注册工具的名称列表。

        Returns:
            工具名称列表
        """
        return list(self.tools.keys())

    def execute(self, name: str, *args, **kwargs) -> Any:
        """
        直接执行指定工具。

        Args:
            name: 工具名称
            *args, **kwargs: 传递给工具的参数

        Returns:
            工具执行结果
        """
        tool_func = self.getTool(name)
        if tool_func is None:
            return f"错误: 未找到名为 '{name}' 的工具。"
        return tool_func(*args, **kwargs)

    def __repr__(self):
        return f"<ToolExecutor tools={self.listTools()}>"


if __name__ == "__main__":
    # 测试工具执行器
    executor = ToolExecutor()

    def dummy_tool(x):
        return f"处理了: {x}"

    executor.registerTool("Dummy", "一个示例工具", dummy_tool)
    print("\n可用工具:")
    print(executor.getAvailableTools())
    print("\n执行结果:")
    print(executor.execute("Dummy", "测试输入"))
