"""
异常体系定义
=============

HelloAgents框架定义了统一的异常体系，便于错误处理和调试。

异常层次结构:
    HelloAgentsError (基类)
    ├── LLMPricingError (LLM调用相关错误)
    ├── ToolExecutionError (工具执行错误)
    └── AgentRuntimeError (Agent运行时错误)

使用示例:
    try:
        result = registry.execute_tool("calculator", "2+2")
    except ToolExecutionError as e:
        print(f"工具执行失败: {e}")
"""

from typing import Optional


class HelloAgentsError(Exception):
    """
    HelloAgents框架的基类异常

    所有框架自定义异常都继承自此类，便于统一捕获和处理。
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        初始化异常

        Args:
            message: 错误描述信息
            details: 额外的错误详情（可选），用于传递调试信息
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """返回格式化的错误信息"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (详情: {details_str})"
        return self.message


class LLMPricingError(HelloAgentsError):
    """
    LLM调用相关错误

    当与语言模型交互时发生错误（如API调用失败、认证错误等）时抛出此异常。

    常见场景:
    - API密钥无效或缺失
    - 网络连接失败
    - 模型响应格式错误
    - 请求超时
    """

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        """
        初始化LLM错误

        Args:
            message: 错误描述
            provider: 出错的LLM提供商名称（如"openai", "modelscope"等）
            **kwargs: 其他参数传递给父类
        """
        super().__init__(message, **kwargs)
        self.provider = provider


class ToolExecutionError(HelloAgentsError):
    """
    工具执行错误

    当工具执行过程中发生错误时抛出此异常。

    常见场景:
    - 工具不存在
    - 参数格式错误
    - 工具内部逻辑错误
    - 外部服务调用失败（如搜索API、超时等）
    """

    def __init__(self, tool_name: str, message: str, **kwargs):
        """
        初始化工具执行错误

        Args:
            tool_name: 出错的工具名称
            message: 错误描述
            **kwargs: 其他参数传递给父类
        """
        super().__init__(f"工具 '{tool_name}' 执行失败: {message}", **kwargs)
        self.tool_name = tool_name


class AgentRuntimeError(HelloAgentsError):
    """
    Agent运行时错误

    当Agent在执行任务过程中发生错误时抛出此异常。

    常见场景:
    - Agent循环超过最大步数限制
    - 历史记录管理错误
    - Agent配置无效
    - 未提供必要的组件（如未配置LLM）
    """

    def __init__(self, agent_name: str, message: str, **kwargs):
        """
        初始化Agent运行时错误

        Args:
            agent_name: 出错的Agent名称
            message: 错误描述
            **kwargs: 其他参数传递给父类
        """
        super().__init__(f"Agent '{agent_name}' 运行时错误: {message}", **kwargs)
        self.agent_name = agent_name