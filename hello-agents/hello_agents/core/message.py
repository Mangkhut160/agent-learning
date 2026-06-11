"""
消息系统
========

本模块定义了HelloAgents框架中的统一消息格式，用于规范Agent与LLM之间的信息传递。

核心类:
- MessageRole: 消息角色类型定义（user, assistant, system, tool）
- Message: 消息类，包含内容、角色、时间戳和元数据

设计原则:
1. 对内丰富：内部使用Message对象，包含完整的时间戳和元数据
2. 对外兼容：通过to_dict()方法转换为OpenAI API兼容的字典格式

使用示例:
    from hello_agents import Message, MessageRole

    # 创建用户消息
    user_msg = Message(content="你好，请介绍一下自己", role="user")

    # 转换为API格式
    api_dict = user_msg.to_dict()
    # {'role': 'user', 'content': '你好，请介绍一下自己'}

    # 打印消息
    print(user_msg)
    # [user] 你好，请介绍一下自己
"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel


# 定义消息角色的类型，限制其取值范围为四种
# 这直接对应OpenAI API的规范，保证了类型安全
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """
    消息类 - 框架内部使用的统一消息格式

    消息是Agent与LLM之间传递信息的基本单元。每个消息包含:
    - content: 消息的实际内容
    - role: 发送者的角色（user/assistant/system/tool）
    - timestamp: 消息创建时间（自动生成）
    - metadata: 额外的元数据（可选，用于扩展）

    Attributes:
        content: 消息的实际内容文本
        role: 消息发送者的角色，决定了消息在对话中的语义
        timestamp: 消息创建时间，用于日志记录和上下文管理
        metadata: 扩展字段，可存储工具调用结果、令牌统计等信息
    """

    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        """
        初始化消息实例

        Args:
            content: 消息内容，不能为空
            role: 消息角色，必须是 MessageRole 类型定义的值之一
            timestamp: 创建时间，默认为当前时间
            metadata: 额外元数据，默认为空字典
        """
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将消息转换为字典格式（OpenAI API格式）

        这是核心功能方法，负责将内部使用的Message对象转换为
        与OpenAI API兼容的字典格式，体现了"对内丰富，对外兼容"的设计原则。

        Returns:
            符合OpenAI API规范的字典，包含 role 和 content 字段

        Example:
            >>> msg = Message("你好", role="user")
            >>> msg.to_dict()
            {'role': 'user', 'content': '你好'}
        """
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        """
        返回消息的可读字符串表示

        Returns:
            格式化的字符串，包含角色标签和内容

        Example:
            >>> msg = Message("你好", role="user")
            >>> print(msg)
            [user] 你好
        """
        return f"[{self.role}] {self.content}"

    def __repr__(self) -> str:
        """返回消息的技术表示，用于调试"""
        return f"Message(content='{self.content[:50]}...' if len(self.content) > 50 else '{self.content}', role='{self.role}')"