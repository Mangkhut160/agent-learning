"""
Agent基类
==========

本模块定义了HelloAgents框架中所有Agent的抽象基类，为上层具体Agent实现
提供了统一的接口规范和通用功能。

核心设计:
- 抽象基类：所有具体Agent必须继承此类并实现run方法
- 通用功能：历史记录管理、消息添加、清空等
- 统一接口：所有Agent都遵循相同的初始化参数和方法签名

框架结构:
    Agent (抽象基类)
    ├── SimpleAgent - 基础对话Agent
    ├── ReActAgent - 推理与行动结合的Agent
    ├── ReflectionAgent - 自我反思Agent
    └── PlanAndSolveAgent - 规划与执行Agent

使用示例:
    from hello_agents import Agent, HelloAgentsLLM, Config

    # 继承Agent创建自定义Agent
    class MyAgent(Agent):
        def run(self, input_text: str, **kwargs) -> str:
            # 实现具体的Agent逻辑
            messages = [{"role": "user", "content": input_text}]
            return self.llm.invoke(messages)

    # 使用Agent
    llm = HelloAgentsLLM()
    agent = MyAgent(name="我的助手", llm=llm)
    response = agent.run("你好")
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List

from .message import Message
from .llm import HelloAgentsLLM
from .config import Config


class Agent(ABC):
    """
    Agent抽象基类 - 定义所有智能体的通用行为和属性

    此类是HelloAgents框架的顶层抽象，定义了智能体应该具备的核心接口。
    通过继承ABC（Abstract Base Classes）模块实现抽象类，
    强制所有具体智能体实现必须遵循的接口规范。

    核心属性:
        name: Agent的名称，用于标识和日志
        llm: LLM客户端实例，负责与语言模型通信
        system_prompt: 系统提示词，设定Agent的角色和行为准则
        config: 配置对象，传递框架级设置
        _history: 对话历史记录列表

    核心方法:
        run: 抽象方法，所有Agent必须实现此方法
        add_message: 添加消息到历史记录
        clear_history: 清空历史记录
        get_history: 获取历史记录的副本

    设计原则:
    1. 抽象约束：使用@abstractmethod强制子类实现run方法
    2. 组合优于继承：通过组合llm、config等组件实现功能
    3. 统一接口：所有Agent遵循相同的初始化参数和方法签名

    Example:
        >>> class MyAgent(Agent):
        ...     def run(self, input_text: str, **kwargs) -> str:
        ...         messages = [{"role": "user", "content": input_text}]
        ...         return self.llm.invoke(messages)
        >>>
        >>> agent = MyAgent(name="助手", llm=llm)
        >>> response = agent.run("你好")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        初始化Agent实例

        Args:
            name: Agent的名称，用于标识、日志输出和历史记录管理
            llm: HelloAgentsLLM的实例，负责与大语言模型通信
            system_prompt: 系统提示词，用于设定Agent的角色和行为准则。
                          如果为None，将使用框架的默认提示词。
            config: 配置对象，用于传递框架级的设置。如果为None，
                  将使用默认Config实例。
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []

        # 打印初始化信息
        print(f"🤖 Agent '{name}' 初始化完成")

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Agent处理输入

        这是最核心的抽象方法，所有具体Agent实现必须实现此方法。
        此方法定义了Agent的主要工作流程：接收输入、处理、返回输出。

        Args:
            input_text: 用户的输入文本
            **kwargs: 其他参数，子类可以定义自己的参数

        Returns:
            Agent生成的响应文本字符串

        Note:
            此方法是抽象方法，子类必须实现，否则会抛出TypeError
        """
        pass

    def add_message(self, message: Message) -> None:
        """
        添加消息到历史记录

        此方法用于记录Agent与用户的对话历史，支持后续的上下文管理
        和对话续写功能。

        Args:
            message: Message实例，包含消息内容和角色信息
        """
        self._history.append(message)

    def clear_history(self) -> None:
        """
        清空历史记录

        在需要开始新的对话主题时调用此方法。注意：此操作不可逆。
        """
        self._history.clear()
        print(f"🗑️ Agent '{self.name}' 历史记录已清空")

    def get_history(self) -> List[Message]:
        """
        获取历史记录的副本

        返回历史记录的副本而非引用，保证历史记录的安全性。

        Returns:
            Message对象列表，是历史记录的浅拷贝
        """
        return self._history.copy()

    def __str__(self) -> str:
        """返回Agent的可读表示"""
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        """返回Agent的技术表示"""
        return f"Agent(name='{self.name}', llm={self.llm}, system_prompt='{self.system_prompt[:30]}...' if self.system_prompt else None)"