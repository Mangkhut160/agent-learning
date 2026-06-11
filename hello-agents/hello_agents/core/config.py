"""
配置管理
=========

本模块提供了HelloAgents框架的中心化配置方案，支持从环境变量读取配置，
使框架的行为易于调整和扩展，无需修改代码即可适应不同部署环境。

核心功能:
1. 集中管理配置项，按逻辑划分为LLM配置、系统配置等
2. 提供合理的默认值，保证框架零配置下也能工作
3. 支持从环境变量读取配置，便于不同环境部署

设计原则:
- 约定优于配置：提供默认值，减少用户配置负担
- 环境变量优先：允许通过环境变量覆盖默认配置

使用示例:
    from hello_agents import Config

    # 使用默认配置
    config = Config()

    # 从环境变量创建配置
    config = Config.from_env()

    # 访问配置项
    print(config.default_model)  # 默认模型
    print(config.temperature)     # 温度参数
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """
    HelloAgents配置类 - 框架级配置管理

    配置项按逻辑划分为以下几个分组：

    LLM配置:
    - default_model: 默认使用的模型
    - default_provider: 默认的LLM提供商
    - temperature: 生成温度参数
    - max_tokens: 最大生成token数

    系统配置:
    - debug: 调试模式开关
    - log_level: 日志级别

    其他配置:
    - max_history_length: 最大历史记录长度

    Attributes:
        default_model: 默认模型名称，默认为 "gpt-3.5-turbo"
        default_provider: 默认LLM提供商，默认为 "openai"
        temperature: 采样温度，控制随机性（0-2，越高越随机）
        max_tokens: 最大生成token数，None表示不限制
        debug: 是否开启调试模式
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        max_history_length: 对话历史的最大条数限制
    """

    # LLM配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置实例

        此方法是实现"零配置"体验的关键。用户可以通过设置环境变量来
        覆盖默认配置，无需修改代码即可在不同的部署环境中运行。

        支持的环境变量:
        - DEBUG: 调试模式（true/false）
        - LOG_LEVEL: 日志级别
        - TEMPERATURE: 温度参数
        - MAX_TOKENS: 最大token数

        Returns:
            Config: 从环境变量构建的配置实例

        Example:
            # 在终端设置环境变量
            export DEBUG=true
            export LOG_LEVEL=DEBUG
            export TEMPERATURE=0.8

            # Python代码中自动读取
            config = Config.from_env()
        """
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        Returns:
            包含所有配置项的字典
        """
        return self.dict()

    def update(self, **kwargs) -> None:
        """
        更新配置项

        此方法允许在运行时动态修改配置，适用于需要根据
        用户输入或上下文调整配置的场景。

        Args:
            **kwargs: 要更新的配置项键值对

        Example:
            config = Config()
            config.update(temperature=0.9, debug=True)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)