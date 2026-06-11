"""
HelloAgentsLLM - 统一的大语言模型接口
======================================

本模块是框架的核心组件，提供了与多种LLM服务商通信的统一接口。

主要功能:
1. 多提供商支持：支持OpenAI、ModelScope、智谱AI、VLLM、Ollama等
2. 自动检测机制：根据环境变量自动识别LLM服务商
3. 流式响应支持：支持逐字流式输出，提升用户体验
4. 对话历史管理：内置消息历史管理功能

核心设计:
- 对外统一接口：对不同服务商使用相同的调用方式
- 自动适配：通过自动检测机制简化配置流程
- 向后兼容：保留与OpenAI API兼容的接口

支持的提供商:
- openai: OpenAI官方API（GPT-3.5, GPT-4等）
- modelscope: 魔搭社区ModelScope平台
- zhipu: 智谱AI（GLM模型）
- vllm: 本地VLLM推理服务
- ollama: 本地Ollama推理服务
- local: 通用本地部署方案

使用示例:
    from hello_agents import HelloAgentsLLM

    # 自动检测（推荐）
    llm = HelloAgentsLLM()

    # 手动指定提供商
    llm = HelloAgentsLLM(provider="modelscope")

    # 调用示例
    messages = [{"role": "user", "content": "你好"}]
    for chunk in llm.think(messages):
        print(chunk, end="", flush=True)
"""

import os
import re
from typing import Optional, Iterator, List, Dict, Any, Union
from openai import OpenAI

from .exceptions import LLMPricingError


class HelloAgentsLLM:
    """
    HelloAgents框架的LLM统一接口

    此类封装了与不同LLM服务商通信的所有逻辑，为上层Agent提供统一的调用接口。
    支持自动检测、手动指定、流式响应等功能。

    Attributes:
        provider: 当前使用的LLM服务商名称
        model: 使用的模型名称
        temperature: 采样温度参数
        max_tokens: 最大生成token数
        timeout: 请求超时时间（秒）
        api_key: API密钥（自动从环境变量读取或手动设置）
        base_url: API基础URL

    Example:
        >>> llm = HelloAgentsLLM(provider="openai")
        >>> messages = [{"role": "user", "content": "解释量子计算"}]
        >>> response = llm.invoke(messages)
        >>> print(response)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs
    ):
        """
        初始化HelloAgentsLLM客户端

        Args:
            model: 模型名称，如 "gpt-3.5-turbo"、"Qwen/Qwen2.5-VL-72B-Instruct"
                  如果为None，将使用环境变量LLM_MODEL_ID或默认值
            api_key: API密钥，如果为None将自动从环境变量读取
            base_url: API基础URL，如果为None将根据provider自动设置
            provider: LLM服务商，"auto"表示自动检测，可选值包括:
                     openai, modelscope, zhipu, vllm, ollama, local
            temperature: 采样温度，控制生成随机性（0-2，越高越随机）
            max_tokens: 最大生成token数，None表示不限制
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数，将传递给底层HTTP客户端
        """
        # 首先自动检测provider（需要读取环境变量）
        self.provider = self._auto_detect_provider(api_key, base_url, provider)

        # 根据provider解析credentials
        resolved_api_key, resolved_base_url = self._resolve_credentials(
            api_key, base_url, self.provider
        )

        # 设置实例属性
        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.model = model or os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_NAME") or self._get_default_model(self.provider)
        self.temperature = kwargs.get('temperature', temperature)
        self.max_tokens = max_tokens or kwargs.get('max_tokens')
        self.timeout = timeout

        # 创建OpenAI客户端（兼容所有OpenAI格式的API）
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            **kwargs
        )

        # 打印初始化信息
        print(f"✅ HelloAgentsLLM 初始化完成")
        print(f"   - Provider: {self.provider}")
        print(f"   - Model: {self.model}")
        print(f"   - Base URL: {self.base_url}")

    def _auto_detect_provider(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        explicit_provider: Optional[str]
    ) -> str:
        """
        自动检测LLM提供商

        检测优先级（从高到低）：
        1. 显式指定的provider参数
        2. 特定服务商的环境变量（如OPENAI_API_KEY、MODELSCOPE_API_KEY等）
        3. 根据base_url判断（域名匹配、端口匹配）
        4. 根据API密钥格式辅助判断

        Args:
            api_key: 用户提供的API密钥
            base_url: 用户提供的base_url
            explicit_provider: 显式指定的provider

        Returns:
            检测到的provider字符串
        """
        # 如果显式指定了provider，直接使用
        if explicit_provider and explicit_provider != "auto":
            print(f"🔍 使用显式指定的provider: {explicit_provider}")
            return explicit_provider

        # 获取实际使用的环境变量值
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 1. 检查特定提供商的环境变量（最高优先级）
        if os.getenv("MODELSCOPE_API_KEY"):
            print("🔍 自动检测到: ModelScope")
            return "modelscope"
        if os.getenv("OPENAI_API_KEY"):
            print("🔍 自动检测到: OpenAI")
            return "openai"
        if os.getenv("ZHIPU_API_KEY"):
            print("🔍 自动检测到: 智谱AI")
            return "zhipu"

        # 2. 根据 base_url 判断
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api-inference.modelscope.cn" in base_url_lower:
                print("🔍 自动检测到: ModelScope (通过URL)")
                return "modelscope"
            if "open.bigmodel.cn" in base_url_lower:
                print("🔍 自动检测到: 智谱AI (通过URL)")
                return "zhipu"
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower:
                    print("🔍 自动检测到: Ollama (通过端口)")
                    return "ollama"
                if ":8000" in base_url_lower:
                    print("🔍 自动检测到: VLLM (通过端口)")
                    return "vllm"
                print("🔍 自动检测到: local (本地服务)")
                return "local"

        # 3. 根据 API 密钥格式辅助判断
        if actual_api_key:
            if actual_api_key.startswith("ms-"):
                print("🔍 自动检测到: ModelScope (通过密钥格式)")
                return "modelscope"

        # 4. 默认返回 'openai'
        print("🔍 未检测到特定provider，使用默认: openai")
        return "openai"

    def _resolve_credentials(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        provider: str
    ) -> tuple:
        """
        根据provider解析API密钥和base_url

        每个provider都有其默认的base_url和环境变量名，
        此方法负责根据provider自动设置这些值。

        Args:
            api_key: 用户提供的API密钥
            base_url: 用户提供的base_url
            provider: 已确定的provider

        Returns:
            (resolved_api_key, resolved_base_url) 元组
        """
        if provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL") or "https://api.openai.com/v1"
            if not resolved_api_key:
                raise LLMPricingError("未找到OpenAI API密钥，请设置OPENAI_API_KEY或LLM_API_KEY环境变量", provider="openai")
            return resolved_api_key, resolved_base_url

        elif provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            if not resolved_api_key:
                raise LLMPricingError("未找到ModelScope API密钥，请设置MODELSCOPE_API_KEY环境变量", provider="modelscope")
            return resolved_api_key, resolved_base_url

        elif provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"
            if not resolved_api_key:
                raise LLMPricingError("未找到智谱AI API密钥，请设置ZHIPU_API_KEY环境变量", provider="zhipu")
            return resolved_api_key, resolved_base_url

        elif provider in ["vllm", "ollama", "local"]:
            # 本地服务通常不需要真实API Key
            resolved_api_key = api_key or "local"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or (
                "http://localhost:8000/v1" if provider == "vllm" else
                "http://localhost:11434/v1"
            )
            return resolved_api_key, resolved_base_url

        else:
            # 默认为openai
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

    def _get_default_model(self, provider: str) -> str:
        """
        获取指定provider的默认模型

        Args:
            provider: LLM提供商

        Returns:
            默认模型名称
        """
        defaults = {
            "openai": "gpt-3.5-turbo",
            "modelscope": "Qwen/Qwen2.5-VL-72B-Instruct",
            "zhipu": "glm-4",
            "vllm": "Qwen/Qwen1.5-0.5B-Chat",
            "ollama": "llama3",
            "local": "llama3",
        }
        return defaults.get(provider, "gpt-3.5-turbo")

    def invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用LLM生成完整响应（非流式）

        这是最常用的调用方法，用于获取完整的响应内容。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数，将传递给API调用

        Returns:
            生成的完整响应文本字符串

        Raises:
            LLMPricingError: 当API调用失败时抛出

        Example:
            >>> messages = [{"role": "user", "content": "解释什么是量子纠缠"}]
            >>> response = llm.invoke(messages)
            >>> print(response)
        """
        try:
            # 构建API调用参数
            call_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get('temperature', self.temperature),
            }
            if self.max_tokens is not None:
                call_kwargs["max_tokens"] = self.max_tokens

            # 调用API
            response = self._client.chat.completions.create(**call_kwargs)

            # 提取响应内容
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content or ""

            return ""

        except Exception as e:
            raise LLMPricingError(f"LLM调用失败: {str(e)}", provider=self.provider)

    def think(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[str]:
        """
        调用LLM并以流式方式返回响应

        此方法逐字返回响应内容，适用于需要实时显示生成过程的场景。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数，将传递给API调用

        Yields:
            生成的文本片段（通常为单个字符或单词）

        Example:
            >>> messages = [{"role": "user", "content": "写一首关于春天的诗"}]
            >>> for chunk in llm.think(messages):
            ...     print(chunk, end="", flush=True)
        """
        # 构建API调用参数
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": True,
        }
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens

        # 调用API
        response = self._client.chat.completions.create(**call_kwargs)

        # 流式处理响应
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[str]:
        """
        流式调用LLM（stream_invoke的别名方法）

        为了保持与SimpleAgent的兼容性提供此别名方法。

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        return self.think(messages, **kwargs)

    def __str__(self) -> str:
        """返回LLM客户端的可读表示"""
        return f"HelloAgentsLLM(provider={self.provider}, model={self.model})"

    def __repr__(self) -> str:
        """返回LLM客户端的技术表示"""
        return f"HelloAgentsLLM(provider='{self.provider}', model='{self.model}', base_url='{self.base_url}')"