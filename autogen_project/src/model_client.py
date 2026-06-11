"""
模型客户端模块
负责创建和配置 LLM 客户端
"""

import os
from dotenv import load_dotenv

load_dotenv()


def create_model_client():
    """
    创建并配置 OpenAI 兼容的模型客户端。

    从环境变量读取 API 配置：
    - LLM_API_KEY: API 密钥
    - LLM_BASE_URL: API 基础地址
    - LLM_MODEL_ID: 模型名称

    Returns:
        OpenAIChatCompletionClient 实例
    """
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model_id = os.getenv("LLM_MODEL_ID", "gpt-4o")

    if not api_key:
        raise ValueError("LLM_API_KEY 未配置！请在 .env 文件中设置。")

    # MiniMax 模型不支持 function calling，声明其能力
    # include_name_in_message=False 避免 MiniMax 的 "user name must be consistent" 错误
    client = OpenAIChatCompletionClient(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
        include_name_in_message=False,
    )

    print(f"[模型客户端] 已初始化，模型: {model_id}，地址: {base_url}")
    return client