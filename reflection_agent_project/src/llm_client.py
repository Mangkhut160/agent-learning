"""
LLM 客户端封装模块
提供统一的大语言模型调用接口
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class HelloAgentsLLM:
    """
    一个封装了OpenAI兼容API的LLM客户端。
    支持任何兼容OpenAI接口的服务。
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None
    ):
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError(
                "API_KEY 未配置！请在 .env 文件中设置 API_KEY"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        print(f"[LLM客户端] 已初始化，模型: {self.model_name}")

    def think(self, messages: list, temperature: float = 0.7) -> str:
        """
        调用LLM生成响应。

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 生成温度，控制随机性

        Returns:
            LLM生成的文本响应
        """
        try:
            print(f"🧠 正在调用 {self.model_name} 模型...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            result = response.choices[0].message.content
            print(f"✅ 大语言模型响应成功")
            return self._clean_response(result)
        except Exception as e:
            print(f"[LLM错误] 调用失败: {e}")
            return None

    def _clean_response(self, response_text: str) -> str:
        """移除模型思考过程，只保留面向用户的答案。"""
        return re.sub(
            r"<think>.*?</think>",
            "",
            response_text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    def __repr__(self):
        return f"<HelloAgentsLLM model={self.model_name}>"
