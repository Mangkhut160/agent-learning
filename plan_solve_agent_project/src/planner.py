"""
规划器模块
负责将复杂问题分解为结构化的行动计划
"""

import ast
import re

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


class Planner:
    """规划器：接收原始问题，生成分步骤的行动计划。"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。

        Args:
            question: 用户提出的复杂问题

        Returns:
            由步骤字符串组成的列表，解析失败时返回空列表
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""

        print(f"✅ 计划已生成:\n{response_text}")

        plan = self._parse_plan(response_text)
        if not plan:
            print(f"❌ 解析计划失败，原始响应: {response_text}")
        return plan

    def _parse_plan(self, response_text: str) -> list[str]:
        """从模型响应中提取 Python 列表格式的计划。"""
        code_blocks = re.findall(
            r"```(?:python)?\s*(.*?)```",
            response_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        candidates = list(reversed(code_blocks))
        if not candidates:
            start = response_text.find("[")
            end = response_text.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidates.append(response_text[start:end + 1])

        for candidate in candidates:
            try:
                plan = ast.literal_eval(candidate.strip())
            except (ValueError, SyntaxError):
                continue

            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                return plan

        return []
