"""
执行器模块
负责按计划逐步执行每个子任务，并管理状态上下文
"""

import re

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class Executor:
    """执行器：按计划顺序执行每个步骤，维护历史上下文。"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划逐步执行并解决问题。

        Args:
            question: 原始问题
            plan: 由步骤字符串组成的计划列表

        Returns:
            最终答案（最后一个步骤的执行结果）
        """
        history = ""
        response_text = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )

            messages = [{"role": "user", "content": prompt}]

            response_text = self._clean_response(self.llm_client.think(messages=messages) or "")

            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        return response_text

    def _clean_response(self, response_text: str) -> str:
        """移除模型思考过程，只保留面向用户的答案。"""
        return re.sub(
            r"<think>.*?</think>",
            "",
            response_text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
