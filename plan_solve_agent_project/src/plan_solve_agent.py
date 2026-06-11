"""
Plan-and-Solve 智能体模块
实现"先规划，后执行"范式的智能体
"""

from .planner import Planner
from .executor import Executor


class PlanAndSolveAgent:
    """
    Plan-and-Solve 智能体。

    将复杂任务拆分为两个阶段：
    1. 规划阶段（Planner）：将问题分解为清晰的子步骤
    2. 执行阶段（Executor）：按顺序执行每个步骤并得出最终答案

    与 ReAct 范式不同，Plan-and-Solve 在行动之前先制定完整计划，
    更适合结构性强、可以被清晰分解的复杂任务。
    """

    def __init__(self, llm_client):
        """
        初始化智能体，同时创建规划器和执行器实例。

        Args:
            llm_client: LLM客户端实例（需提供think方法）
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str) -> str:
        """
        运行智能体的完整流程：先规划，后执行。

        Args:
            question: 用户的问题

        Returns:
            最终答案字符串，如果计划生成失败则返回None
        """
        print(f"\n{'='*60}")
        print(f"🤖 Plan-and-Solve 智能体启动")
        print(f"📝 问题: {question}")
        print(f"{'='*60}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划。")
            return None

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n{'='*60}")
        print(f"🎉 任务完成")
        print(f"📋 最终答案: {final_answer}")
        print(f"{'='*60}")

        return final_answer
