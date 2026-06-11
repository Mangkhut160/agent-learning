"""
PlanAndSolveAgent - 规划与执行Agent
====================================

PlanAndSolveAgent将复杂问题分解为多个简单步骤，然后逐步执行。
Planner负责将问题分解为计划，Executor负责按计划执行每一步。

核心思想:
1. Plan（规划）：将复杂问题分解为有序的步骤列表
2. Execute（执行）：按顺序执行每个步骤，记录结果
3. 完成：所有步骤执行完毕后，给出最终答案

适用场景:
- 多步骤复杂问题
- 需要逻辑推理的任务
- 数学问题、代码调试等

与ReAct的区别:
- ReAct：边推理边行动，思考和行动交织
- PlanAndSolve：先规划后执行，计划和执行分离

使用示例:
    from hello_agents import HelloAgentsLLM

    agent = PlanAndSolveAgent(name="规划助手", llm=llm)

    # 复杂问题
    result = agent.run("一个水果店周一卖出了15个苹果，周二卖出了周一的2倍，"
                       "周三比周二少5个，请问三天共卖出多少苹果？")
"""

from typing import Optional, Dict, List

from ..core.agent import Agent
from ..core.message import Message
from ..core.llm import HelloAgentsLLM
from ..core.config import Config


# 默认规划器提示词
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词
DEFAULT_EXECUTOR_PROMPT = """
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


class PlanAndSolveAgent(Agent):
    """
    规划与执行Agent - 先规划后执行的智能体

    PlanAndSolveAgent将复杂任务分解为"规划"和"执行"两个阶段。
    这种分离使得Agent能够更好地处理多步骤问题，因为规划阶段
    可以看到问题的全貌，而执行阶段只需专注于当前步骤。

    继承自:
        Agent: 抽象基类

    核心属性:
        planner_prompt: 规划器提示词模板
        executor_prompt: 执行器提示词模板
        max_steps: 最大执行步数

    工作流程:
        1. 规划阶段：使用planner_prompt生成步骤计划
        2. 执行循环：
           a. 取出当前步骤
           b. 使用executor_prompt执行该步骤
           c. 记录结果到历史
           d. 重复直到所有步骤完成
        3. 返回最终答案

    Example:
        >>> agent = PlanAndSolveAgent(name="规划助手", llm=llm)
        >>> result = agent.run("计算 10 + 20 * 2")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 10,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化PlanAndSolveAgent

        Args:
            name: Agent名称
            llm: HelloAgentsLLM实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            max_steps: 最大执行步数，默认为10
            custom_prompts: 自定义提示词，包含planner和executor键
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_steps = max_steps

        # 设置提示词
        if custom_prompts:
            self.planner_prompt = custom_prompts.get("planner", DEFAULT_PLANNER_PROMPT)
            self.executor_prompt = custom_prompts.get("executor", DEFAULT_EXECUTOR_PROMPT)
        else:
            self.planner_prompt = DEFAULT_PLANNER_PROMPT
            self.executor_prompt = DEFAULT_EXECUTOR_PROMPT

        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, question: str, **kwargs) -> str:
        """
        运行PlanAndSolveAgent处理问题

        工作流程：
        1. 规划阶段：调用LLM将问题分解为步骤列表
        2. 执行循环：
           - 取出当前步骤
           - 调用LLM执行该步骤
           - 记录结果
        3. 返回最终答案

        Args:
            question: 复杂问题描述
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {question}")

        # 步骤1: 生成计划
        print("📋 步骤1: 生成计划...")
        plan = self._create_plan(question, **kwargs)
        print(f"   计划步骤: {plan}")

        # 步骤2: 执行计划
        print("🎬 步骤2: 执行计划...")
        history = []  # 记录已完成步骤的结果

        for step_idx, step in enumerate(plan):
            if step_idx >= self.max_steps:
                print(f"⚠️ 达到最大步数限制: {self.max_steps}")
                break

            print(f"   执行步骤 {step_idx + 1}: {step}")
            result = self._execute_step(question, plan, history, step, **kwargs)
            history.append(f"步骤{step_idx + 1}: {result}")
            print(f"   结果: {result}")

        # 汇总结果
        final_answer = self._summarize(question, plan, history, **kwargs)

        # 保存到历史
        self.add_message(Message(question, "user"))
        self.add_message(Message(final_answer, "assistant"))

        print(f"✅ {self.name} 处理完成")
        return final_answer

    def _create_plan(self, question: str, **kwargs) -> List[str]:
        """
        生成问题解决计划

        使用planner_prompt调用LLM，将问题分解为步骤列表。

        Args:
            question: 问题描述
            **kwargs: 其他参数

        Returns:
            步骤列表
        """
        prompt = self.planner_prompt.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        response = self.llm.invoke(messages, **kwargs)

        # 解析计划（期望格式：["步骤1", "步骤2", ...]）
        plan = self._parse_plan(response)
        return plan

    def _parse_plan(self, text: str) -> List[str]:
        """
        解析LLM返回的计划文本

        期望格式：
        ```python
        ["步骤1", "步骤2", ...]
        ```
        或其他包含步骤列表的格式。

        Args:
            text: LLM返回的计划文本

        Returns:
            步骤列表
        """
        import re

        # 尝试提取Python列表格式
        # 匹配 ```python ... ``` 或 ``` ... ``` 包裹的内容
        patterns = [
            r'```python\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```',
            r'\[.*?\]',  # 简单的列表匹配
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                list_str = match.group(1) if match.lastindex else match.group(0)
                try:
                    # 尝试eval安全解析
                    import ast
                    result = ast.literal_eval(list_str)
                    if isinstance(result, list) and all(isinstance(item, str) for item in result):
                        return result
                except:
                    pass

        # 如果解析失败，尝试手动解析
        # 查找所有引号包裹的内容
        items = re.findall(r'"([^"]*)"', text)
        if items:
            return items

        # 返回空列表作为后备
        return ["理解问题", "执行计算", "汇总答案"]

    def _execute_step(
        self,
        question: str,
        plan: List[str],
        history: List[str],
        current_step: str,
        **kwargs
    ) -> str:
        """
        执行单个步骤

        使用executor_prompt调用LLM，执行当前步骤。

        Args:
            question: 原始问题
            plan: 完整计划
            history: 已完成步骤的结果
            current_step: 当前要执行的步骤
            **kwargs: 其他参数

        Returns:
            步骤执行结果
        """
        history_str = "\n".join(history) if history else "（暂无历史结果）"
        plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))

        prompt = self.executor_prompt.format(
            question=question,
            plan=plan_str,
            history=history_str,
            current_step=current_step
        )

        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs)

    def _summarize(
        self,
        question: str,
        plan: List[str],
        history: List[str],
        **kwargs
    ) -> str:
        """
        汇总所有步骤结果，生成最终答案

        Args:
            question: 原始问题
            plan: 完整计划
            history: 所有步骤的执行结果
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        history_str = "\n".join(history) if history else "无"

        summarize_prompt = f"""
基于以下步骤执行结果，请给出问题的最终答案。

# 原始问题:
{question}

# 执行步骤:
{history_str}

请直接给出最终答案，不需要额外解释。
"""

        messages = [{"role": "user", "content": summarize_prompt}]
        return self.llm.invoke(messages, **kwargs)