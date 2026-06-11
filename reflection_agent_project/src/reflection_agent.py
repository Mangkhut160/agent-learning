"""
Reflection 智能体模块
实现"执行-反思-优化"迭代范式的智能体
"""

from .memory import Memory


# ============================================================
# 提示词模板
# ============================================================

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答"无需改进"。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

# 评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:
    """
    Reflection 智能体。

    通过"执行 → 反思 → 优化"的迭代循环来提升代码质量：
    1. 初始执行：根据任务生成初版代码
    2. 反思：以评审专家身份批判性分析代码，找出性能瓶颈
    3. 优化：根据反馈改进代码
    4. 循环步骤2-3，直到评审认为"无需改进"或达到最大迭代次数
    """

    def __init__(self, llm_client, max_iterations: int = 3):
        """
        初始化 Reflection 智能体。

        Args:
            llm_client: LLM客户端实例（需提供think方法）
            max_iterations: 最大迭代次数，防止无限循环
        """
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        """
        运行 Reflection 智能体的完整流程。

        Args:
            task: 编程任务描述

        Returns:
            最终优化后的代码字符串
        """
        print(f"\n{'='*60}")
        print(f"🤖 Reflection 智能体启动")
        print(f"📝 任务: {task}")
        print(f"🔄 最大迭代次数: {self.max_iterations}")
        print(f"{'='*60}")

        # --- 1. 初始执行 ---
        print("\n--- 阶段1: 初始执行 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环: 反思 → 优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 阶段2: 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task=task, code=last_code
            )
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查终止条件
            if "无需改进" in feedback:
                print("\n✅ 评审认为代码已无需改进，优化完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        # --- 3. 输出最终结果 ---
        final_code = self.memory.get_last_execution()
        print(f"\n{'='*60}")
        print(f"🎉 任务完成")
        print(f"📋 记忆共存储 {len(self.memory.records)} 条记录")
        print(f"📋 最终代码:\n```python\n{final_code}\n```")
        print(f"{'='*60}")

        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """调用LLM并获取响应文本。"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text

    def get_trajectory(self) -> str:
        """获取完整的执行-反思轨迹。"""
        return self.memory.get_trajectory()
