"""
ReflectionAgent - 自我反思Agent
================================

ReflectionAgent通过迭代改进的方式提升生成质量。每次生成后，
Agent会反思结果并提出改进建议，然后基于反馈生成更好的版本。

核心思想:
1. Initial（初始生成）：根据任务生成初始回答
2. Reflect（反思）：审查初始回答，找出不足
3. Refine（改进）：根据反馈改进回答
4. 重复反思-改进循环直到满意

适用场景:
- 文本生成任务（文章、报告、创意写作）
- 代码生成和优化
- 需要高质量输出的任务

使用示例:
    from hello_agents import HelloAgentsLLM

    agent = ReflectionAgent(name="写作助手", llm=llm)

    # 使用默认通用提示词
    result = agent.run("写一篇关于人工智能的文章")

    # 使用自定义提示词
    code_prompts = {
        "initial": "你是Python专家，请编写函数:{task}",
        "reflect": "请审查代码的算法效率",
        "refine": "请根据反馈优化代码"
    }
    code_agent = ReflectionAgent(name="代码助手", llm=llm, custom_prompts=code_prompts)
"""

from typing import Optional, Dict

from ..core.agent import Agent
from ..core.message import Message
from ..core.llm import HelloAgentsLLM
from ..core.config import Config


# 默认的反思提示词模板
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


class ReflectionAgent(Agent):
    """
    自我反思Agent - 通过迭代改进提升生成质量

    ReflectionAgent采用"生成-反思-改进"的循环模式，
    每次迭代都会审视上一轮的输出，提出改进建议，然后生成更好的版本。

    继承自:
        Agent: 抽象基类

    核心属性:
        max_iterations: 最大反思迭代次数
        custom_prompts: 自定义提示词字典

    工作流程:
        1. Initial: 使用initial_prompt生成初始回答
        2. Reflect: 使用reflect_prompt审查回答
        3. Refine: 使用refine_prompt根据反馈改进
        4. 重复2-3直到达到最大迭代次数或无需改进

    Example:
        >>> agent = ReflectionAgent(name="写作助手", llm=llm)
        >>> result = agent.run("写一首关于春天的诗")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化ReflectionAgent

        Args:
            name: Agent名称
            llm: HelloAgentsLLM实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            max_iterations: 最大反思迭代次数，默认为3
            custom_prompts: 自定义提示词字典，包含三个键：
                          initial, reflect, refine
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS

        print(f"✅ {name} 初始化完成，最大迭代次数: {max_iterations}")

    def run(self, task: str, **kwargs) -> str:
        """
        运行ReflectionAgent处理任务

        反思循环：
        1. 使用initial_prompt生成初始回答
        2. 进入反思-改进循环：
           a. 使用reflect_prompt审查当前回答
           b. 如果反馈是"无需改进"，结束循环
           c. 使用refine_prompt根据反馈改进回答
        3. 返回最终回答

        Args:
            task: 用户任务描述
            **kwargs: 其他参数

        Returns:
            改进后的最终回答
        """
        print(f"\n🤖 {self.name} 开始处理任务: {task}")

        # 步骤1: 初始生成
        print("📝 步骤1: 初始生成...")
        current_content = self._generate_initial(task, **kwargs)
        print(f"   初始回答: {current_content[:50]}...")

        # 步骤2: 反思-改进循环
        for iteration in range(self.max_iterations):
            print(f"\n🔄 反思循环 {iteration + 1}/{self.max_iterations}")

            # 反思当前回答
            print("   正在反思...")
            feedback = self._reflect(task, current_content, **kwargs)
            print(f"   反馈: {feedback[:80]}...")

            # 检查是否需要改进
            if "无需改进" in feedback or not feedback.strip():
                print("✅ 反馈表示无需改进，接受当前回答")
                break

            # 根据反馈改进
            print("   正在改进...")
            current_content = self._refine(task, current_content, feedback, **kwargs)
            print(f"   改进后: {current_content[:50]}...")

        # 保存到历史记录
        self.add_message(Message(task, "user"))
        self.add_message(Message(current_content, "assistant"))

        print(f"✅ {self.name} 处理完成")
        return current_content

    def _generate_initial(self, task: str, **kwargs) -> str:
        """
        生成初始回答

        Args:
            task: 任务描述
            **kwargs: 其他参数

        Returns:
            初始生成的回答
        """
        prompt = self.prompts["initial"].format(task=task)
        messages = [{"role": "user", "content": prompt}]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return self.llm.invoke(messages, **kwargs)

    def _reflect(self, task: str, content: str, **kwargs) -> str:
        """
        反思当前回答

        Args:
            task: 原始任务
            content: 当前回答
            **kwargs: 其他参数

        Returns:
            反思反馈
        """
        prompt = self.prompts["reflect"].format(task=task, content=content)
        messages = [{"role": "user", "content": prompt}]

        return self.llm.invoke(messages, **kwargs)

    def _refine(self, task: str, last_attempt: str, feedback: str, **kwargs) -> str:
        """
        根据反馈改进回答

        Args:
            task: 原始任务
            last_attempt: 上一轮回答
            feedback: 反思反馈
            **kwargs: 其他参数

        Returns:
            改进后的回答
        """
        prompt = self.prompts["refine"].format(
            task=task,
            last_attempt=last_attempt,
            feedback=feedback
        )
        messages = [{"role": "user", "content": prompt}]

        return self.llm.invoke(messages, **kwargs)