"""
ReActAgent - 推理与行动结合的Agent
====================================

ReAct（Reasoning + Acting）是一种将推理和行动结合的Agent范式。
通过思考-行动-观察的循环，让Agent能够解决复杂问题。

核心思想:
1. Thought（思考）：分析当前问题，决定下一步行动
2. Action（行动）：执行工具或给出最终答案
3. Observation（观察）：获取工具执行结果，用于下一步推理

ReAct循环:
    while not finished:
        thought: 分析问题
        action: 选择工具或Finish
        observation: 获取结果
        更新上下文

使用示例:
    from hello_agents import HelloAgentsLLM
    from hello_agents.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(my_tool)

    agent = ReActAgent(name="研究者", llm=llm, tool_registry=registry)
    result = agent.run("帮我查一下今天的天气")
"""

from typing import Optional, List, Dict, Tuple

from ..core.agent import Agent
from ..core.message import Message
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..tools.registry import ToolRegistry


# 默认的ReAct提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""


class ReActAgent(Agent):
    """
    ReAct Agent - 推理与行动结合的智能体

    ReAct范式通过显式的思考过程和工具调用，使Agent能够处理复杂的多步骤问题。
    每个步骤包括：思考（分析问题）→ 行动（调用工具或完成任务）→ 观察（获取结果）

    继承自:
        Agent: 抽象基类

    额外属性:
        tool_registry: 工具注册表，管理Agent可用的工具
        max_steps: ReAct循环的最大执行步数，防止无限循环
        current_history: 当前任务的执行历史
        prompt_template: 可自定义的提示词模板

    Example:
        >>> agent = ReActAgent(name="研究者", llm=llm, tool_registry=registry)
        >>> result = agent.run("计算 15 * 8 + 32")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: HelloAgentsLLM实例
            tool_registry: 工具注册表，用于管理和执行工具
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            max_steps: 最大执行步数，默认为5，防止无限循环
            custom_prompt: 自定义的提示词模板，用于替换默认模板
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent处理问题

        ReAct的执行流程：
        1. 初始化执行历史
        2. 进入思考-行动循环：
           a. 构建提示词（包含工具描述、问题、历史）
           b. 调用LLM获取思考和行动
           c. 解析输出，判断是否完成
           d. 如果是工具调用，执行工具并记录结果
        3. 达到最大步数或得到最终答案时结束

        Args:
            input_text: 用户的问题或任务描述
            **kwargs: 其他参数，传递给LLM调用

        Returns:
            Agent生成的最终答案
        """
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history) if self.current_history else "（暂无历史）"
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            # 3. 解析输出
            thought, action = self._parse_output(response_text)
            print(f"💭 思考: {thought}")
            print(f"🎬 行动: {action}")

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self._save_to_history(input_text, final_answer)
                print(f"✅ 问题解决，最终答案: {final_answer[:50]}...")
                return final_answer

            # 5. 执行工具调用
            if action:
                tool_name, tool_input = self._parse_action(action)
                print(f"🔧 执行工具: {tool_name}，输入: {tool_input}")

                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")
                print(f"📍 观察结果: {observation[:100]}...")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self._save_to_history(input_text, final_answer)
        print(f"⚠️ 达到最大步数限制: {self.max_steps}")
        return final_answer

    def _parse_output(self, text: str) -> Tuple[str, str]:
        """
        解析LLM输出，分离思考和行动

        期望的输出格式：
        Thought: ...
        Action: ...

        Args:
            text: LLM返回的原始文本

        Returns:
            (thought, action) 元组
        """
        thought = ""
        action = ""

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[7:].strip()
            elif line.startswith("Action:"):
                action = line[6:].strip()

        return thought, action

    def _parse_action(self, action: str) -> Tuple[str, str]:
        """
        解析行动指令，提取工具名和参数

        行动格式：
        - tool_name[参数] - 调用工具
        - Finish[最终答案] - 完成任务

        Args:
            action: 行动字符串

        Returns:
            (tool_name, tool_input) 元组，如果行动是Finish则返回(None, None)
        """
        # 处理Finish
        if action.startswith("Finish["):
            return None, None

        # 解析工具调用
        # 格式: tool_name[参数]
        if "[" in action and action.endswith("]"):
            tool_name = action[:action.index("[")]
            tool_input = action[action.index("[") + 1:-1]
            return tool_name.strip(), tool_input.strip()

        return action, ""

    def _parse_action_input(self, action: str) -> str:
        """
        从Finish行动中提取最终答案

        Args:
            action: Finish行动字符串，格式: Finish[答案]

        Returns:
            答案文本
        """
        if action.startswith("Finish[") and action.endswith("]"):
            return action[7:-1].strip()
        return action

    def _save_to_history(self, input_text: str, response: str) -> None:
        """保存对话到历史记录"""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))