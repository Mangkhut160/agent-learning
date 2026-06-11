"""
ReAct 智能体模块
实现基于 Reasoning + Acting 范式的智能体
"""

import re
from typing import Optional


# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
你是一个智能助手。请严格按照以下格式输出，否则无法被系统识别。

## 可用工具
{tools}

## 输出格式（必须严格遵守）
Thought: 你对问题的分析
Action: 你要执行的行动

## 可用Action格式
1. Search[查询内容] - 需要搜索信息时使用
2. Calculator[表达式] - 需要计算数学题时使用
3. Finish[你的答案] - 当你知道答案时使用

## 示例
Thought: 我知道这个问题，需要搜索最新信息
Action: Search[华为最新手机]

或者：

Thought: 我已经有足够信息可以回答问题了
Action: Finish[答案是xxx]

现在开始：
Question: {question}
History: {history}
"""


class ReActAgent:
    """
    ReAct智能体的核心实现类。

    通过交替进行"思考"和"行动"，逐步解决问题。
    """

    def __init__(
        self,
        llm_client,
        tool_executor,
        max_steps: int = 5,
        verbose: bool = True
    ):
        """
        初始化ReAct智能体。

        Args:
            llm_client: LLM客户端实例（需提供think方法）
            tool_executor: 工具执行器实例
            max_steps: 最大迭代步数，防止无限循环
            verbose: 是否打印详细过程
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = []
        self.full_trace = []  # 保存完整的执行轨迹

    def run(self, question: str) -> Optional[str]:
        """
        运行ReAct智能体来回答一个问题。

        Args:
            question: 用户的问题

        Returns:
            最终答案字符串，如果未能在最大步数内完成则返回None
        """
        self.history = []
        self.full_trace = []
        current_step = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 ReAct智能体启动")
            print(f"📝 问题: {question}")
            print(f"{'='*60}\n")

        while current_step < self.max_steps:
            current_step += 1
            step_info = {"step": current_step}

            if self.verbose:
                print(f"\n--- 🔄 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history) if self.history else "无"
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            step_info["llm_response"] = response_text

            if not response_text:
                if self.verbose:
                    print("❌ 错误: LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            step_info["thought"] = thought
            step_info["action_raw"] = action

            if thought and self.verbose:
                print(f"💭 思考: {thought}")

            if not action:
                if self.verbose:
                    print("⚠️ 警告: 未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                final_answer_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"🎉 最终答案: {final_answer}")
                        print(f"{'='*60}\n")
                    step_info["final_answer"] = final_answer
                    self.full_trace.append(step_info)
                    return final_answer
                else:
                    if self.verbose:
                        print("⚠️ Finish格式错误，无法提取答案")
                    break

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                if self.verbose:
                    print(f"⚠️ 无法解析Action: {action}")
                observation = f"错误: Action格式不正确，请使用 '工具名[输入]' 格式。"
            else:
                if self.verbose:
                    print(f"🎬 行动: {tool_name}[{tool_input}]")

                tool_function = self.tool_executor.getTool(tool_name)
                if not tool_function:
                    observation = f"❌ 错误: 未找到名为 '{tool_name}' 的工具。可用工具: {self.tool_executor.listTools()}"
                else:
                    observation = tool_function(tool_input)

            step_info["observation"] = observation

            if self.verbose:
                print(f"👀 观察: {observation}")

            # 5. 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

            self.full_trace.append(step_info)

        # 循环结束，未获得最终答案
        if self.verbose:
            print(f"\n⚠️ 已达到最大步数 ({self.max_steps})，流程终止。")
        return None

    def _parse_output(self, text: str) -> tuple:
        """
        解析LLM的输出，提取Thought和Action。
        支持多种格式，更加宽容。
        """
        thought = None
        action = None

        # 尝试匹配 Thought
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # 尝试匹配 Action（多种格式）
        action_patterns = [
            r"Action:\s*(.*?)$",           # Action: xxx
            r"^Action\s*[-:]\s*(.*?)$",    # Action - xxx 或 Action: xxx
            r"^(Finish|Search|Calculator)\[(.*)\]",  # 直接以工具开头
        ]

        for pattern in action_patterns:
            action_match = re.search(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if action_match:
                if action_match.lastindex == 1:
                    # 工具名开头
                    action = action_match.group(1).strip()
                else:
                    action = action_match.group(1).strip()
                break

        # 如果还是没有 Action，但有 Finish 关键字，也尝试提取
        if not action:
            finish_match = re.search(r"Finish\[(.*)\]", text, re.DOTALL | re.IGNORECASE)
            if finish_match:
                action = f"Finish[{finish_match.group(1).strip()}]"

        return thought, action

    def _parse_action(self, action_text: str) -> tuple:
        """
        解析Action字符串，提取工具名称和输入。

        Args:
            action_text: Action字符串，如 "Search[华为手机]"

        Returns:
            (tool_name, tool_input) 元组
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def getTrace(self) -> list:
        """
        获取完整的执行轨迹。

        Returns:
            包含每一步详细信息的列表
        """
        return self.full_trace

    def getHistory(self) -> list:
        """
        获取交互历史记录。

        Returns:
            Action和Observation的交替历史列表
        """
        return self.history


if __name__ == "__main__":
    # 简单测试提示词格式化
    print("=== ReAct提示词模板示例 ===\n")
    sample_prompt = REACT_PROMPT_TEMPLATE.format(
        tools="- Search: 网页搜索工具\n- Calculator: 计算器工具",
        question="华为最新手机是什么？",
        history="无"
    )
    print(sample_prompt)
