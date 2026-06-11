"""
SimpleAgent - 基础对话Agent
===========================

SimpleAgent是HelloAgents框架中最基础的Agent实现，展示了如何基于框架
构建一个完整的对话智能体。它支持可选的工具调用功能、流式响应和
便利的工具管理方法。

核心功能:
1. 基础对话：接收用户输入，返回LLM生成的响应
2. 工具调用：支持通过工具扩展Agent能力
3. 流式响应：支持实时流式输出
4. 历史管理：自动记录对话历史

设计特点:
- 继承自Agent基类，遵循统一接口规范
- 支持可配置的工具调用系统
- 对话历史自动管理

使用示例:
    from hello_agents import HelloAgentsLLM, SimpleAgent

    llm = HelloAgentsLLM()
    agent = SimpleAgent(name="助手", llm=llm, system_prompt="你是一个友好的助手")

    # 基础对话
    response = agent.run("你好，请介绍一下自己")
    print(response)

    # 流式响应
    for chunk in agent.stream_run("请写一首诗"):
        print(chunk, end="", flush=True)
"""

from typing import Optional, Iterator, List, Dict, Any

from ..core.agent import Agent
from ..core.message import Message
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """
    简单对话Agent - 基础对话智能体实现

    SimpleAgent是框架中最简单的Agent实现，适合处理日常对话任务。
    它通过组合系统提示词、历史消息和用户输入来构建完整的对话上下文，
    然后调用LLM生成响应。

    继承自:
        Agent: 抽象基类，定义Agent的通用接口

    额外属性:
        tool_registry: 工具注册表，用于管理可用的工具（可选）
        enable_tool_calling: 是否启用工具调用功能

    Example:
        >>> agent = SimpleAgent(name="小助手", llm=llm)
        >>> response = agent.run("今天天气怎么样？")
        >>> print(response)
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True
    ):
        """
        初始化SimpleAgent

        Args:
            name: Agent名称
            llm: HelloAgentsLLM实例
            system_prompt: 系统提示词，设定Agent角色
            config: 配置对象
            tool_registry: 工具注册表，用于工具调用
            enable_tool_calling: 是否启用工具调用
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        运行Agent处理用户输入

        SimpleAgent的run方法实现以下逻辑：
        1. 构建消息列表（系统消息 + 历史消息 + 用户输入）
        2. 如果启用工具调用，进入工具调用循环
        3. 否则直接调用LLM获取响应
        4. 保存对话到历史记录
        5. 返回最终响应

        Args:
            input_text: 用户输入的文本
            max_tool_iterations: 最大工具调用迭代次数，防止无限循环
            **kwargs: 其他参数，将传递给LLM调用

        Returns:
            Agent生成的最终响应文本
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = self._build_messages(input_text)

        # 根据是否启用工具调用选择不同的处理逻辑
        if not self.enable_tool_calling:
            # 简单对话模式：直接调用LLM
            response = self.llm.invoke(messages, **kwargs)
            self._save_to_history(input_text, response)
            print(f"✅ {self.name} 响应完成")
            return response

        # 工具调用模式：支持多轮工具调用
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """
        构建发送给LLM的消息列表

        消息列表包含：
        1. 系统消息（如果提供了system_prompt）
        2. 历史消息（之前的对话记录）
        3. 当前用户输入

        Args:
            input_text: 用户输入文本

        Returns:
            符合LLM API格式的消息列表
        """
        messages = []

        # 添加系统消息
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        if enhanced_system_prompt:
            messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        return messages

    def _get_enhanced_system_prompt(self) -> str:
        """
        构建增强的系统提示词

        如果启用了工具调用，会在系统提示词中添加工具描述，
        告诉LLM有哪些工具可用以及如何调用。

        Returns:
            增强后的系统提示词
        """
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        # 构建工具部分
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:calculator:2+3]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    def _run_with_tools(
        self,
        messages: List[Dict[str, str]],
        input_text: str,
        max_tool_iterations: int,
        **kwargs
    ) -> str:
        """
        支持工具调用的运行逻辑

        工具调用循环：
        1. 调用LLM获取响应
        2. 检查响应中是否包含工具调用标记
        3. 如果有工具调用，执行工具并获取结果
        4. 将工具结果反馈给LLM继续生成
        5. 重复直到没有工具调用或达到最大迭代次数

        Args:
            messages: 消息列表
            input_text: 用户原始输入
            max_tool_iterations: 最大迭代次数
            **kwargs: 其他参数

        Returns:
            最终响应文本
        """
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 调用LLM
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # 没有工具调用，这是最终回答
                final_response = response
                break

            print(f"🔧 检测到 {len(tool_calls)} 个工具调用")

            # 执行所有工具调用并收集结果
            tool_results_text = ""
            clean_response = response

            for call in tool_calls:
                result = self._execute_tool_call(call['tool_name'], call['parameters'])
                tool_results_text += f"\n{result}"
                # 从响应中移除工具调用标记
                clean_response = clean_response.replace(call['original'], "")

            # 添加助手消息（不含工具调用标记）
            if clean_response.strip():
                messages.append({"role": "assistant", "content": clean_response})
            else:
                messages.append({"role": "assistant", "content": "（工具执行中...）"})

            # 添加工具结果消息
            messages.append({
                "role": "user",
                "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"
            })

            current_iteration += 1
            print(f"📍 工具调用迭代: {current_iteration}/{max_tool_iterations}")

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # 保存到历史记录
        self._save_to_history(input_text, final_response)
        print(f"✅ {self.name} 响应完成")

        return final_response

    def _parse_tool_calls(self, text: str) -> List[Dict[str, str]]:
        """
        解析文本中的工具调用

        使用正则表达式匹配 `[TOOL_CALL:工具名:参数]` 格式的调用标记。

        Args:
            text: 包含工具调用的文本

        Returns:
            工具调用列表，每个调用包含 tool_name, parameters, original
        """
        import re
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            工具执行结果
        """
        if not self.tool_registry:
            return f"❌ 错误: 未配置工具注册表"

        try:
            # 执行工具
            result = self.tool_registry.execute_tool(tool_name, parameters)
            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败: {str(e)}"

    def _save_to_history(self, input_text: str, response: str) -> None:
        """
        保存对话到历史记录

        Args:
            input_text: 用户输入
            response: Agent响应
        """
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent

        此方法实时返回LLM生成的文本片段，适用于需要实时显示生成过程的场景。

        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        # 构建消息列表
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")

        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk

        print()  # 换行

        # 保存完整对话到历史记录
        self._save_to_history(input_text, full_response)
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """
        添加工具到Agent（便利方法）

        如果尚未创建工具注册表，此方法会创建一个新的。
        启用工具调用功能并注册提供的工具。

        Args:
            tool: Tool实例或函数
        """
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 根据工具类型注册
        if hasattr(tool, 'name') and hasattr(tool, 'run'):
            # Tool对象
            self.tool_registry.register_tool(tool)
        else:
            # 函数
            self.tool_registry.register_function(
                name=tool.__name__,
                description=tool.__doc__ or "自定义函数",
                func=tool
            )

        tool_name = getattr(tool, "name", tool.__name__)
        print(f"🔧 工具 '{tool_name}' 已添加")

    def has_tools(self) -> bool:
        """
        检查是否有可用工具

        Returns:
            如果启用了工具调用且有注册工具，返回True
        """
        return self.enable_tool_calling and self.tool_registry is not None

    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具（便利方法）

        Args:
            tool_name: 要移除的工具名称

        Returns:
            是否成功移除
        """
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False

    def list_tools(self) -> List[str]:
        """
        列出所有可用工具

        Returns:
            工具名称列表
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []