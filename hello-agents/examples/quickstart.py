"""
快速开始示例
============

本文件演示了 HelloAgents 框架的基本使用方法。
运行前请确保已配置好环境变量（参见 .env.example）。

运行方式:
    python examples/quickstart.py
"""

from dotenv import load_dotenv

# 导入框架核心组件
from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
from hello_agents.tools.builtin import CalculatorTool


def basic_chat_example():
    """基础对话示例"""
    print("=" * 60)
    print("示例1: 基础对话")
    print("=" * 60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 创建SimpleAgent
    agent = SimpleAgent(
        name="AI助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )

    # 基础对话
    response = agent.run("你好！请介绍一下自己")
    print(f"\n📝 回答: {response}")

    # 查看对话历史
    print(f"\n📚 对话历史: {len(agent.get_history())} 条消息")


def tool_calling_example():
    """工具调用示例"""
    print("\n" + "=" * 60)
    print("示例2: 工具调用")
    print("=" * 60)

    # 创建LLM和工具注册表
    llm = HelloAgentsLLM()
    registry = ToolRegistry()

    # 注册计算器工具
    calculator = CalculatorTool()
    registry.register_tool(calculator)

    print(f"\n🔧 已注册工具: {registry.list_tools()}")

    # 创建带工具的Agent
    agent = SimpleAgent(
        name="数学助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=registry,
        enable_tool_calling=True
    )

    # 使用工具计算
    print("\n🔢 计算 2 + 3 * 4:")
    response = agent.run("请帮我计算 2 + 3 * 4")
    print(f"📝 回答: {response}")

    # 使用工具计算平方根
    print("\n📐 计算 sqrt(16) + pi:")
    response = agent.run("请计算 sqrt(16) 加上圆周率大约是多少")
    print(f"📝 回答: {response}")


def streaming_example():
    """流式响应示例"""
    print("\n" + "=" * 60)
    print("示例3: 流式响应")
    print("=" * 60)

    llm = HelloAgentsLLM()

    agent = SimpleAgent(
        name="流式助手",
        llm=llm,
        system_prompt="你是一个诗人，请根据主题写一首诗。"
    )

    print("\n🌊 流式生成诗歌:")
    print("-" * 40)

    # 流式调用
    for chunk in agent.stream_run("请以'春天'为主题写一首诗"):
        pass  # 内容已在stream_run中实时打印


def history_management_example():
    """历史记录管理示例"""
    print("\n" + "=" * 60)
    print("示例4: 历史记录管理")
    print("=" * 60)

    llm = HelloAgentsLLM()

    agent = SimpleAgent(
        name="对话助手",
        llm=llm,
        system_prompt="你是一个有用的助手。"
    )

    # 多轮对话
    questions = [
        "你好，我叫小明",
        "你知道我叫什么吗？",
        "我的名字是什么？"
    ]

    for q in questions:
        print(f"\n❓ 用户: {q}")
        response = agent.run(q)
        print(f"📝 Agent: {response}")

    # 查看历史
    print(f"\n📚 对话历史 ({len(agent.get_history())} 条消息):")
    for i, msg in enumerate(agent.get_history()):
        print(f"  {i+1}. [{msg.role}] {msg.content[:50]}...")

    # 清空历史
    print("\n🗑️ 清空历史记录...")
    agent.clear_history()
    print(f"📚 对话历史: {len(agent.get_history())} 条消息")


def main():
    """运行所有示例"""
    # 加载环境变量
    load_dotenv()

    print("🚀 HelloAgents 快速开始示例")
    print("=" * 60)

    try:
        basic_chat_example()
        tool_calling_example()
        streaming_example()
        history_management_example()

        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("请确保已配置正确的环境变量（参见 .env.example）")


if __name__ == "__main__":
    main()