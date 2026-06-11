"""
Agent范式对比示例
==================

本文件演示了 HelloAgents 框架中不同Agent范式的使用方法。

包括:
1. SimpleAgent - 基础对话
2. ReActAgent - 推理与行动
3. ReflectionAgent - 自我反思
4. PlanAndSolveAgent - 规划与执行

运行方式:
    python examples/agent_comparison.py
"""

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.agents import (
    SimpleAgent,
    ReActAgent,
    ReflectionAgent,
    PlanAndSolveAgent
)
from hello_agents.tools.builtin import CalculatorTool


def simple_agent_example(llm):
    """SimpleAgent示例 - 基础对话"""
    print("\n" + "=" * 60)
    print("SimpleAgent - 基础对话")
    print("=" * 60)

    agent = SimpleAgent(
        name="简单助手",
        llm=llm,
        system_prompt="你是一个简洁回答问题的助手。"
    )

    response = agent.run("解释什么是人工智能")
    print(f"📝 回答: {response[:100]}...")
    return agent


def react_agent_example(llm, registry):
    """ReActAgent示例 - 推理与行动"""
    print("\n" + "=" * 60)
    print("ReActAgent - 推理与行动")
    print("=" * 60)

    agent = ReActAgent(
        name="研究者",
        llm=llm,
        tool_registry=registry,
        max_steps=5
    )

    # 使用计算器解决数学问题
    response = agent.run("计算 15 * 8 + 32 的结果")
    print(f"📝 回答: {response}")
    return agent


def reflection_agent_example(llm):
    """ReflectionAgent示例 - 自我反思"""
    print("\n" + "=" * 60)
    print("ReflectionAgent - 自我反思")
    print("=" * 60)

    agent = ReflectionAgent(
        name="写作助手",
        llm=llm,
        max_iterations=2
    )

    # 文本生成任务
    response = agent.run("用三句话介绍Python编程语言")
    print(f"📝 回答: {response}")
    return agent


def plan_solve_agent_example(llm):
    """PlanAndSolveAgent示例 - 规划与执行"""
    print("\n" + "=" * 60)
    print("PlanAndSolveAgent - 规划与执行")
    print("=" * 60)

    agent = PlanAndSolveAgent(
        name="规划助手",
        llm=llm,
        max_steps=10
    )

    # 多步骤数学问题
    question = (
        "小明有10个苹果，小红给了他3个，"
        "然后小明吃了2个，小明现在还有多少个苹果？"
    )

    response = agent.run(question)
    print(f"📝 回答: {response}")
    return agent


def main():
    """运行所有Agent示例"""
    # 加载环境变量
    load_dotenv()

    print("🚀 HelloAgents Agent范式对比")
    print("=" * 60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 创建工具注册表
    registry = ToolRegistry()
    calculator = CalculatorTool()
    registry.register_tool(calculator)

    try:
        # 运行各Agent示例
        simple_agent_example(llm)
        react_agent_example(llm, registry)
        reflection_agent_example(llm)
        plan_solve_agent_example(llm)

        print("\n" + "=" * 60)
        print("✅ 所有Agent示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()