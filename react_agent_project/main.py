"""
ReAct 智能体项目主程序

演示如何使用 ReAct 范式构建一个能够调用外部工具的智能助手。
"""

import sys

from dotenv import load_dotenv

from src.llm_client import HelloAgentsLLM
from src.tools import search, calculator, SEARCH_DESCRIPTION, CALCULATOR_DESCRIPTION
from src.tool_executor import ToolExecutor
from src.agent import ReActAgent


def main():
    """主程序入口"""
    # 加载环境变量
    load_dotenv()

    print("\n" + "=" * 60)
    print("🚀 ReAct 智能体项目")
    print("=" * 60)

    # 1. 初始化 LLM 客户端
    print("\n📡 正在初始化 LLM 客户端...")
    try:
        llm_client = HelloAgentsLLM()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        print("请检查 .env 文件中的 API_KEY 配置。")
        sys.exit(1)

    # 2. 初始化工具执行器并注册工具
    print("\n🔧 正在初始化工具执行器...")
    tool_executor = ToolExecutor()

    # 注册搜索工具
    tool_executor.registerTool("Search", SEARCH_DESCRIPTION, search)

    # 注册计算器工具（可选）
    tool_executor.registerTool("Calculator", CALCULATOR_DESCRIPTION, calculator)

    # 打印可用工具
    print("\n📋 可用工具列表:")
    print(tool_executor.getAvailableTools())

    # 3. 初始化 ReAct 智能体
    print("\n🤖 正在初始化 ReAct 智能体...")
    agent = ReActAgent(
        llm_client=llm_client,
        tool_executor=tool_executor,
        max_steps=15,  # 增加最大步数
        verbose=True
    )

    # 4. 运行示例问题
    print("\n" + "=" * 60)
    print("📝 开始运行示例问题")
    print("=" * 60)

    # 示例问题列表
    questions = [
        "华为最新手机是什么型号？",
        # "英伟达最新的GPU型号是什么？",
        # "Python最新版本是多少？",
    ]

    for question in questions:
        answer = agent.run(question)

        if answer is None:
            print("\n⚠️ 智能体未能在限定步数内得出答案。")
        else:
            print(f"\n✅ 最终回答: {answer}")

        print("\n" + "-" * 60)

    # 5. 交互模式（可选）
    # 默认跳过交互模式，直接退出
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n🎮 进入交互模式 (输入 'quit' 退出)")
        print("-" * 60)

        while True:
            try:
                user_question = input("\n请输入问题: ").strip()
                if user_question.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break

                if not user_question:
                    continue

                answer = agent.run(user_question)

                if answer:
                    print(f"\n✅ 回答: {answer}")
                else:
                    print("\n⚠️ 未能得出答案，请尝试换个问法。")

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
    else:
        print("\n✅ 运行完成！如需进入交互模式，请运行: py main.py --interactive")


if __name__ == "__main__":
    main()
