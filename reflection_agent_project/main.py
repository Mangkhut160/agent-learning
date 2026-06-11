"""
Reflection 智能体项目主程序

演示如何使用 Reflection 范式构建智能体，
通过"执行-反思-优化"的迭代循环来提升代码质量。
"""

import sys

from dotenv import load_dotenv

from src.llm_client import HelloAgentsLLM
from src.reflection_agent import ReflectionAgent


def main():
    """主程序入口"""
    load_dotenv()

    print("\n" + "=" * 60)
    print("🚀 Reflection 智能体项目")
    print("=" * 60)

    # 1. 初始化 LLM 客户端
    print("\n📡 正在初始化 LLM 客户端...")
    try:
        llm_client = HelloAgentsLLM()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        print("请检查 .env 文件中的 API_KEY 配置。")
        sys.exit(1)

    # 2. 初始化 Reflection 智能体
    print("\n🤖 正在初始化 Reflection 智能体...")
    agent = ReflectionAgent(
        llm_client=llm_client,
        max_iterations=2  # 限制迭代次数控制成本
    )

    # 3. 运行示例任务
    print("\n" + "=" * 60)
    print("📝 开始运行示例任务")
    print("=" * 60)

    tasks = [
        "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。",
    ]

    for task in tasks:
        final_code = agent.run(task)

        if final_code is None:
            print("\n⚠️ 智能体未能生成代码。")
        else:
            print(f"\n✅ 最终生成 {len(final_code)} 字符的代码")

        print("\n" + "-" * 60)

        # 打印完整轨迹摘要
        print("\n📊 完整执行轨迹摘要:")
        print(agent.get_trajectory())

    # 4. 交互模式（可选）
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n🎮 进入交互模式 (输入 'quit' 退出)")
        print("提示: 描述你想要的 Python 函数，智能体会迭代优化它。")
        print("-" * 60)

        while True:
            try:
                user_task = input("\n请输入编程任务: ").strip()
                if user_task.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break

                if not user_task:
                    continue

                agent.memory.clear()
                final_code = agent.run(user_task)

                if final_code:
                    print(f"\n✅ 最终代码:\n{final_code}")
                else:
                    print("\n⚠️ 未能生成代码，请尝试换个描述方式。")

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
    else:
        print("\n✅ 运行完成！如需进入交互模式，请运行: py main.py --interactive")


if __name__ == "__main__":
    main()
