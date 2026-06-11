"""
Plan-and-Solve 智能体项目主程序

演示如何使用 Plan-and-Solve 范式构建智能体，
通过"先规划，后执行"的策略解决多步骤推理问题。
"""

import sys

from dotenv import load_dotenv

from src.llm_client import HelloAgentsLLM
from src.plan_solve_agent import PlanAndSolveAgent


def main():
    """主程序入口"""
    load_dotenv()

    print("\n" + "=" * 60)
    print("🚀 Plan-and-Solve 智能体项目")
    print("=" * 60)

    # 1. 初始化 LLM 客户端
    print("\n📡 正在初始化 LLM 客户端...")
    try:
        llm_client = HelloAgentsLLM()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        print("请检查 .env 文件中的 API_KEY 配置。")
        sys.exit(1)

    # 2. 初始化 Plan-and-Solve 智能体
    print("\n🤖 正在初始化 Plan-and-Solve 智能体...")
    agent = PlanAndSolveAgent(llm_client=llm_client)

    # 3. 运行示例问题
    print("\n" + "=" * 60)
    print("📝 开始运行示例问题")
    print("=" * 60)

    questions = [
        # 1. 逻辑推理 — 需要多步推导，中间结论依赖前序步骤
        "甲、乙、丙、丁四人在一场比赛中获得了前四名。已知："
        "甲不是第一名；乙既不是第一名也不是最后一名；"
        "丙在乙之后但不在最后；丁在甲之前。"
        "请推理出四个人的具体排名（从第一到第四）。",

        # 2. 财务计算 — 多阶段复合运算，每步结果影响后续
        "小张2021年初存入银行10万元，年利率3.5%，按年复利计算。"
        "2023年初他又追加存入了5万元。"
        "请问到2025年底，他的账户总共有多少钱？（保留两位小数）",

        # 3. 代码结构设计 — 需要先规划模块再实现
        "请用 Python 设计一个简单的图书管理系统。"
        "需要支持：添加图书（书名+作者+ISBN）、删除图书、根据书名搜索图书、"
        "列出所有图书。请先规划出类和方法的结构，再给出完整代码。",

        # 4. 多实体比例问题 — 复杂的数量关系推导
        "一个农场有鸡、鸭、鹅三种家禽共360只。"
        "鸡的数量是鸭的3倍，鸭的数量比鹅多20只。"
        "后来卖掉了四分之一的鸡和一半的鸭。"
        "请问卖掉后农场的鸡比鹅多多少只？",
    ]

    for question in questions:
        answer = agent.run(question)

        if answer is None:
            print("\n⚠️ 智能体未能得出答案。")
        else:
            print(f"\n✅ 最终回答: {answer}")

        print("\n" + "-" * 60)

    # 4. 交互模式（可选）
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
        print("\n✅ 运行完成！如需进入交互模式，请运行: python main.py --interactive")


if __name__ == "__main__":
    main()
