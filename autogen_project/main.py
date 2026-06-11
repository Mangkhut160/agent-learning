"""
AutoGen 多智能体软件开发团队 - 主程序入口

演示 AutoGen 0.7.5 的多智能体协作能力。
团队由四个角色组成：产品经理、工程师、代码审查员、用户代理。
它们将协作完成"比特币价格显示应用"的开发任务。

运行方式:
    py main.py
"""

import asyncio

from autogen_agentchat.ui import Console

from src.model_client import create_model_client
from src.agents import (
    create_product_manager,
    create_engineer,
    create_code_reviewer,
    create_user_proxy,
)
from src.team import create_team


async def run_software_development_team():
    """运行软件开发团队的异步协作流程"""

    print("\n" + "=" * 60)
    print("🚀 AutoGen 多智能体软件开发团队")
    print("=" * 60)

    # 1. 初始化模型客户端
    print("\n📡 正在初始化模型客户端...")
    try:
        model_client = create_model_client()
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 2. 创建各角色智能体
    print("\n👥 正在创建智能体团队...")
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()

    print(f"  ✅ ProductManager (产品经理)")
    print(f"  ✅ Engineer (工程师)")
    print(f"  ✅ CodeReviewer (代码审查员)")
    print(f"  ✅ UserProxy (用户代理)")

    # 3. 创建团队
    print("\n🔧 正在组建团队...")
    team = create_team([
        product_manager,
        engineer,
        code_reviewer,
        user_proxy,
    ])
    print("  ✅ 团队创建完成")

    # 4. 定义任务
    task = """
我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。
"""

    # 5. 运行团队协作
    print("\n" + "=" * 60)
    print("🏁 团队协作开始")
    print("=" * 60)

    result = await Console(team.run_stream(task=task))

    print("\n" + "=" * 60)
    print("🏆 协作完成")
    print("=" * 60)

    return result


def main():
    """同步入口，包装异步运行"""
    result = asyncio.run(run_software_development_team())
    print(f"\n📊 最终结果: {result}")


if __name__ == "__main__":
    main()