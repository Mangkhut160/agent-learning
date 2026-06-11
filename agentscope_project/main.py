"""
AgentScope 三国狼人杀 - 主程序入口

基于 AgentScope 1.0 的消息驱动架构，
构建融合三国文化的多智能体狼人杀游戏。

运行方式:
    py main.py
"""

import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

from src.game_controller import GameController, GamePhase
from src.agents import create_model, create_all_agents
from src.game_flow import (
    run_init_phase,
    run_guard_phase,
    run_werewolf_phase,
    run_seer_phase,
    run_witch_phase,
    run_dawn_announcement,
    run_day_discussion,
    run_voting,
)

load_dotenv()

LOG_FILE = os.path.join(os.path.dirname(__file__), "werewolf_game_log.txt")


class TeeLogger:
    """同时输出到控制台和文件的日志器"""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()  # 实时写入，不缓冲

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


MAX_ROUNDS = 3  # 最大游戏轮次（减少以加速）


async def main():
    """异步主函数：运行完整的一局三国狼人杀"""

    # 设置双输出
    logger = TeeLogger(LOG_FILE)
    sys.stdout = logger

    print(f"\n📝 游戏日志将保存到: {LOG_FILE}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60)
    print("🏯 三国狼人杀 — AgentScope 多智能体演示")
    print("=" * 60)

    # 1. 创建模型和智能体
    print("\n📡 正在初始化模型和智能体...")
    model = create_model()
    moderator, players = create_all_agents(model)

    # 2. 初始化游戏控制器
    from src.roles import DEFAULT_ROLE_ASSIGNMENT
    player_names = list(DEFAULT_ROLE_ASSIGNMENT.keys())
    controller = GameController(player_names, DEFAULT_ROLE_ASSIGNMENT)

    # 3. 初始化阶段
    await run_init_phase(controller, moderator, players)

    # 4. 游戏主循环
    for round_num in range(1, MAX_ROUNDS + 1):
        controller.round_num = round_num

        print(f"\n{'=' * 60}")
        print(f"🏴 第 {round_num} 轮游戏")
        print(f"{'=' * 60}")

        if controller.winner:
            break

        # 夜晚阶段（按正确顺序：守卫→狼人→预言家→女巫）
        await run_guard_phase(controller, moderator, players)
        await asyncio.sleep(0.3)

        await run_werewolf_phase(controller, moderator, players)
        await asyncio.sleep(0.3)

        await run_seer_phase(controller, moderator, players)
        await asyncio.sleep(0.3)

        await run_witch_phase(controller, moderator, players)
        await asyncio.sleep(0.3)

        # 天亮公告
        await run_dawn_announcement(controller, moderator, players)

        if controller.winner:
            break

        # 白天阶段
        await run_day_discussion(controller, moderator, players)
        await asyncio.sleep(0.5)

        # 投票
        eliminated = await run_voting(controller, moderator, players)
        await asyncio.sleep(0.5)

        if controller.winner:
            break

        print(f"\n  📊 当前存活: {controller.get_alive_players_str()}")

    # 5. 宣布结果
    print(f"\n{'=' * 60}")
    print(f"🏆 游戏结束")
    print(f"{'=' * 60}")

    winner = controller.winner or "平局"
    print(f"\n胜利方: {winner}")
    print(f"存活玩家: {controller.get_alive_players_str()}")

    # 揭示所有身份
    print(f"\n📋 身份揭示:")
    for name in player_names:
        role = controller.get_player_role(name)
        status = "存活" if controller.is_alive(name) else "阵亡"
        print(f"  {name}: {role}（{status}）")

    print(f"\n📜 游戏日志:")
    for log in controller.history:
        print(f"  {log}")

    print(f"\n{'=' * 60}")
    print(f"🎮 游戏结束，感谢观看！")
    print(f"{'=' * 60}")

    print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 完整日志已保存到: {LOG_FILE}")

    sys.stdout = logger.stdout
    logger.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ 游戏异常: {e}")
        import traceback
        traceback.print_exc()
