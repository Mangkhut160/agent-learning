"""
团队协作模块
定义团队聊天模式和终止条件
"""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination


def create_team(participants: list):
    """
    创建轮询群聊团队。

    采用 RoundRobinGroupChat 模式，按参与者列表顺序轮询发言。
    当任何消息中包含 "TERMINATE" 时，整个协作流程结束。

    Args:
        participants: 智能体列表，按发言顺序排列

    Returns:
        配置好的 RoundRobinGroupChat 实例
    """
    team_chat = RoundRobinGroupChat(
        participants=participants,
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=20,
    )
    return team_chat