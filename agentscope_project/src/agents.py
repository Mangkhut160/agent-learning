"""
智能体管理模块
创建和管理游戏中的智能体（主持人与玩家）
"""

import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from src.roles import get_role_prompt, DEFAULT_ROLE_ASSIGNMENT

load_dotenv()


def create_model():
    """创建 LLM 模型客户端"""
    return OpenAIChatModel(
        model_name=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        client_kwargs={
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        },
        generate_kwargs={"temperature": 0.8},
    )


def create_formatter():
    """创建 OpenAI 兼容的对话格式化器"""
    return OpenAIChatFormatter()


def create_moderator(model, formatter) -> ReActAgent:
    """
    创建游戏主持人智能体。
    负责宣布游戏进程、发布公告、收集信息。
    """
    sys_prompt = """你是三国狼人杀的游戏主持人。

你的职责：
1. 用古风语言宣布游戏阶段
2. 向特定角色传达信息（如告知预言家查验结果）
3. 公布夜晚死亡结果
4. 引导投票流程
5. 不参与推理和讨论，保持中立

请始终以 JSON 格式回复：
{"announcement": "公告内容", "target": "all/玩家名", "private_info": "秘密信息(如有)"}
"""
    return ReActAgent(
        name="游戏主持人",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        max_iters=1,
    )


def create_player_agents(model, formatter, player_names: list[str]) -> dict[str, ReActAgent]:
    """
    为每位玩家创建智能体。
    每个智能体同时具有三国人物性格和狼人杀游戏角色。
    """
    agents = {}
    for name in player_names:
        role = DEFAULT_ROLE_ASSIGNMENT.get(name, "村民")
        sys_prompt = get_role_prompt(name, role)

        agent = ReActAgent(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            max_iters=1,  # 纯角色扮演，无需工具循环
        )
        agents[name] = agent
        print(f"  ✅ {name}（{role}）已就位")

    return agents


def create_all_agents(model):
    """
    创建所有游戏智能体。
    Returns: (moderator, players_dict)
    """
    formatter = create_formatter()
    moderator = create_moderator(model, formatter)
    player_names = list(DEFAULT_ROLE_ASSIGNMENT.keys())
    players = create_player_agents(model, formatter, player_names)
    return moderator, players
