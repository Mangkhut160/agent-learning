"""
结构化数据模型
用于约束智能体输出，确保游戏规则被遵守
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================
# 讨论阶段模型
# ============================================================

class DiscussionModel(BaseModel):
    """白天讨论阶段的输出格式"""
    thought_process: str = Field(
        description="你的思考推理过程"
    )
    public_statement: str = Field(
        description="你公开发表的言论"
    )
    target_suspect: Optional[str] = Field(
        default=None,
        description="你怀疑的对象（如有）"
    )
    confidence_level: int = Field(
        default=5,
        ge=1, le=10,
        description="对自己推理的信心程度（1-10）"
    )


# ============================================================
# 狼人阶段模型
# ============================================================

class WerewolfDiscussionModel(BaseModel):
    """狼人讨论输出格式"""
    reasoning: str = Field(description="你的击杀策略推理")
    suggested_target: Optional[str] = Field(
        default=None,
        description="建议击杀的目标"
    )
    reach_agreement: bool = Field(
        default=False,
        description="是否已与其他狼人达成一致"
    )


class WerewolfKillModel(BaseModel):
    """狼人击杀投票格式"""
    target_name: str = Field(description="要击杀的目标玩家姓名")
    reason: str = Field(description="选择该目标的理由")


# ============================================================
# 预言家阶段模型
# ============================================================

class SeerCheckModel(BaseModel):
    """预言家查验选择"""
    target_name: str = Field(description="要查验的玩家姓名")
    reason: str = Field(description="选择查验该玩家的理由")


# ============================================================
# 女巫阶段模型
# ============================================================

class WitchActionModel(BaseModel):
    """女巫行动选择"""
    use_antidote: bool = Field(
        default=False,
        description="是否使用解药救人"
    )
    use_poison: bool = Field(
        default=False,
        description="是否使用毒药杀人"
    )
    target_name: Optional[str] = Field(
        default=None,
        description="毒药目标玩家姓名（使用毒药时必填）"
    )
    reasoning: str = Field(description="你的决策理由")


# ============================================================
# 投票阶段模型
# ============================================================

class VoteModel(BaseModel):
    """白天投票格式"""
    vote_target: str = Field(description="投票淘汰的目标玩家姓名")
    reason: str = Field(description="投票理由")


# ============================================================
# 游戏状态
# ============================================================

class GameState(BaseModel):
    """全局游戏状态"""
    round: int = Field(default=0, description="当前轮次")
    phase: str = Field(default="init", description="当前阶段")
    alive_players: List[str] = Field(default_factory=list)
    dead_players: List[str] = Field(default_factory=list)
    werewolves: List[str] = Field(default_factory=list)
    seer: Optional[str] = Field(default=None)
    witch: Optional[str] = Field(default=None)
    witch_has_antidote: bool = Field(default=True)
    witch_has_poison: bool = Field(default=True)
    last_kill_target: Optional[str] = Field(default=None)