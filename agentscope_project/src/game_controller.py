"""
游戏控制器模块
管理游戏状态、阶段流转、胜负判定、死亡结算
"""

from enum import Enum
from typing import Optional, List, Dict
from src.models import GameState


class GamePhase(Enum):
    INIT = "init"
    GUARD_NIGHT = "guard_night"
    WEREWOLF_NIGHT = "werewolf_night"
    SEER_NIGHT = "seer_night"
    WITCH_NIGHT = "witch_night"
    DAWN_ANNOUNCEMENT = "dawn_announcement"
    DAY_DISCUSSION = "day_discussion"
    VOTING = "voting"
    GAME_OVER = "game_over"


class GameController:
    """三国狼人杀游戏主控制器。"""

    def __init__(self, players: List[str], role_assignment: Dict[str, str]):
        self.players: Dict[str, Dict] = {}
        for name in players:
            werewolf_role = role_assignment.get(name, "村民")
            self.players[name] = {
                "role": werewolf_role,
                "alive": True,
                "character": name,
            }

        self.alive_players: List[str] = list(players)
        self.dead_players: List[str] = []
        self.history: List[str] = []

        # 角色分组
        self.werewolves = [n for n, p in self.players.items() if p["role"] == "狼人"]
        self.guard = next((n for n, p in self.players.items() if p["role"] == "守卫"), None)
        self.seer = next((n for n, p in self.players.items() if p["role"] == "预言家"), None)
        self.witch = next((n for n, p in self.players.items() if p["role"] == "女巫"), None)

        # 女巫药物
        self.witch_has_antidote = True
        self.witch_has_poison = True

        # 守卫追踪 — 连续两晚不能守同一人
        self.guard_last_target: Optional[str] = None

        # 每夜行动记录（用于天亮结算）
        self._night_kill_target: Optional[str] = None   # 狼人选的目标
        self._night_guard_target: Optional[str] = None  # 守卫守的目标
        self._night_witch_save: bool = False            # 女巫是否用了解药
        self._night_witch_poison_target: Optional[str] = None  # 女巫毒杀目标

        self.round_num: int = 0
        self.phase: GamePhase = GamePhase.INIT

    # ================================================================
    # 胜负判定
    # ================================================================

    @property
    def winner(self) -> Optional[str]:
        alive = set(self.alive_players)
        wolves_alive = alive & set(self.werewolves)
        villagers_alive = alive - set(self.werewolves)

        if not wolves_alive:
            return "村民阵营"
        if not villagers_alive:
            return "狼人阵营"
        if len(wolves_alive) >= len(villagers_alive):
            return "狼人阵营"
        return None

    # ================================================================
    # 夜间行动记录
    # ================================================================

    def set_werewolf_kill(self, target: str):
        self._night_kill_target = target
        self.history.append(f"[狼人] 决定击杀 {target}")

    def set_guard_protect(self, target: str) -> str | None:
        """守卫守护。返回 None 表示成功，否则返回错误信息。"""
        if target == self.guard_last_target:
            return f"规则禁止：守卫不能连续两晚守护同一个人（{target}）"
        self._night_guard_target = target
        self.guard_last_target = target
        self.history.append(f"[守卫] 守护 {target}")
        return None

    def set_seer_check(self, target: str) -> str:
        """预言家查验。返回 '狼人' 或 '好人'。"""
        role = self.players.get(target, {}).get("role", "村民")
        result = "狼人" if role == "狼人" else "好人"
        self.history.append(f"[预言家] 查验 {target} → {result}")
        return result

    def set_witch_antidote(self, target: str) -> str | None:
        """女巫使用解药。返回 None 成功，否则返回错误信息。"""
        if not self.witch_has_antidote:
            return "你的解药已经用过了"
        self._night_witch_save = True
        self.witch_has_antidote = False
        self.history.append(f"[女巫] 使用解药救 {target}")
        return None

    def set_witch_poison(self, target: str) -> str | None:
        """女巫使用毒药。返回 None 成功，否则返回错误信息。"""
        if not self.witch_has_poison:
            return "你的毒药已经用过了"
        self._night_witch_poison_target = target
        self.witch_has_poison = False
        self.history.append(f"[女巫] 使用毒药毒杀 {target}")
        return None

    # ================================================================
    # 天亮结算 — 核心规则引擎
    # ================================================================

    def resolve_night(self) -> List[str]:
        """
        天亮时结算所有夜间行动。
        返回：昨夜死亡的玩家列表。

        结算规则（按顺序）：
        1. 守卫守护 且 未被狼人选为目标 → 活（无事发生）
        2. 狼人选的目标 同时被 守卫守护 且 女巫也用了解药 → 死（"同守同救"规则）
        3. 狼人选的目标 被 守卫守护 但女巫未救 → 活
        4. 狼人选的目标 被 女巫解药救 → 活
        5. 狼人选的目标 无人守也无人救 → 死
        6. 女巫毒药目标 → 死（无视守卫）
        """
        dead_tonight: List[str] = []

        kill = self._night_kill_target
        guard = self._night_guard_target
        saved = self._night_witch_save
        poison = self._night_witch_poison_target

        # 处理狼人击杀
        if kill:
            guarded = (guard == kill)   # 守卫是否守了被杀者
            antidote_used = saved       # 女巫是否用了救药

            if guarded and antidote_used:
                # ★ 核心规则：同守同救 = 死亡
                dead_tonight.append(kill)
                self.history.append(f"[结算] {kill} 同时被守卫守护和女巫解救，不幸身亡！")
            elif guarded and not antidote_used:
                # 仅守卫守护 → 活
                self.history.append(f"[结算] {kill} 被守卫守护，平安无事")
            elif antidote_used and not guarded:
                # 仅女巫解救 → 活
                self.history.append(f"[结算] {kill} 被女巫解药救活")
            else:
                # 无人守护 → 死
                dead_tonight.append(kill)
                self.history.append(f"[结算] {kill} 被狼人击杀")

        # 处理女巫毒药（无视守卫守护）
        if poison:
            dead_tonight.append(poison)
            self.history.append(f"[结算] {poison} 被女巫毒杀")

        # 执行死亡
        for name in dead_tonight:
            self.kill_player(name)

        # 重置夜间记录
        self._night_kill_target = None
        self._night_guard_target = None
        self._night_witch_save = False
        self._night_witch_poison_target = None

        return dead_tonight

    # ================================================================
    # 基本操作
    # ================================================================

    def kill_player(self, name: str, reason: str = "被击杀"):
        if name in self.alive_players:
            self.alive_players.remove(name)
            self.dead_players.append(name)
            self.players[name]["alive"] = False
            # 移除可能有多个理由，只追加 name 到已有多条
            return True
        return False

    def vote_eliminate(self, name: str):
        """投票淘汰"""
        return self.kill_player(name, "被投票淘汰")

    def get_alive_players_str(self) -> str:
        return "、".join(self.alive_players)

    def get_player_role(self, name: str) -> str:
        return self.players.get(name, {}).get("role", "未知")

    def is_alive(self, name: str) -> bool:
        return name in self.alive_players

    def get_night_info_for_witch(self) -> str:
        """生成告知女巫的夜间信息"""
        info = ""
        if self._night_kill_target:
            info += f"今晚 {self._night_kill_target} 被狼人袭击。"
        else:
            info += "今晚无人被狼人袭击。"
        info += f" 解药{'可用' if self.witch_has_antidote else '已用'}，"
        info += f"毒药{'可用' if self.witch_has_poison else '已用'}。"
        return info

    def get_guard_rule_reminder(self) -> str:
        """守卫规则提示"""
        if self.guard_last_target:
            return f"（你昨晚守护了 {self.guard_last_target}，今晚不能再次守护此人）"
        return ""

    def to_state(self) -> GameState:
        return GameState(
            round=self.round_num,
            phase=self.phase.value,
            alive_players=list(self.alive_players),
            dead_players=list(self.dead_players),
            werewolves=list(self.werewolves),
            seer=self.seer,
            witch=self.witch,
            witch_has_antidote=self.witch_has_antidote,
            witch_has_poison=self.witch_has_poison,
            last_kill_target=self._night_kill_target,
        )
