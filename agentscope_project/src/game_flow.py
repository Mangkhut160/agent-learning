"""
游戏流程模块
实现狼人杀各阶段的交互逻辑（含完整规则执行）
"""

import json
import re
from typing import Optional

from agentscope.message import Msg
from agentscope.agent import ReActAgent
from src.game_controller import GameController, GamePhase


# ============================================================
# 辅助函数
# ============================================================

def _escape_json_controls(json_str: str) -> str:
    """状态机：将 JSON 字符串值内的控制字符转为转义序列。"""
    result = []
    in_string = False
    escaped = False
    for c in json_str:
        if escaped:
            result.append(c)
            escaped = False
        elif c == '\\':
            result.append(c)
            escaped = True
        elif c == '"':
            result.append(c)
            in_string = not in_string
        elif in_string and ord(c) < 32:
            if c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            else:
                result.append(' ')
        else:
            result.append(c)
    return ''.join(result)


def _extract_json(text: str) -> Optional[dict]:
    """从智能体响应中提取 JSON 内容（容错解析）。"""
    if not text:
        return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    json_str = match.group()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    cleaned = _escape_json_controls(json_str)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    return None


async def _send_and_get_json(agent: ReActAgent, msg_text: str) -> Optional[dict]:
    """向智能体发送消息并解析 JSON 回复。"""
    msg = Msg(name="游戏主持人", content=msg_text, role="user")
    try:
        response = await agent(msg)
        content = response.get_text_content() if hasattr(response, 'get_text_content') else str(response)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        result = _extract_json(content)
        if result:
            stmt = result.get("public_statement", "") or result.get("announcement", "")
            print(f"    [{agent.name}]: {stmt[:150]}")
        else:
            print(f"    [{agent.name}]: {content[:150]}...")
        return result
    except Exception as e:
        print(f"    WARNING [{agent.name}] error: {e}")
        return None


def _format_players(players: list[str]) -> str:
    return "、".join(players)


def _build_choice_prompt(instruction: str, valid_names: list[str]) -> str:
    """构建带选项约束的提示词。强制 LLM 从给定列表中选一个名字。"""
    options = "\n".join(f"  - {n}" for n in valid_names)
    return f"""{instruction}

可选目标（你必须从以下名单中选择一个，写出完整姓名）：
{options}

请在你的 JSON 回复中，将 target_name 字段设为上面名单中的一个姓名。"""


def _extract_name_strict(result: dict | None, valid_names: list[str]) -> str | None:
    """严格从 JSON 结果中提取目标名——必须是 valid_names 里的精确匹配。"""
    if not result:
        return None
    # 优先取 target_name 字段
    for field in ["target_name", "vote_target", "poison_target"]:
        val = result.get(field, "")
        if isinstance(val, str) and val.strip() in valid_names:
            return val.strip()
    # 其次取 action 字段，尝试匹配
    action = result.get("action", "")
    if isinstance(action, str):
        for name in valid_names:
            if name in action:
                return name
    return None


def _extract_player_name(text: str | None, alive: list[str],
                        self_name: str = "", exclude: list[str] | None = None,
                        priority: str = "") -> str | None:
    """从 LLM 的自然语言 action 中提取玩家名，含程序化规则校验。

    - self_name: 如果 text 里出现 自己/self/自保 等自指关键词，返回 self_name
    - exclude: 排除列表（如女巫毒药不能毒自己），匹配到排除名时跳过
    - priority: 聚焦 priority 字符串周围的名字
    """
    if not text:
        return None
    if isinstance(text, dict):
        text = str(text)
    if exclude is None:
        exclude = []

    # ---- 规则1: 自指关键词 → 映射到 self_name ----
    self_keywords = ["自己", "我自", "自守", "自保", "自身", "自我", "self", "myself"]
    if self_name and any(kw in text for kw in self_keywords):
        return self_name

    # ---- 规则2: priority 聚焦搜索 ----
    if priority and priority in text:
        idx = text.rfind(priority)
        focus = text[max(0, idx - 15): idx + len(priority) + 15]
        for name in alive:
            if name in focus and name not in exclude:
                return name

    # ---- 规则3: 全局精确匹配（反向，排除列表） ----
    for name in reversed(alive):
        if name in text and name not in exclude:
            return name

    # ---- 规则4: 模糊匹配（拼音） ----
    candidates = {
        "zhuge": "诸葛亮", "caocao": "曹操", "zhouyu": "周瑜",
        "zhangfei": "张飞", "sima": "司马懿", "zhaoyun": "赵云",
    }
    text_lower = text.lower()
    for key, name in candidates.items():
        if key in text_lower and name not in exclude:
            return name
    return None


# ============================================================
# 初始化阶段
# ============================================================

async def run_init_phase(controller, moderator, players):
    print("\n" + "=" * 60)
    print("GAME 三国狼人杀")
    print("=" * 60)

    alive = controller.alive_players
    print(f"\nParticipants: {_format_players(alive)}")

    for name, agent in players.items():
        role = controller.get_player_role(name)
        intro = await _send_and_get_json(
            agent,
            f"你的角色是{role}。请用{name}的风格做简短自我介绍，表达你对这场游戏的期待。"
        )
        if intro:
            stmt = intro.get("public_statement", "")
            print(f"  [P] {name}({role}): {stmt[:120]}...")

    print(f"\nSetup done: {len(alive)} players")
    controller.phase = GamePhase.GUARD_NIGHT


# ============================================================
# 1. 守卫阶段
# ============================================================

async def run_guard_phase(controller, moderator, players):
    print(f"\n{'─' * 40}")
    print(f"[Guard Phase] Night {controller.round_num}")

    if not controller.guard or not controller.is_alive(controller.guard):
        print("  Guard is dead, skip")
        controller.phase = GamePhase.WEREWOLF_NIGHT
        return

    alive_str = controller.get_alive_players_str()
    reminder = controller.get_guard_rule_reminder()
    valid_names = controller.alive_players  # 可选目标 = 所有存活玩家

    prompt = _build_choice_prompt(
        f"（守卫阶段）请选择今晚要守护的人。{reminder}",
        valid_names
    )

    result = await _send_and_get_json(players[controller.guard], prompt)
    target = _extract_name_strict(result, valid_names)

    # 重试循环（守卫不能连续守同一人时自动重试）
    for attempt in range(2):
        if target:
            err = controller.set_guard_protect(target)
            if not err:
                print(f"  Guard -> {target}")
                break
            print(f"  BLOCKED: {err}")
        else:
            print(f"  Guard invalid target in: {str(result.get('action', ''))[:80]}")
        # 重试
        retry = await _send_and_get_json(
            players[controller.guard],
            f"刚才的选择无效。{err if target else '目标名不在可选列表中'}。"
            f"请从以下名单中选择：{alive_str}"
        )
        target = _extract_name_strict(retry, valid_names)
        if not target:
            target = _extract_player_name(
                retry.get("target_name") or retry.get("action"),
                valid_names, self_name=controller.guard
            )

    controller.phase = GamePhase.WEREWOLF_NIGHT


# ============================================================
# 2. 狼人阶段
# ============================================================

async def run_werewolf_phase(controller, moderator, players):
    print(f"\n[Werewolf Phase]")

    wolves = [w for w in controller.werewolves if controller.is_alive(w)]
    if not wolves:
        controller.phase = GamePhase.SEER_NIGHT
        return

    alive_str = controller.get_alive_players_str()

    # 讨论
    for r in range(1):
        print(f"  Wolves discuss (round {r+1})...")
        for wolf_name in wolves:
            result = await _send_and_get_json(
                players[wolf_name],
                f"（狼人秘密协商）存活玩家：{alive_str}。请与其他狼人讨论今晚的击杀目标。"
            )
            if result:
                stmt = result.get("public_statement", "")
                if stmt:
                    print(f"    {wolf_name}: {stmt[:120]}")

    # 投票（带选项约束）
    kill_votes: dict[str, int] = {}
    vote_prompt = _build_choice_prompt("请投票选择今晚的击杀目标。", controller.alive_players)
    for wolf_name in wolves:
        result = await _send_and_get_json(players[wolf_name], vote_prompt)
        target = _extract_name_strict(result, controller.alive_players)
        if not target:
            target = _extract_player_name(
                result.get("target_name") or result.get("action"),
                controller.alive_players
            )
        if target:
            kill_votes[target] = kill_votes.get(target, 0) + 1
            print(f"    {wolf_name} -> {target}")

    if kill_votes:
        chosen = max(kill_votes, key=kill_votes.get)
        controller.set_werewolf_kill(chosen)
        print(f"  Wolf target: {chosen}")
    else:
        print(f"  Wolves failed to agree")

    controller.phase = GamePhase.SEER_NIGHT


# ============================================================
# 3. 预言家阶段
# ============================================================

async def run_seer_phase(controller, moderator, players):
    print(f"\n[Seer Phase]")

    if not controller.seer or not controller.is_alive(controller.seer):
        print("  Seer is dead, skip")
        controller.phase = GamePhase.WITCH_NIGHT
        return

    alive_str = controller.get_alive_players_str()
    prompt = _build_choice_prompt("（预言家查验阶段）请选择要查验的玩家。", controller.alive_players)
    result = await _send_and_get_json(players[controller.seer], prompt)

    if result:
        target = _extract_name_strict(result, controller.alive_players)
        if not target:
            target = _extract_player_name(
                result.get("target_name") or result.get("action"),
                controller.alive_players
            )
        if target:
            role_result = controller.set_seer_check(target)
            print(f"  Seer checked {target} -> {role_result}")
            await _send_and_get_json(
                players[controller.seer],
                f"查验结果：{target} 是{role_result}。请记住此信息。"
            )

    controller.phase = GamePhase.WITCH_NIGHT


# ============================================================
# 4. 女巫阶段
# ============================================================

async def run_witch_phase(controller, moderator, players):
    print(f"\n[Witch Phase]")

    if not controller.witch or not controller.is_alive(controller.witch):
        print("  Witch is dead, skip")
        controller.phase = GamePhase.DAWN_ANNOUNCEMENT
        return

    witch = controller.witch
    info = controller.get_night_info_for_witch()
    info += f" 存活玩家：{controller.get_alive_players_str()}。"

    result = await _send_and_get_json(players[witch], f"（女巫阶段）{info}")

    if result:
        # 推断解药使用（显式字段 + action文本推断）
        action_text = (result.get("action", "") or "") + (result.get("public_statement", "") or "")
        wants_antidote = (
            result.get("use_antidote") is True or
            bool(re.search(r"使用解药|用解药救|救[自己人我]|antidote", action_text))
        )
        # 毒药：必须明确是"使用毒药毒杀XX"格式，避免"先留着毒药"误触发
        wants_poison = (
            result.get("use_poison") is True or
            bool(re.search(r"毒杀|毒死|使用毒药毒|用毒药毒", action_text))
        )

        # 解药
        if wants_antidote and controller.witch_has_antidote:
            kill_target = controller._night_kill_target
            if kill_target:
                err = controller.set_witch_antidote(kill_target)
                if err:
                    print(f"  Witch antidote error: {err}")
                else:
                    print(f"  Witch used antidote on {kill_target}")

        # 毒药（排除女巫自己 + 不能用毒在狼人杀目标上）
        if wants_poison and controller.witch_has_poison:
            valid_poison_targets = [n for n in controller.alive_players
                                    if n != controller.witch]  # 不能毒自己
            poison_prompt = _build_choice_prompt(
                "（女巫阶段）请选择毒药目标。注意：不能毒自己！",
                valid_poison_targets
            )
            poison_result = await _send_and_get_json(players[controller.witch], poison_prompt)
            target = _extract_name_strict(poison_result, valid_poison_targets)
            if not target:
                target = _extract_player_name(
                    poison_result.get("target_name") or poison_result.get("poison_target") or poison_result.get("action", ""),
                    valid_poison_targets,
                    priority="毒"
                )
            # 程序兜底：绝对不能毒自己
            if target == controller.witch:
                print(f"  BLOCKED: 女巫试图毒自己，已拦截")
                target = None
            if target:
                err = controller.set_witch_poison(target)
                if err:
                    print(f"  Witch poison error: {err}")
                else:
                    print(f"  Witch poisoned {target}")

    controller.phase = GamePhase.DAWN_ANNOUNCEMENT


# ============================================================
# 5. 天亮公告（含结算）
# ============================================================

async def run_dawn_announcement(controller, moderator, players):
    print(f"\n{'─' * 40}")
    print(f"[Dawn] Round {controller.round_num}")

    dead = controller.resolve_night()

    if not dead:
        print("  Last night was peaceful - nobody died.")
    else:
        for name in dead:
            role = controller.get_player_role(name)
            print(f"  DEAD: {name} ({role})")

    # 告知所有存活玩家夜晚结果
    for name in controller.alive_players:
        if name in players:
            if not dead:
                await _send_and_get_json(players[name], "昨夜无人死亡。")
            else:
                await _send_and_get_json(
                    players[name],
                    f"昨夜 {_format_players(dead)} 死亡。"
                )

    controller.phase = GamePhase.DAY_DISCUSSION


# ============================================================
# 6. 白天讨论
# ============================================================

async def run_day_discussion(controller, moderator, players):
    print(f"\n[Day Discussion]")
    print(f"  Alive: {controller.get_alive_players_str()}")

    for name in controller.alive_players:
        if name not in players:
            continue
        result = await _send_and_get_json(
            players[name],
            f"（白天公开讨论）存活玩家：{controller.get_alive_players_str()}。"
            f"请以{name}的风格发表分析和推理。你怀疑谁？为什么？"
        )
        if result:
            stmt = result.get("public_statement", "")
            if stmt:
                print(f"  [SAY] {name}: {stmt[:150]}...")

    controller.phase = GamePhase.VOTING


# ============================================================
# 7. 投票阶段
# ============================================================

async def run_voting(controller, moderator, players) -> Optional[str]:
    print(f"\n[Voting]")

    alive_str = controller.get_alive_players_str()
    votes: dict[str, int] = {}
    vote_prompt = _build_choice_prompt(
        "（投票淘汰阶段）请选择要投票淘汰的玩家。",
        controller.alive_players
    )

    for name in controller.alive_players:
        if name not in players:
            continue
        result = await _send_and_get_json(players[name], vote_prompt)
        target = _extract_name_strict(result, controller.alive_players)
        if not target:
            target = _extract_player_name(
                result.get("vote_target") or result.get("action") or result.get("target_name"),
                controller.alive_players
            )
        if target:
            votes[target] = votes.get(target, 0) + 1
            reason = result.get("reason", "")
            print(f"  [VOTE] {name} -> {target}" + (f" ({reason[:40]})" if reason else ""))

    if not votes:
        print("  No votes cast, nobody eliminated")
        return None

    max_votes = max(votes.values())
    eliminated = [n for n, v in votes.items() if v == max_votes]

    if len(eliminated) == 1:
        target = eliminated[0]
        controller.vote_eliminate(target)
        role = controller.get_player_role(target)
        print(f"\n  ELIMINATED: {target} ({max_votes} votes, was {role})")
        return target
    else:
        print(f"\n  TIE: {_format_players(eliminated)} ({max_votes} votes each), nobody eliminated")
        return None
