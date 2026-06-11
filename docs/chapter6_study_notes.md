# Chapter 6 深度学习笔记：多智能体框架综合实践

## 目录
1. [AutoGen 扩展实践](#1-autogen-扩展实践)
2. [AgentScope 深入分析](#2-agentscope-深入分析)
3. [CAMEL 冲突解决与多智能体扩展](#3-camel-冲突解决与多智能体扩展)
4. [LangGraph 流程建模与循环机制](#4-langgraph-流程建模与循环机制)
5. [框架选型决策](#5-框架选型决策)

---

## 1. AutoGen 扩展实践

### 1.1 动态回退机制设计

#### 问题分析

当前团队使用 `RoundRobinGroupChat`（轮询群聊）模式，发言顺序固定：
```
产品经理 → 工程师 → 代码审查员 → 用户代理 → (循环)
```

**问题场景**：如果代码审查员发现代码需要重大修改，如何让流程回退到工程师重新开发？

#### 设计方案：带状态的条件路由团队

```python
"""
动态回退团队实现
支持代码审查失败后自动回退到工程师重新开发
"""

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, StopMessageTermination
from enum import Enum
from typing import Literal


class TaskState(Enum):
    """任务流转状态"""
    REQUIREMENTS = "requirements"      # 需求分析阶段
    IMPLEMENTATION = "implementation"  # 开发阶段
    REVIEW = "review"                  # 审查阶段
    REVISION = "revision"               # 修订阶段（回退状态）
    ACCEPTED = "accepted"              # 验收通过
    REJECTED = "rejected"              # 最终拒绝


class DynamicFallbackTeam:
    """支持动态回退的智能体团队"""

    def __init__(self, participants: list):
        self.state = TaskState.REQUIREMENTS
        self.participants = participants
        self.revision_count = 0
        self.max_revisions = 3

        # 创建带状态感知的终止条件
        self.termination = TextMentionTermination("TERMINATE")

    def get_next_speaker(self, last_speaker: str) -> str:
        """
        根据当前状态动态决定下一个发言者

        状态机规则：
        - REQUIREMENTS → IMPLEMENTATION（产品经理完成）
        - IMPLEMENTATION → REVIEW（工程师完成）
        - REVIEW → REVISION（审查不通过）| ACCEPTED（审查通过）
        - REVISION → REVIEW（修订完成，重新审查）
        """
        speaker_order = ["ProductManager", "Engineer", "CodeReviewer", "UserProxy"]

        if last_speaker == "ProductManager":
            return "Engineer"

        if last_speaker == "Engineer":
            return "CodeReviewer"

        if last_speaker == "CodeReviewer":
            # 状态转换：审查结果决定下一步
            return self._handle_review_result()

        if last_speaker == "UserProxy":
            return "TERMINATE"

        return speaker_order[0]

    def _handle_review_result(self) -> str:
        """处理审查结果，决定是否回退"""
        # 检查是否需要回退（由审查员设置）
        if self.state == TaskState.REVISION:
            self.revision_count += 1
            if self.revision_count >= self.max_revisions:
                self.state = TaskState.REJECTED
                return "UserProxy"  # 最终拒绝
            return "Engineer"  # 回退到工程师

        # 默认流程：审查通过
        self.state = TaskState.ACCEPTED
        return "UserProxy"

    def transition_to(self, new_state: TaskState):
        """显式状态转换（由智能体通过消息触发）"""
        valid_transitions = {
            TaskState.REQUIREMENTS: [TaskState.IMPLEMENTATION],
            TaskState.IMPLEMENTATION: [TaskState.REVIEW],
            TaskState.REVIEW: [TaskState.REVISION, TaskState.ACCEPTED],
            TaskState.REVISION: [TaskState.REVIEW],
            TaskState.ACCEPTED: [TaskState.REJECTED],
            TaskState.REJECTED: []
        }

        if new_state in valid_transitions.get(self.state, []):
            self.state = new_state
            print(f"状态转换: {self.state.value} → {new_state.value}")
            return True
        return False


def create_dynamic_team(model_client):
    """
    创建支持动态回退的团队
    """

    # 产品经理
    pm = AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message="""你是产品经理。完成需求分析后说"需求确认，转工程师"。

当检测到审查员的反馈包含"需要重大修改"时，更新任务状态。"""
    )

    # 工程师
    engineer = AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message="""你是工程师。完成代码后说"代码完成，待审查"。

当收到审查反馈要求修改时，重新实现并说"代码已修订"."""
    )

    # 代码审查员
    reviewer = AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message="""你是代码审查员。审查后：

- 如果代码合格：说"代码审查通过"
- 如果需要修改：说"需要重大修改，请工程师重新实现"并设置状态为REVISION

审查维度：功能正确性、代码质量、安全性。"""
    )

    # 用户代理
    user_proxy = UserProxyAgent(
        name="UserProxy",
        input_func=lambda _: "TERMINATE"
    )

    participants = [pm, engineer, reviewer, user_proxy]

    # 创建团队
    team = RoundRobinGroupChat(
        participants=participants,
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=30
    )

    return team
```

#### 流程图

```
                    ┌─────────────────┐
                    │   REQUIREMENTS   │
                    │    (产品经理)    │
                    └────────┬────────┘
                             │ 需求确认
                             ▼
                    ┌─────────────────┐
                    │ IMPLEMENTATION  │
                    │    (工程师)     │
                    └────────┬────────┘
                             │ 代码完成
                             ▼
                    ┌─────────────────┐
              ┌─────│     REVIEW      │─────┐
              │     │  (代码审查员)   │     │
              │     └─────────────────┘     │
              │            │                │
        需要修改          通过               │
              │            │                │
              ▼            ▼                │
    ┌─────────────────┐                   │
    │    REVISION     │                   │
    │  (修订阶段)     │                   │
    └────────┬────────┘                   │
             │ 修订完成                   │
             └──────────────┬─────────────┘
                            │
                      达到最大修订次数?
                      /              \
                    否                是
                    /                  \
                   ▼                    ▼
            ┌──────────┐          ┌──────────┐
            │  REVIEW  │          │ REJECTED │
            └──────────┘          └──────────┘
                                     (终止)
```

### 1.2 测试工程师角色设计

#### 新角色：Quality Assurance Engineer

```python
def create_qa_engineer(model_client):
    """
    创建测试工程师智能体

    职责：
    1. 编写自动化测试用例
    2. 执行单元测试和集成测试
    3. 生成测试报告
    4. 验证修复的有效性
    """
    system_message = """你是资深测试工程师（QA），专注于自动化测试和质量保证。

你的核心技能：
1. 单元测试：pytest, unittest
2. 集成测试：API 测试、数据库测试
3. 端到端测试：Selenium, Playwright
4. 性能测试：locust, JMeter

审查流程：
1. 阅读代码，理解功能需求
2. 识别需要测试的关键路径
3. 编写测试用例（包含边界条件）
4. 执行测试并记录结果
5. 报告缺陷并提出修复建议

输出格式（JSON）：
{
    "test_cases": [
        {
            "name": "测试用例名称",
            "input": "输入参数",
            "expected": "期望结果",
            "type": "unit|integration|e2e"
        }
    ],
    "test_results": {
        "passed": 数量,
        "failed": 数量,
        "coverage": "覆盖率"
    },
    "defects": [
        {
            "severity": "high|medium|low",
            "description": "缺陷描述",
            "location": "代码位置"
        }
    ]
}

完成测试后说"测试完成，请产品经理验收"。"""

    return AssistantAgent(
        name="QAEngineer",
        model_client=model_client,
        system_message=system_message,
    )
```

#### 扩展后的团队流程

```
产品经理 → 工程师 → 代码审查员 → QA工程师 → 用户代理
    ↑                                      │
    └───────────── (需要修改) ──────────────┘
```

### 1.3 对话质量监控机制

#### 设计目标
1. **偏离检测**：检测对话是否偏离主题
2. **循环检测**：检测是否陷入重复循环
3. **质量评分**：评估当前回复的质量
4. **干预机制**：在异常时进行干预

```python
"""
对话质量监控器
"""

from collections import deque
import re


class ConversationMonitor:
    """对话质量监控器"""

    def __init__(self, max_turns_same_topic=5, max_repeat_ratio=0.5):
        # 最近 N 条消息
        self.recent_messages = deque(maxlen=20)
        # 相同回复计数器
        self.repeat_counter = {}
        # 主题追踪
        self.current_topic = None
        self.topic_turns = 0

        # 阈值配置
        self.max_turns_same_topic = max_turns_same_topic
        self.max_repeat_ratio = max_repeat_ratio

    def add_message(self, speaker: str, content: str):
        """添加消息到监控队列"""
        self.recent_messages.append({
            "speaker": speaker,
            "content": content,
            "content_hash": hash(content)
        })

        # 更新重复计数
        content_hash = hash(content)
        self.repeat_counter[content_hash] = self.repeat_counter.get(content_hash, 0) + 1

    def check_quality(self) -> dict:
        """
        检查对话质量，返回监控结果

        Returns:
            dict: {
                "status": "normal" | "warning" | "critical",
                "issues": ["问题列表"],
                "recommendations": ["建议列表"]
            }
        """
        issues = []
        recommendations = []

        # 检查1：重复内容
        repeat_issues = self._check_repetition()
        issues.extend(repeat_issues)

        # 检查2：话题漂移
        topic_issues = self._check_topic_drift()
        issues.extend(topic_issues)

        # 检查3：回复长度异常
        length_issues = self._check_response_length()
        issues.extend(length_issues)

        # 生成建议
        if issues:
            recommendations = self._generate_recommendations(issues)

        # 确定状态
        if len(issues) >= 3:
            status = "critical"
        elif issues:
            status = "warning"
        else:
            status = "normal"

        return {
            "status": status,
            "issues": issues,
            "recommendations": recommendations
        }

    def _check_repetition(self) -> list:
        """检查重复内容"""
        issues = []
        total = len(self.recent_messages)

        if total < 3:
            return issues

        # 计算重复率
        hash_counts = {}
        for msg in self.recent_messages:
            h = msg["content_hash"]
            hash_counts[h] = hash_counts.get(h, 0) + 1

        max_repeat = max(hash_counts.values()) if hash_counts else 1
        repeat_ratio = max_repeat / total

        if repeat_ratio > self.max_repeat_ratio:
            issues.append(f"重复率过高: {repeat_ratio:.1%}")

        # 检查连续相同回复
        consecutive = 1
        for i in range(1, len(self.recent_messages)):
            if self.recent_messages[i]["content_hash"] == self.recent_messages[i-1]["content_hash"]:
                consecutive += 1
                if consecutive >= 3:
                    issues.append(f"检测到连续重复回复 ({consecutive}次)")
                    break
            else:
                consecutive = 1

        return issues

    def _check_topic_drift(self) -> list:
        """检查话题漂移"""
        issues = []

        if self.topic_turns > self.max_turns_same_topic:
            issues.append(f"话题持续时间过长 ({self.topic_turns}轮)")

        return issues

    def _check_response_length(self) -> list:
        """检查回复长度"""
        issues = []

        if len(self.recent_messages) < 2:
            return issues

        # 检查过短的回复
        for msg in list(self.recent_messages)[-5:]:
            content = msg["content"]
            if len(content) < 10:
                issues.append(f"回复过短: '{content[:20]}...'")

        return issues

    def _generate_recommendations(self, issues: list) -> list:
        """生成干预建议"""
        recommendations = []

        for issue in issues:
            if "重复率" in issue:
                recommendations.append("建议：跳过当前讨论，直接进入下一阶段")
            elif "话题持续" in issue:
                recommendations.append("建议：重置话题，回到主线任务")
            elif "回复过短" in issue:
                recommendations.append("建议：要求智能体更详细地回答")

        return recommendations

    def should_intervene(self) -> bool:
        """判断是否需要干预"""
        quality = self.check_quality()
        return quality["status"] == "critical"


# 干预策略
def intervene(team, reason: str):
    """执行干预"""
    print(f"⚠️ 干预触发: {reason}")
    # 可以选择：
    # 1. 发送系统消息提醒
    # 2. 强制转换到特定智能体
    # 3. 重置对话上下文
    # 4. 终止当前流程
    return reason
```

---

## 2. AgentScope 深入分析

### 2.1 消息驱动架构 vs 传统函数调用

#### 传统函数调用模式

```python
# 传统方式：直接函数调用
def game_flow():
    guard_result = run_guard_phase()      # 阻塞等待
    wolf_result = run_werewolf_phase()    # 阻塞等待
    seer_result = run_seer_phase()        # 阻塞等待
    witch_result = run_witch_phase()      # 阻塞等待
    dawn_result = run_dawn_announcement() # 阻塞等待
```

**特点**：
- 同步执行，严格顺序
- 调用方需要知道被调用方的完整接口
- 耦合度高，修改困难
- 难以表达复杂的并发交互

#### 消息驱动架构（MsgHub）

```
┌──────────────────────────────────────────────────────────────┐
│                        MsgHub (消息中心)                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐    │
│  │曹操 │────▶│周瑜 │────▶│诸葛亮│────▶│张飞 │────▶│司马 │    │
│  │(狼) │     │(狼) │     │(预言)│     │(女巫)│     │(守卫)│    │
│  └─────┘     └─────┘     └─────┘     └─────┘     └─────┘    │
│                                                               │
│  所有消息通过 MsgHub 路由                                     │
│  每个智能体独立运行，通过消息通信                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**消息驱动优势**：

| 特性 | 传统函数调用 | 消息驱动架构 |
|------|-------------|-------------|
| **耦合度** | 高（直接依赖） | 低（通过消息解耦） |
| **扩展性** | 差（修改困难） | 好（易于添加新智能体） |
| **并发性** | 差（顺序执行） | 好（消息异步处理） |
| **可调试性** | 简单（调用栈清晰） | 复杂（需要消息追踪） |
| **一致性** | 强（直接调用） | 需要额外机制保证 |
| **适用场景** | 简单流程 | 复杂交互、多智能体协同 |

#### 消息驱动特别有价值的场景

1. **实时游戏**：狼人杀、扑克等需要多轮交互的游戏
2. **分布式系统**：智能体部署在不同服务器上
3. **事件驱动应用**：触发-响应模式
4. **多智能体协作**：需要广播、组播场景
5. **松耦合系统**：模块间需要低依赖通信

### 2.2 猎人角色设计

#### 结构化输出模型

```python
"""
猎人角色结构化模型
"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class HunterActionModel(BaseModel):
    """猎人行动选择

    猎人在死亡时可以发动技能，带走一名玩家。

    死亡场景：
    - 白天被投票淘汰
    - 夜间被狼人击杀
    - 被女巫毒杀

    技能规则：
    - 只有猎人存活时才能发动
    - 猎人死于女巫毒药时不能发动技能
    - 选择带走目标时不能带走自己
    """

    # 行动决策
    action: Literal["shoot", "pass", "cannot_act"] = Field(
        description="猎人行动：shoot=开枪带走目标, pass=不开枪, cannot_act=无法行动（死亡时非猎人杀）"
    )

    # 技能状态
    skill_available: bool = Field(
        description="技能是否可用（死于女巫毒药时为false）"
    )

    # 射击目标（使用技能时必填）
    target_name: Optional[str] = Field(
        default=None,
        description="要带走的目标玩家姓名"
    )

    # 推理过程
    reasoning: str = Field(
        description="你的决策推理"
    )

    # 死亡原因（由系统填入）
    death_cause: Optional[str] = Field(
        default=None,
        description="死因：voted=投票, wolf_kill=狼人杀, witch_poison=女巫毒"
    )

    @model_validator(mode="after")
    def validate_action(self):
        """验证行动的有效性"""
        # 规则1：无法行动时不能射击
        if self.action == "shoot" and not self.skill_available:
            raise ValueError("技能不可用，无法开枪")

        # 规则2：开枪时必须指定目标
        if self.action == "shoot" and not self.target_name:
            raise ValueError("开枪时必须指定目标")

        # 规则3：不能带走自己
        if self.action == "shoot" and self.target_name == "自己":
            raise ValueError("不能带走自己")

        return self


class HunterNightResult(BaseModel):
    """猎人夜间结果"""

    hunter_name: str = Field(description="猎人姓名")
    actioned: bool = Field(description="是否发动了技能")
    target: Optional[str] = Field(default=None, description="带走的目标")
    success: bool = Field(description="是否成功")
    blocked_reason: Optional[str] = Field(default=None, description="被阻止的原因")


class HunterDayResult(BaseModel):
    """猎人白天投票结果"""

    hunter_name: str = Field(description="猎人姓名")
    voted_out: bool = Field(description="是否被投票淘汰")
    death_cause: str = Field(description="死亡原因")
    skill_triggered: bool = Field(description="是否发动技能")
    target: Optional[str] = Field(default=None, description="带走的目标")
    result: str = Field(description="结果描述：eliminated_only=仅淘汰, hunter_eliminated=猎人也淘汰")
```

#### 猎人在游戏流程中的集成

```python
"""
猎人角色集成到游戏控制器
"""

class HunterExtension:
    """猎人扩展模块"""

    def __init__(self):
        self.hunter_role = "猎人"
        self.hunter_name: Optional[str] = None
        self.hunter_can_shoot = True  # 是否可以开枪

    def register_hunter(self, name: str):
        """注册猎人"""
        self.hunter_name = name

    def on_death(self, dead_player: str, cause: str) -> tuple[bool, Optional[str]]:
        """
        猎人死亡处理

        Args:
            dead_player: 死亡的玩家
            cause: 死亡原因 (voted, wolf_kill, witch_poison)

        Returns:
            (技能是否发动, 带走的目标或None)
        """
        if dead_player != self.hunter_name:
            return False, None

        # 猎人死于女巫毒药，不能发动技能
        if cause == "witch_poison":
            self.hunter_can_shoot = False
            print(f"[猎人] {self.hunter_name} 死于女巫毒药，无法发动技能")
            return False, None

        # 其他死因，可以发动技能
        self.hunter_can_shoot = True
        return True, None  # 实际目标由猎人选择

    def resolve_shoot(self, shooter: str, target: str) -> str:
        """
        结算猎人技能

        Args:
            shooter: 猎人姓名
            target: 带走的目标

        Returns:
            结算结果描述
        """
        if not self.hunter_can_shoot:
            return "技能已被阻止"

        if shooter != self.hunter_name:
            return "不是猎人"

        # 执行带走
        return f"{shooter} 发动猎人技能，带走了 {target}"
```

### 2.3 分布式部署挑战分析

#### 技术挑战

```
┌─────────────────────────────────────────────────────────────────┐
│                   分布式狼人杀部署架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Server 1 (曹操, 周瑜 - 狼人)        Server 2 (诸葛亮 - 预言家) │
│   ┌─────────────────────────┐         ┌─────────────────────────┐
│   │  ReActAgent            │         │  ReActAgent              │
│   │  - 消息队列 Consumer   │         │  - 消息队列 Consumer     │
│   └───────────┬───────────┘         └───────────┬─────────────┘
│               │                                     │
│               │         ┌─────────────────┐         │
│               └────────▶│   Redis Pub/Sub │◀────────┘
│                         │   (消息总线)     │
│               ┌────────▶│                 │◀────────┐
│               │         └─────────────────┘         │
│   ┌───────────┴───────────┐         ┌───────────────┴─────────┐
│   │  Server 3 (张飞-女巫) │         │  Server 4 (司马-守卫)  │
│   │  ReActAgent           │         │  ReActAgent             │
│   └───────────────────────┘         └─────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 挑战 1：消息顺序性

**问题**：不同服务器上的智能体可能不按预期顺序收到消息

```python
# 场景：狼人投票
# Server 1: 曹操 投了 周瑜
# Server 2: 周瑜 投了 赵云

# 由于网络延迟，可能出现：
# 周瑜比曹操先收到消息
# 导致投票结果不符合预期
```

**解决方案**：
```python
class OrderedMessageQueue:
    """有序消息队列"""

    def __init__(self):
        self.queue = []
        self.sequence = 0

    def publish(self, topic: str, message: dict, priority: int = 0):
        """发布消息（带序列号）"""
        self.sequence += 1
        envelope = {
            "seq": self.sequence,
            "topic": topic,
            "message": message,
            "priority": priority,
            "timestamp": time.time()
        }
        # 写入消息队列
        redis.lpush(f"queue:{topic}", json.dumps(envelope))

    def consume(self, topic: str, agent_id: str) -> Optional[dict]:
        """消费消息（保证顺序）"""
        # 使用 BRPOPLPUSH 原子操作
        result = redis.brpoplpush(
            f"queue:{topic}",
            f"processing:{agent_id}",
            timeout=5
        )
        return json.loads(result) if result else None
```

#### 挑战 2：状态一致性

**问题**：游戏状态（存活玩家、角色分配）需要全局一致

```python
# 问题场景
# 守卫在 Server 3 上守护了 诸葛亮
# 同时，狼人在 Server 1 上投票杀了 诸葛亮
# 如何保证这两个操作的一致性？

# 如果守护先生效：诸葛亮存活
# 如果击杀先结算：诸葛亮死亡
```

**解决方案**：分布式锁 + 事务

```python
class GameStateManager:
    """分布式游戏状态管理器"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.lock_timeout = 5

    def acquire_lock(self, resource: str) -> bool:
        """获取分布式锁"""
        return self.redis.set(
            f"lock:{resource}",
            "locked",
            nx=True,
            ex=self.lock_timeout
        )

    def execute_night_action(self, action: NightAction) -> bool:
        """原子执行夜间操作"""
        resource = "night_resolution"

        # 获取锁
        if not self.acquire_lock(resource):
            return False  # 忙碌中

        try:
            # 在锁内执行所有夜间操作的合并结算
            pipeline = self.redis.pipeline()

            # 1. 记录狼人投票
            pipeline.zadd("wolf_votes", {action.target: 1})

            # 2. 记录守卫守护
            if action.guard_target:
                pipeline.set("guard_target", action.guard_target)

            # 3. 记录女巫行动
            if action.witch_save:
                pipeline.set("witch_save", action.target)

            pipeline.execute()

            # 触发结算（清除锁后）
            self.redis.delete(f"lock:{resource}")
            self.trigger_resolution()

            return True

        except Exception as e:
            self.redis.delete(f"lock:{resource}")
            raise e
```

#### 挑战 3：实时性要求

**问题**：游戏需要毫秒级响应

| 操作 | 延迟要求 | 分布式挑战 |
|------|---------|-----------|
| 消息传递 | <100ms | 网络延迟、路由 |
| 状态同步 | <50ms | 锁竞争、复制 |
| UI 更新 | <200ms | 推送机制 |

**解决方案**：
- 使用 WebSocket 保持连接
- 消息压缩 + 批量处理
- 本地缓存 + 最终一致性

---

## 3. CAMEL 冲突解决与多智能体扩展

### 3.1 冲突解决机制设计

#### 问题场景

当两个智能体对任务是否完成产生分歧：

```
研究员智能体：认为文献综述已经充分，可以终止
作家智能体：认为需要补充更多案例素材，不想终止
```

#### 冲突解决机制

```python
"""
CAMEL 冲突解决模块
"""

from enum import Enum
from typing import Literal


class ConflictType(Enum):
    """冲突类型"""
    TERMINATION = "termination"      # 终止时机分歧
    CONTENT = "content"              # 内容方向分歧
    STYLE = "style"                  # 风格偏好分歧
    PRIORITY = "priority"            # 优先级分歧


class ConflictResolver:
    """冲突解决器"""

    def __init__(self, llm):
        self.llm = llm

    def detect_conflict(self, agent_a_opinion: str, agent_b_opinion: str) -> bool:
        """
        检测是否存在冲突

        Returns:
            True 表示存在冲突
        """
        # 冲突检测逻辑
        termination_keywords_a = ["完成", "可以结束", "终止"]
        termination_keywords_b = ["不够", "需要更多", "继续"]

        a_wants_terminate = any(kw in agent_a_opinion for kw in termination_keywords_a)
        b_wants_continue = any(kw in agent_b_opinion for kw in termination_keywords_b)

        return a_wants_terminate and b_wants_continue

    def resolve_termination_conflict(
        self,
        agent_a: str,  # 主张终止的智能体
        agent_b: str,  # 主张继续的智能体
        task_context: str,
        max_cycles: int = 3
    ) -> dict:
        """
        解决终止时机冲突

        策略：
        1. 引入中立仲裁者判断
        2. 设置明确的完成标准
        3. 限制最大协商轮次
        """

        prompt = f"""作为中立仲裁者，分析以下分歧并做出裁决：

主张终止的智能体（{agent_a}）观点：
{task_context}

主张继续的智能体（{agent_b}）观点：
请参考上方内容

请评估：
1. 任务是否已经达到了基本的完成标准？
2. 继续工作是否能显著提升质量？
3. 时间/资源成本是否值得？

给出裁决："可以终止" 或 "需要继续" 或 "折中方案"。

同时说明理由。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 解析裁决
        decision = self._parse_decision(response.content)

        return {
            "decision": decision,
            "reasoning": response.content,
            "remaining_cycles": max_cycles - 1
        }

    def _parse_decision(self, content: str) -> Literal["terminate", "continue", "compromise"]:
        """解析裁决结果"""
        content_lower = content.lower()

        if "终止" in content or "terminate" in content_lower:
            return "terminate"
        elif "继续" in content or "continue" in content_lower:
            return "continue"
        else:
            return "compromise"

    def apply_compromise(self, agent_a_opinion: str, agent_b_opinion: str) -> str:
        """
        产生折中方案

        例如：
        - A: 完成即可，B: 补充案例
        - 折中: 在当前基础上补充 2 个案例，不追求完美
        """
        prompt = f"""作为调解者，产生一个折中方案：

观点A：{agent_a_opinion}
观点B：{agent_b_opinion}

请提出一个双方都能接受的折中方案，具体说明：
1. 哪些部分可以简化
2. 哪些部分是必要的
3. 明确的终止条件
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

#### 冲突解决流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     冲突检测                                 │
│  检测到 Agent A 主张终止，Agent B 主张继续                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   协商阶段 1                                 │
│  仲裁者分析双方观点，判断是否确实存在冲突                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │      存在实质冲突?         │
         └────────────────────────────┘
            │                    │
           是                    否
            │                    │
            ▼                    ▼
┌───────────────────────┐   ┌────────────────────────┐
│     折中方案生成        │   │  接受当前状态，继续执行  │
│  调解者提出折中建议     │   └────────────────────────┘
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   协商阶段 2                                 │
│  双方是否接受折中方案？                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                        │
        是                       否
         │                        │
         ▼                        ▼
┌────────────────┐      ┌────────────────────────┐
│ 按折中方案执行  │      │ 引入第三方裁判/人工介入  │
│ 设定明确终止点 │      └────────────────────────┘
└────────────────┘
```

### 3.2 CAMEL Workforce vs AutoGen 群聊

#### CAMEL Workforce 架构

```python
"""
CAMEL Workforce 多智能体协作模块
"""

# WorkForce 示例架构
from camel.agents import RoleAssignmentAgent, TaskAgent
from camel.configs import WorkForceConfig


class WorkForce:
    """
    CAMEL WorkForce - 多智能体工作团队

    核心概念：
    - Workforce：整个工作团队
    - Agent：具体角色（RoleAssignmentAgent 分配角色）
    - Task：具体任务（TaskAgent 执行）

    与 AutoGen 的关键区别：
    - AutoGen：基于消息传递的群聊
    - CAMEL：基于任务分解的树形结构
    """

    def __init__(self, task: str):
        self.task = task
        self.agents = []
        self.task_graph = {}  # 任务依赖图

    def add_agent(self, role: str, agent_type: str):
        """添加智能体到团队"""
        pass

    def define_dependency(self, task_a: str, task_b: str):
        """定义任务依赖：A 完成后才能执行 B"""
        if task_a not in self.task_graph:
            self.task_graph[task_a] = []
        self.task_graph[task_a].append(task_b)
```

#### 对比分析

| 特性 | CAMEL WorkForce | AutoGen RoundRobinGroupChat |
|------|----------------|---------------------------|
| **架构** | 树形任务分解 | 扁平群聊 |
| **通信模式** | 任务导向，父子节点通信 | 广播式群聊 |
| **角色分配** | 动态角色分配 | 静态角色定义 |
| **依赖管理** | 显式任务依赖图 | 隐式顺序（通过消息） |
| **扩展性** | 好（树形扩展） | 中等（添加agent到列表） |
| **适用场景** | 复杂任务分解 | 对话式协作 |
| **调试难度** | 中等 | 中等 |

#### 架构图对比

```
CAMEL WorkForce（树形结构）：

                    ┌─────────────┐
                    │   Root      │
                    │  任务分解   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ 子任务1  │     │ 子任务2  │     │ 子任务3  │
    │ (研究员) │     │ (作家)   │     │ (编辑)   │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ 执行具体 │     │ 执行具体 │     │ 执行具体 │
    │   任务   │     │   任务   │     │   任务   │
    └──────────┘     └──────────┘     └──────────┘


AutoGen RoundRobin（扁平群聊）：

    ┌──────────┐
    │  Agent1  │◀────────────────────┐
    │ (研究员) │                     │
    └────┬─────┘                     │
         │ 轮询发言                  │
         ▼                           │
    ┌──────────┐                    │
    │  Agent2  │────────────────────┤
    │ (作家)   │                     │
    └────┬─────┘                    │
         │ 轮询发言                  │ 消息循环
         ▼                           │
    ┌──────────┐                    │
    │  Agent3  │────────────────────┘
    │ (编辑)   │
    └──────────┘
```

---

## 4. LangGraph 流程建模与循环机制

### 4.1 三步问答助手图结构

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph 状态机图                          │
│                  "理解-搜索-回答" 线性流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     ┌─────────────────────────────────────────────────────┐     │
│     │                    State: SearchState               │     │
│     ├─────────────────────────────────────────────────────┤     │
│     │  messages: Annotated[list, add_messages]           │     │
│     │  user_query: str           # 理解后的用户需求       │     │
│     │  search_query: str         # 优化后的搜索词         │     │
│     │  search_results: str       # 搜索返回结果           │     │
│     │  final_answer: str         # 最终答案               │     │
│     │  step: str                 # 当前步骤标记           │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                 │
│                         START                                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                             │
│                  │   understand    │  节点1                      │
│                  │   (理解节点)     │                             │
│                  │                 │                             │
│                  │ 输入: messages  │                             │
│                  │ 输出:          │                             │
│                  │   - user_query │                             │
│                  │   - search_query                             │
│                  │   - step="understood"                        │
│                  └────────┬────────┘                             │
│                           │                                      │
│                     边: understand → search                       │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │     search      │  节点2                      │
│                  │   (搜索节点)    │                             │
│                  │                 │                             │
│                  │ 输入: search_query│                            │
│                  │ 输出:           │                             │
│                  │   - search_results│                          │
│                  │   - step="searched" │                        │
│                  └────────┬────────┘                             │
│                           │                                      │
│                     边: search → answer                           │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │     answer      │  节点3                      │
│                  │   (回答节点)    │                             │
│                  │                 │                             │
│                  │ 输入:           │                             │
│                  │   - user_query  │                             │
│                  │   - search_results│                          │
│                  │ 输出:           │                             │
│                  │   - final_answer│                             │
│                  │   - step="completed"│                        │
│                  └────────┬────────┘                             │
│                           │                                      │
│                     边: answer → END                             │
│                           │                                      │
│                           ▼                                      │
│                          END                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 带"反思"节点的扩展流程

#### 设计思路

```
当生成的答案质量低时（过短、缺乏细节），
系统应该：
1. 评估答案质量
2. 如果质量低，重新搜索或重新生成
3. 最多循环 N 次，防止无限循环
```

#### 扩展后的图结构

```
┌─────────────────────────────────────────────────────────────────┐
│            扩展后的问答流程（含反思循环）                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         START                                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                             │
│                  │   understand    │                            │
│                  │   (理解节点)    │                            │
│                  └────────┬────────┘                             │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │     search      │                            │
│                  │   (搜索节点)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │     answer      │                            │
│                  │   (回答节点)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │    evaluate     │  ← 新增：质量评估节点       │
│                  │   (评估节点)    │                            │
│                  │                 │                            │
│                  │ 评估标准：       │                            │
│                  │ - 答案长度       │                            │
│                  │ - 信息完整性     │                            │
│                  │ - 来源可靠性     │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│            ┌──────────────┴──────────────┐                      │
│            │                             │                      │
│     质量低                            质量高                    │
│            │                             │                      │
│            ▼                             ▼                      │
│   ┌─────────────────┐           ┌──────────────┐                │
│   │  should_retry   │           │   answer     │                │
│   │ (条件边判断)    │           │   (输出答案) │                │
│   └────────┬────────┘           └──────────────┘                │
│            │                                                  │
│     ┌──────┴──────┐                                          │
│     │             │                                          │
│  重试<3次       重试>=3次                                     │
│     │             │                                          │
│     ▼             ▼                                          │
│ ┌────────┐    ┌────────┐                                     │
│ │ search │    │ answer │                                     │
│ │ (重搜索)│    │ (强制  │                                     │
│ │ 或重生成│    │  输出) │                                     │
│ └────────┘    └────────┘                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 代码实现

```python
"""
扩展版问答助手（带反思循环）
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class ExtendedSearchState(TypedDict):
    """扩展状态（含反思相关字段）"""
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str
    retry_count: int           # 重试次数
    quality_score: float       # 质量评分
    last_feedback: str         # 质量反馈


def evaluate_answer_quality(state: ExtendedSearchState) -> dict:
    """
    评估答案质量

    评估维度：
    1. 长度：是否过短（<100字）
    2. 完整性：是否包含关键信息
    3. 具体性：是否有具体细节
    """
    answer = state.get("final_answer", "")

    score = 0.0
    feedback = []

    # 评估1：长度
    if len(answer) < 100:
        feedback.append("答案过短")
    else:
        score += 0.3

    # 评估2：结构
    if any(marker in answer for marker in ["1.", "2.", "3.", "•", "-"]):
        score += 0.3
    else:
        feedback.append("缺乏结构化表达")

    # 评估3：具体性
    if len(answer) > 300:
        score += 0.4
    else:
        feedback.append("内容不够详细")

    return {
        "quality_score": score,
        "last_feedback": "; ".join(feedback) if feedback else "质量良好",
        "step": "evaluated"
    }


def should_retry(state: ExtendedSearchState) -> Literal["retry", "accept"]:
    """
    条件边：决定是否重试

    重试条件：
    1. 质量评分 < 0.5
    2. 且重试次数 < 3

    接受条件：
    1. 质量评分 >= 0.5
    2. 或重试次数 >= 3（防止无限循环）
    """
    quality = state.get("quality_score", 0)
    retry_count = state.get("retry_count", 0)

    if quality < 0.5 and retry_count < 3:
        return "retry"
    return "accept"


def retry_node(state: ExtendedSearchState) -> dict:
    """
    重试节点

    策略：
    1. 如果搜索结果不够好 → 优化搜索词重新搜索
    2. 如果答案不够好 → 使用更详细的提示重新生成
    """
    retry_count = state.get("retry_count", 0)
    quality = state.get("quality_score", 0)

    print(f"\n🔄 重试 #{retry_count + 1}，质量评分: {quality:.2f}")

    if quality < 0.3:
        # 搜索质量差，重新搜索
        return {
            "step": "retry_search",
            "retry_count": retry_count + 1,
            "messages": [AIMessage(content=f"重新搜索...")]
        }
    else:
        # 答案质量差但搜索还行，重新生成答案
        return {
            "step": "retry_answer",
            "retry_count": retry_count + 1,
            "messages": [AIMessage(content=f"重新生成答案...")]
        }


def create_extended_search_assistant():
    """创建带反思的问答助手"""
    workflow = StateGraph(ExtendedSearchState)

    # 节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    workflow.add_node("evaluate", evaluate_answer_quality)
    workflow.add_node("retry", retry_node)

    # 边
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", "evaluate")

    # 条件边
    workflow.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "retry",     # 重试
            "accept": END         # 接受答案
        }
    )

    # 重试后重新搜索或重新生成
    workflow.add_edge("retry", "search")

    return workflow.compile()
```

### 4.3 复杂循环应用设计

#### 应用场景 1：代码生成-测试-修复循环

```
┌─────────────────────────────────────────────────────────────────┐
│                代码生成-测试-修复 循环                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         START                                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │   requirements  │  理解需求                    │
│                  │   (需求分析)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │    generate     │  生成代码                    │
│                  │   (代码生成)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │      test       │  运行测试                    │
│                  │   (测试执行)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │   evaluate      │  评估测试结果                │
│                  │   (结果评估)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│            ┌──────────────┴──────────────┐                      │
│            │                             │                      │
│         测试失败                      测试通过                   │
│            │                             │                      │
│            ▼                             ▼                      │
│   ┌─────────────────┐           ┌──────────────┐              │
│   │   should_fix    │           │    output    │              │
│   │  (条件:循环?)   │           │   (输出结果) │              │
│   └────────┬────────┘           └──────────────┘              │
│            │                                                  │
│     ┌──────┴──────┐                                          │
│     │             │                                          │
│  修复<5次     修复>=5次                                       │
│     │             │                                          │
│     ▼             ▼                                          │
│  ┌──────┐    ┌──────────┐                                    │
│  │ fix  │    │ 失败报告 │                                    │
│  │ 修复 │    │ (放弃)   │                                    │
│  └──┬───┘    └──────────┘                                    │
│     │                                                         │
│     └─────────────────────┐                                   │
│                           │                                   │
│                           ▼                                   │
│                      generate                                 │
│                      (重新生成)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键节点说明**：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| requirements | 解析需求 | 用户需求描述 | 结构化需求文档 |
| generate | 生成代码 | 需求文档 | 代码文件 |
| test | 执行测试 | 代码文件 | 测试结果（通过/失败/错误详情） |
| evaluate | 评估结果 | 测试结果 | 评估报告 |
| fix | 修复问题 | 错误详情 | 修复后的代码 |

#### 应用场景 2：论文写作-审阅-修改循环

```
┌─────────────────────────────────────────────────────────────────┐
│                论文写作-审阅-修改 循环                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         START                                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │   outline       │  生成大纲                    │
│                  │   (大纲生成)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │     write      │  撰写章节                    │
│                  │   (写作节点)   │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │     review      │  审阅论文                    │
│                  │   (审阅节点)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                            │
│                  │  evaluate       │  评估审阅意见                │
│                  │  (评估节点)    │                            │
│                  └────────┬────────┘                            │
│                           │                                    │
│            ┌──────────────┴──────────────┐                      │
│            │                             │                      │
│         需要修改                      通过                       │
│            │                             │                      │
│            ▼                             ▼                      │
│   ┌─────────────────┐           ┌──────────────┐              │
│   │   should_revise │           │    finalize  │              │
│   │  (条件:继续?)   │           │   (定稿)     │              │
│   └────────┬────────┘           └──────────────┘              │
│            │                                                  │
│     ┌──────┴──────┐                                          │
│     │             │                                          │
│  修改<4次     修改>=4次                                       │
│     │             │                                          │
│     ▼             ▼                                          │
│  ┌────────┐    ┌──────────┐                                 │
│  │ revise │    │ 强制定稿 │                                 │
│  │ 修改   │    │ (防止循环 │                                 │
│  └────┬───┘    └──────────┘                                 │
│       │                                                        │
│       └────────────────────┐                                  │
│                            │                                  │
│                            ▼                                  │
│                       write                                   │
│                       (继续写下一节)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键节点说明**：

| 节点 | 功能 | 审阅标准 |
|------|------|---------|
| outline | 生成论文大纲 | 逻辑完整性、章节合理 |
| write | 撰写论文内容 | 语言流畅、论证充分 |
| review | 审阅论文 | 学术规范、逻辑严谨、引用规范 |
| evaluate | 评估审阅意见 | 判断是否需要修改 |
| revise | 执行修改 | 针对性修改问题 |

---

## 5. 框架选型决策

### 5.1 选型决策矩阵

| 维度 | AutoGen | AgentScope | CAMEL | LangGraph |
|------|---------|------------|-------|-----------|
| **架构范式** | 多智能体对话 | 多智能体消息 | 双智能体协作 | 状态机/图 |
| **循环支持** | 需手动实现 | 需手动实现 | 需手动实现 | **原生支持** |
| **扩展性** | 好 | 好 | 中 | 极好 |
| **学习曲线** | 低 | 低 | 中 | 中 |
| **调试难度** | 中 | 中 | 中 | 低 |
| **生产成熟度** | 高 | 中 | 中 | 高 |

### 5.2 应用场景分析

#### 应用 A：智能客服系统

**需求**：
- 高并发：每秒 1000+ 请求
- 低延迟：响应时间 <2秒
- 7×24 运行
- 水平扩展

**推荐框架**：**不借助框架从零开发**

**理由**：

1. **高并发需求**：
   - 框架（如 AutoGen、AgentScope）设计为单次对话流程
   - 不适合高并发、长连接场景
   - 需要专门的消息队列、会话管理

2. **延迟要求**：
   - 框架有额外的编排开销
   - 直接调用 LLM API 更高效

3. **水平扩展**：
   - 框架通常单实例运行
   - 需要自建负载均衡、会话保持

4. **推荐架构**：
```
                    ┌─────────────────────────────────────┐
                    │            负载均衡器               │
                    │         (Nginx/AWS ALB)            │
                    └─────────────┬─────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   API Server 1  │ │   API Server 2  │ │   API Server 3  │
    │   (FastAPI)     │ │   (FastAPI)     │ │   (FastAPI)     │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │       Message Queue        │
                    │     (Redis/RabbitMQ)       │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  Worker Pod 1   │ │  Worker Pod 2   │ │  Worker Pod N   │
    │  (对话处理)     │ │  (对话处理)     │ │  (对话处理)     │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**技术栈建议**：
- API 层：FastAPI + Uvicorn
- 消息队列：Redis Stream
- 会话存储：Redis + PostgreSQL
- LLM 调用：批量请求 + 缓存

---

#### 应用 B：科研论文辅助写作平台

**需求**：
- 两个智能体深度协作：研究员 + 写作助手
- 多轮深度讨论
- 自主推进任务
- 文献综述、实验设计、数据分析、论文撰写

**推荐框架**：**LangGraph**

**理由**：

1. **多轮迭代**：
   - 论文写作需要"写-审-改"循环
   - LangGraph 原生支持循环

2. **状态管理**：
   - 论文写作有明确的阶段性状态
   - LangGraph 状态机非常适合

3. **可追溯性**：
   - 论文写作过程需要版本控制
   - LangGraph 支持检查点（checkpointing）

4. **灵活性**：
   - 可以定义复杂的条件边
   - 例如：文献不够 → 继续搜索，结构不合理 → 重写

**架构设计**：

```python
class PaperWritingState(TypedDict):
    """论文写作状态"""
    phase: Literal["research", "outline", "write", "review", "revise", "finalize"]
    research_notes: str
    outline: str
    current_section: str
    written_content: str
    review_comments: str
    revision_count: int
    paper_draft: str


def create_paper_writing_workflow():
    """创建论文写作工作流"""

    workflow = StateGraph(PaperWritingState)

    # 节点
    workflow.add_node("research", research_node)      # 文献研究
    workflow.add_node("outline", outline_node)        # 生成大纲
    workflow.add_node("write", write_node)            # 撰写章节
    workflow.add_node("review", review_node)          # 审阅
    workflow.add_node("revise", revise_node)          # 修改
    workflow.add_node("finalize", finalize_node)     # 定稿

    # 边
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "write")

    # 条件边：审阅后决定是否修改
    workflow.add_conditional_edges(
        "review",
        should_revise,
        {
            "revise": "revise",
            "finalize": "finalize"
        }
    )

    # 修改后重新撰写
    workflow.add_edge("revise", "write")

    return workflow.compile()
```

---

#### 应用 C：金融风控审批系统

**需求**：
- 严格流程：资料审核 → 风险评估 → 额度计算 → 合规检查 → 人工复核 → 最终决策
- 明确判断标准和分支逻辑
- 流程可追溯、可审计

**推荐框架**：**LangGraph**

**理由**：

1. **流程清晰**：
   - 每个环节有明确的输入、输出、判断条件
   - 非常适合用状态机建模

2. **可追溯性**：
   - 每个节点记录操作日志
   - 便于审计和回溯

3. **分支逻辑**：
   - 合规检查可能有多种结果（通过/拒绝/需人工复核）
   - 条件边完美支持

4. **监管合规**：
   - 金融系统需要完整的审计日志
   - LangGraph 可以记录每个状态转换

**流程图**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    金融风控审批流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   START                                                          │
│     │                                                            │
│     ▼                                                            │
│ ┌─────────────┐                                                  │
│ │  资料审核   │  审核材料完整性、真实性                           │
│ └──────┬──────┘                                                  │
│        │                                                         │
│        ▼                                                         │
│ ┌─────────────────┐                                              │
│ │    风险评估     │  评估申请人信用、还款能力                      │
│ └──────┬──────────┘                                              │
│        │                                                         │
│        ▼                                                         │
│ ┌─────────────────┐                                              │
│ │    额度计算     │  基于风险评估计算额度                          │
│ └──────┬──────────┘                                              │
│        │                                                         │
│        ▼                                                         │
│ ┌─────────────────┐                                              │
│ │    合规检查     │  反洗钱、黑名单、信用记录                      │
│ └──────┬──────────┘                                              │
│        │                                                         │
│        ▼                                                         │
│   ┌────┴────┐                                                    │
│   │         │                                                   │
│ 通过     拒绝/需人工                                             │
│   │         │                                                   │
│   ▼         ▼                                                    │
│ ┌────────┐ ┌──────────────┐                                    │
│ │人工复核│ │  最终决策    │                                    │
│ └──┬────┘ └──────────────┘                                    │
│    │                                                           │
│    ▼                                                           │
│ ┌──────────────┐                                               │
│ │  最终决策    │                                               │
│ └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**代码设计**：

```python
class ApprovalState(TypedDict):
    """审批流程状态"""
    applicant_id: str
    application_data: dict

    # 各阶段结果
    document_verification: Literal["passed", "failed", "pending"]
    risk_assessment: dict  # 包含 score, factors
    credit_limit: float
    compliance_check: Literal["passed", "flagged", "rejected"]

    # 流程控制
    current_stage: str
    stage_history: list  # 记录每个阶段的操作
    final_decision: Literal["approved", "rejected", "manual_review"]


def create_approval_workflow():
    """创建风控审批工作流"""
    workflow = StateGraph(ApprovalState)

    # 节点
    workflow.add_node("document_verification", verify_documents)
    workflow.add_node("risk_assessment", assess_risk)
    workflow.add_node("credit_calculation", calculate_credit)
    workflow.add_node("compliance_check", check_compliance)
    workflow.add_node("manual_review", manual_review)
    workflow.add_node("final_decision", make_decision)

    # 边
    workflow.add_edge(START, "document_verification")
    workflow.add_edge("document_verification", "risk_assessment")
    workflow.add_edge("risk_assessment", "credit_calculation")
    workflow.add_edge("credit_calculation", "compliance_check")

    # 条件边
    workflow.add_conditional_edges(
        "compliance_check",
        compliance_decision,
        {
            "passed": "final_decision",
            "manual": "manual_review",
            "rejected": "final_decision"
        }
    )

    workflow.add_edge("manual_review", "final_decision")
    workflow.add_edge("final_decision", END)

    return workflow.compile()
```

### 5.3 选型决策总结

| 应用 | 推荐框架 | 核心原因 |
|------|---------|---------|
| **A: 智能客服** | 自建 | 高并发、低延迟、水平扩展需求 |
| **B: 论文写作** | LangGraph | 多轮迭代、状态管理、可追溯 |
| **C: 风控审批** | LangGraph | 流程清晰、分支逻辑、审计追溯 |

### 5.4 框架选择决策树

```
项目需求评估
│
├── 是否需要复杂循环？（代码生成-测试-修复、写作-审阅-修改）
│   └── 是 → LangGraph
│
├── 是否需要高并发、低延迟？
│   └── 是 → 自建框架
│
├── 是否是多智能体对话/群聊？
│   ├── 是 → AgentScope（游戏场景）
│   └── 否
│
├── 是否需要明确的工作流程/状态机？
│   └── 是 → LangGraph
│
└── 默认选择
    └── AutoGen（成熟、简单）
```

---

## 附录：关键概念速查

### A1. 状态机四要素

| 要素 | 说明 | 示例 |
|------|------|------|
| **状态** | 系统的当前状态 | REQUIREMENTS, IMPLEMENTATION |
| **事件** | 触发状态转换的事件 | "审查通过", "审查失败" |
| **动作** | 状态转换时执行的操作 | 发送通知、更新状态 |
| **转换** | 从一个状态到另一个状态的转移 | REQUIREMENTS → IMPLEMENTATION |

### A2. 消息驱动 vs 函数调用

| 维度 | 消息驱动 | 函数调用 |
|------|---------|---------|
| 时序 | 异步 | 同步 |
| 耦合 | 松耦合 | 紧耦合 |
| 扩展性 | 好 | 差 |
| 延迟 | 可能较高 | 低 |
| 一致性 | 需要额外机制 | 自然保证 |

### A3. 循环检测策略

1. **计数器法**：跟踪迭代次数
2. **内容哈希法**：检测重复内容
3. **时间窗口法**：限制在一定时间内
4. **质量阈值法**：低于阈值则终止