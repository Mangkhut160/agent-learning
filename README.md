# agent-learning · 智能体学习与实践

> 从零到一的智能体（Agent）学习仓库。包含 **3 大经典范式**（ReAct / Plan-and-Solve / Reflection）的**手写实现**、**3 个主流多智能体框架**（AgentScope / AutoGen / LangGraph）的实战 demo，以及**自研 HelloAgents 框架**和详尽的学习笔记。

---

## 📌 项目简介

本仓库系统性地整理了我对**大模型智能体**的学习与实践。覆盖三个层次：

1. **3 种经典范式**——手写实现，理解 LLM Agent 的本质
2. **3 个主流框架**——生产级多智能体协作的工程化方案
3. **1 个自研框架**——HelloAgents：从零搭建轻量级 Agent 框架（pip installable）
4. **3 份学习笔记**——从概念到深度分析到框架对比的完整学习路径

> 🎯 配套教材：[Hello Agents（你好，智能体）](https://github.com/jjyaoao/HelloAgents) — Datawhale 开源教程

---

## 🧩 子项目一览（8 个）

| 子项目 | 框架 | 范式 / 任务 | 大小 |
|---|---|---|---|
| `react_agent_project/` | 纯 Python + OpenAI SDK | **ReAct**（Reason+Act 循环，正则解析） | ~25 KB |
| `plan_solve_agent_project/` | 纯 Python + OpenAI SDK | **Plan-and-Solve**（先规划后执行，ast 解析） | ~23 KB |
| `reflection_agent_project/` | 纯 Python + OpenAI SDK | **Reflection**（执行-反思-改进，3 prompt 角色） | ~23 KB |
| `autogen_project/` | **AutoGen 0.7+** | 多智能体软件开发团队（PM/工程师/评审/用户代理） | ~23 KB |
| `agentscope_project/` | **AgentScope 1.0+** | 三国狼人杀（6 角色 MsgHub 异步消息） | ~95 KB |
| `langgraph_project/` | **LangGraph 0.2+** | 状态机 + 三步搜索问答助手 | ~23 KB |
| `hello-agents/` | **自研 HelloAgents** | 4 范式一体的 pip installable 框架 | ~29 KB |
| `docs/chapter6_study_notes.md` | — | 第 6 章多智能体框架对比学习笔记 | 80 KB |

---

## 📁 完整目录结构

```
agent-learning/
├── README.md                                              ← 本文件
├── react_agent_project/                                   # ReAct 范式手写实现
│   ├── main.py + README.md
│   ├── src/
│   │   ├── agent.py                  # ReActAgent 主循环（正则解析 Thought/Action）
│   │   ├── tool_executor.py          # 工具注册与执行
│   │   ├── tools.py                  # Search (SerpApi) + Calculator
│   │   └── llm_client.py             # HelloAgentsLLM 风格客户端
│   ├── .env / .env.example           # ⚠️ .env 不会 push
│   └── requirements.txt
├── plan_solve_agent_project/                              # Plan-and-Solve 范式
│   ├── main.py + README.md
│   ├── src/
│   │   ├── planner.py                # 规划器（ast.literal_eval 解析 list 计划）
│   │   ├── executor.py               # 执行器（累积历史 → 下一步上下文）
│   │   ├── plan_solve_agent.py
│   │   └── llm_client.py
│   ├── .env / .env.example
│   └── requirements.txt
├── reflection_agent_project/                              # Reflection 范式
│   ├── main.py + README.md
│   ├── src/
│   │   ├── reflection_agent.py       # 3 prompt 角色：执行 / 评审 / 改进
│   │   ├── memory.py                 # 短期记忆 + 轨迹跟踪
│   │   └── llm_client.py             # 含 <think> 标签清理
│   ├── .env / .env.example
│   └── requirements.txt
├── autogen_project/                                       # AutoGen 多智能体团队
│   ├── main.py + README.md           # 4 角色 + RoundRobinGroupChat
│   ├── src/
│   │   ├── agents.py                 # ProductManager / Engineer / CodeReviewer / UserProxy
│   │   ├── model_client.py           # OpenAIChatCompletionClient 配置
│   │   └── team.py
│   ├── .env / .env.example
│   └── requirements.txt
├── agentscope_project/                                    # AgentScope 三国狼人杀
│   ├── main.py                        # 异步 6 角色游戏 (3 轮)
│   ├── src/
│   │   ├── game_controller.py        # GameController + GamePhase 枚举
│   │   ├── game_flow.py              # 18KB 各阶段流程
│   │   ├── agents.py                 # create_model + create_all_agents
│   │   ├── roles.py                  # 6 角色默认分配
│   │   └── models.py
│   ├── werewolf_game_log.txt          # 70KB 最近一次游戏日志
│   ├── .env / .env.example
│   └── requirements.txt
├── langgraph_project/                                     # LangGraph 状态机
│   ├── main.py + README.md            # 支持 interactive / demo1 / demo2 模式
│   ├── src/
│   │   ├── demo1_basic_workflow.py   # Planner↔Executor 循环图
│   │   └── demo2_search_assistant.py # 线性 3 步搜索问答图（含 Tavily）
│   ├── docs/langgraph_guide.md        # 20KB LangGraph 深度教程
│   ├── .env / .env.example
│   └── requirements.txt
├── hello-agents/                                         # 自研 HelloAgents 框架
│   ├── README.md                      # 451 行完备文档
│   ├── setup.py                       # pip install -e .
│   ├── .env / .env.example
│   ├── hello_agents/                  # 框架源码（pip 包）
│   │   ├── core/                      # agent / llm / message / config / exceptions
│   │   ├── agents/                    # SimpleAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent
│   │   └── tools/                     # base / registry / builtin (Calculator, Search)
│   └── examples/
│       ├── quickstart.py
│       └── agent_comparison.py
├── docs/
│   └── chapter6_study_notes.md        # 第 6 章多智能体框架对比笔记
├── 智能体学习笔记.md                                       # 13KB 入门笔记（范式 + Function Calling）
├── 智能体范式深度分析与实践.md                              # 51KB 深度分析（7 章节）
└── werewolf_game_log.txt                # 143KB 累计狼人杀游戏日志
```

---

## 🧠 3 大经典范式对比

| 维度 | **ReAct** | **Plan-and-Solve** | **Reflection** |
|---|---|---|---|
| 思考粒度 | 步级（每步一思） | 任务级（先全局规划） | 任务级 + 迭代 |
| 行动粒度 | 细（一步一动作） | 粗（多步连续执行） | 中（执行 → 整体反思） |
| 信息增量 | 实时（适合搜索） | 静态（plan 不变） | 累积（带历史反思） |
| 灵活性 | 高 | 中 | 高 |
| 核心依赖 | Thought/Action 正则解析 | Python list 字面量 | 3 prompt 角色 + 终止判断 |
| 适用场景 | 实时信息查询 | 多步逻辑推理 | 代码优化 / 写作迭代 |
| 解析脆弱性 | 高（中文冒号 / 嵌套括号） | 中 | 低 |

**P+R³ 混合架构**（生产级推荐）：Planner → Executor(每步 ReAct) → Reflector → Refiner

详见 `智能体范式深度分析与实践.md` 第 7 节决策树。

---

## 🏗️ 4 大框架对比

| 框架 | 协调范式 | 优势 | 痛点 |
|---|---|---|---|
| **AutoGen 0.7+** | 角色 + GroupChat | 易上手，生态成熟 | RoundRobinGroupChat 不能 fallback |
| **AgentScope 1.0+** | MsgHub 异步消息 | 高并发，适合游戏/对话 | 概念略多 |
| **LangGraph 0.2+** | 显式状态图 | 流程可控，支持 checkpoint | 写起来比 Agent 重 |
| **HelloAgents** | 4 范式 + 工具注册 | 教学友好，pip 可装 | 生产级需自己加固 |

详见 `docs/chapter6_study_notes.md`。

---

## 🚀 快速开始

### 1. 环境要求

- Python ≥ 3.10（`hello-agents` 要求 3.10+，其他子项目 3.8+ 即可）
- OpenAI 兼容的 API key（DeepSeek / 通义千问 / Ollama / MiniMax 等均可）
- 部分子项目需要：
  - **Tavily API key**（`langgraph_project` demo2）
  - **SerpApi API key**（`react_agent_project` 的搜索功能）

### 2. 克隆 & 配置

```bash
git clone https://github.com/Mangkhut160/agent-learning.git
cd agent-learning

# 复制 .env.example 为 .env 并填入 API key
cp .env.example .env
# 各子项目独立的 .env（每个都需同样操作）
cd react_agent_project && cp .env.example .env
cd ../plan_solve_agent_project && cp .env.example .env
# ... 以此类推
```

### 3. 安装依赖 & 运行（以 ReAct 为例）

```bash
cd react_agent_project
pip install -r requirements.txt
# 编辑 .env 填入 OPENAI_API_KEY / OPENAI_BASE_URL
python main.py
# 或交互模式
python main.py --interactive
```

### 4. 体验三国狼人杀

```bash
cd agentscope_project
pip install -r requirements.txt
python main.py
# 全自动跑 3 局，输出到 werewolf_game_log.txt
```

### 5. 安装 HelloAgents 框架

```bash
cd hello-agents
pip install -e .

# 或直接装发布版
# pip install hello-agents==0.1.1

python examples/quickstart.py          # 4 范式一网打尽
python examples/agent_comparison.py   # 范式对比
```

---

## 📚 学习路径

按这个顺序读笔记 + 跑代码，2 周可上手 Agent 开发：

| 阶段 | 阅读 | 跑代码 | 时长 |
|---|---|---|---|
| 1️⃣ 入门概念 | `智能体学习笔记.md`（13KB） | `react_agent_project` | 1 天 |
| 2️⃣ 范式深入 | `智能体范式深度分析与实践.md`（51KB，7 章） | `plan_solve_agent_project` + `reflection_agent_project` | 3 天 |
| 3️⃣ 框架了解 | `docs/chapter6_study_notes.md`（80KB，5 节） | `autogen_project` + `agentscope_project` + `langgraph_project` | 5 天 |
| 4️⃣ 框架自研 | `hello-agents/README.md`（451 行） | `hello-agents/examples/` | 2 天 |
| 5️⃣ 综合实战 | 把 P+R³ 混合架构落到自己项目 | — | 1 周+ |

---

## 🛠️ 各子项目详情

### 1. ReAct 范式（`react_agent_project/`）
- **核心循环**：Thought → Action → Observation → Thought...
- **解析方式**：正则 `(\w+)\[(.*)\]` 抽取 Action 名称 + 参数
- **支持工具**：SerpApi 搜索 + Calculator
- **示例问题**："华为最新手机是什么型号？"

### 2. Plan-and-Solve 范式（`plan_solve_agent_project/`）
- **两阶段**：规划（生成 Python list 字面量计划）+ 执行（按部就班）
- **解析方式**：`ast.literal_eval` 解析 plan
- **示例问题**：4 人排名 / 复利计算 / 图书馆系统设计 / 农场动物比例

### 3. Reflection 范式（`reflection_agent_project/`）
- **3 prompt 角色**：执行（程序员）/ 评审（reviewer）/ 改进（程序员）
- **终止条件**：评审说"无需改进"或 `max_iterations` 达到
- **示例问题**："找出 1~n 所有素数"——从试除法 O(n√n) 优化到埃氏筛 O(n log log n)

### 4. AutoGen 多智能体（`autogen_project/`）
- **4 角色协作**：ProductManager / Engineer / CodeReviewer / UserProxy
- **协调机制**：`RoundRobinGroupChat`，`TextMentionTermination("TERMINATE")`，`max_turns=20`
- **任务**：协作开发一个 Streamlit 比特币价格展示应用

### 5. AgentScope 三国狼人杀（`agentscope_project/`）
- **6 角色**：曹操（狼）、周瑜（狼）、诸葛亮（预言家）、张飞（女巫）、司马懿（守卫）、赵云（村民）
- **3 轮**完整游戏：守卫 → 狼人 → 预言家 → 女巫 → 黎明 → 讨论 → 投票
- **异步消息**：`asyncio.run` + `MsgHub`
- **完整日志**：`werewolf_game_log.txt`（70KB 最新 / 143KB 累计）

### 6. LangGraph 状态机（`langgraph_project/`）
- **Demo 1**：Planner↔Executor 循环图（带条件边）
- **Demo 2**：线性 3 步搜索问答（understand → search Tavily → answer）
- **运行**：`python main.py demo1` / `demo2` / `interactive "query"`

### 7. HelloAgents 自研框架（`hello-agents/`）
- **统一接口**：`HelloAgentsLLM`（自动识别 OpenAI/ModelScope/智谱/VLLM/Ollama）
- **4 范式一包**：SimpleAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent
- **工具系统**：`ToolRegistry` + 内置 Calculator/Search
- **可装可改**：`pip install -e .` 或 `pip install hello-agents==0.1.1`
- **许可**：MIT

---

## 🔑 关键设计要点

1. **3 种范式共享 `HelloAgentsLLM` 客户端**（`src/llm_client.py`），对比学习时只要换 main.py 即可
2. **正则 vs Function Calling**：早期手写 ReAct 依赖正则解析 `(\w+)\[(.*)\]`，易受 LLM 输出格式漂移影响；后期迁移到 OpenAI Function Calling（结构化保证）
3. **P+R³ 混合**：对真实业务（旅行预订 / 客服 / 代码生成 / 实验设计）通常是生产级答案
4. **异步消息 vs 函数调用**：AgentScope 的 MsgHub 在多角色游戏中比传统 blocking 调用更适合（避免狼人投票的竞态条件）
5. **状态图 vs GroupChat**：LangGraph 的显式图对**确定性流程**更友好，AutoGen GroupChat 对**开放对话**更友好

---

## 🛠️ 已知问题 & 后续

- 所有子项目都需要 `.env` 包含 API key 才能跑（仓库已排除）
- `agentscope_project` 的三国狼人杀需要 6 个角色的 API key（成本较高，可用便宜模型）
- 部分 `.env.example` 里写的是 `OPENAI_BASE_URL=...`，可以指向任何 OpenAI 兼容服务（DeepSeek / 通义 / Ollama）
- `react_agent_project` 的中文冒号 / 嵌套括号解析在某些 LLM 上失败，迁移到 Function Calling 后已解决
- `reflection_agent_project` 的 `max_iterations=2` 默认值偏小，复杂任务可调到 3-5

---

## 🙏 致谢

- [Datawhale Hello Agents](https://github.com/jjyaoao/HelloAgents) — 配套教材
- [AutoGen (Microsoft)](https://github.com/microsoft/autogen)
- [AgentScope (阿里)](https://github.com/modelscope/agentscope)
- [LangGraph (LangChain)](https://github.com/langchain-ai/langgraph)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

## 📜 License

- `hello-agents/` 子项目：**MIT License**（来自 [jjyaoao/HelloAgents](https://github.com/jjyaoao/HelloAgents)）
- 其他代码与笔记：仅用于学习与研究
