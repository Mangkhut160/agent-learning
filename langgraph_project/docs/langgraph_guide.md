# LangGraph 学习指南

## 目录
1. [核心概念](#1-核心概念)
2. [Demo 1：基础状态机工作流](#2-demo-1基础状态机工作流)
3. [Demo 2：三步问答助手](#3-demo-2三步问答助手)
4. [对比分析](#4-对比分析)
5. [进阶技巧](#5-进阶技巧)

---

## 1. 核心概念

### 1.1 什么是 LangGraph？

LangGraph 是 LangChain 生态系统的重要扩展，它将智能体执行流程建模为**状态机（State Machine）**，用**有向图（Directed Graph）**表示。

```
与传统 Agent 框架的区别：

传统框架（AutoGen, AgentScope）:
┌─────────────────────────────────────┐
│  Agent A ←→ Agent B ←→ Agent C     │  ← 基于对话
│  (消息传递、角色扮演)               │
└─────────────────────────────────────┘

LangGraph:
┌─────────────────────────────────────┐
│  State → Node A → Node B → Node C  │  ← 基于状态
│  (状态流转、明确控制流)             │
└─────────────────────────────────────┘
```

### 1.2 三大核心要素

#### ① 状态（State）

状态是贯穿整个工作流的共享数据结构。

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    """状态就像一个'共享笔记本'"""
    messages: List[str]      # 对话历史
    current_task: str        # 当前任务
    final_answer: str        # 最终答案
```

**关键特性：**
- 所有节点都可以读取和更新状态
- 状态在工作流执行期间持久化
- 使用 TypedDict 提供类型提示

#### ② 节点（Nodes）

节点是执行具体工作的 Python 函数。

```python
def my_node(state: AgentState) -> dict:
    """节点函数签名：接收状态，返回更新"""
    
    # 读取状态
    task = state["current_task"]
    
    # 执行工作（调用 LLM、调用 API 等）
    result = do_something(task)
    
    # 返回要更新的状态字段
    return {"result": result}
```

**关键特性：**
- 输入是完整状态，输出是状态更新（字典）
- 返回的字典会**合并**到当前状态中
- 每个节点专注于一个明确的任务

#### ③ 边（Edges）

边定义节点之间的跳转逻辑。

**常规边：** 固定跳转
```python
workflow.add_edge("node_a", "node_b")  # A → B
```

**条件边：** 动态路由（LangGraph 的核心优势）
```python
def should_continue(state: AgentState) -> str:
    """根据状态决定下一步"""
    if state["iteration"] >= 3:
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "node_a",
    should_continue,
    {"continue": "node_b", "end": END}
)
```

### 1.3 执行流程图

```
┌──────────────────────────────────────────────────────────┐
│                     LangGraph 执行流程                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   1. 初始化状态                                           │
│      inputs = {"current_task": "xxx", ...}              │
│                        ↓                                 │
│   2. 从入口点开始                                         │
│      workflow.set_entry_point("start_node")             │
│                        ↓                                 │
│   3. 执行节点函数                                         │
│      output = node_func(current_state)                  │
│      state.update(output)  # 合并更新                    │
│                        ↓                                 │
│   4. 根据边跳转                                           │
│      if 常规边: → 固定目标节点                            │
│      if 条件边: → 调用判断函数 → 返回目标节点             │
│                        ↓                                 │
│   5. 重复 3-4 直到到达 END                               │
│                        ↓                                 │
│   6. 返回最终状态                                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Demo 1：基础状态机工作流

### 2.1 项目目标

展示 LangGraph 最核心的功能：
- 状态定义与更新
- 节点之间的协作
- 条件边实现循环

### 2.2 架构图

```
┌─────────────────────────────────────────────────────────┐
│                   规划者-执行者 循环                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                     START                               │
│                       │                                 │
│                       ▼                                 │
│              ┌─────────────┐                            │
│              │   planner   │◄───────────┐              │
│              │  (规划者)   │             │              │
│              └──────┬──────┘             │              │
│                     │                    │              │
│                     ▼                    │              │
│              ┌─────────────┐             │              │
│              │  executor   │             │              │
│              │  (执行者)   │             │              │
│              └──────┬──────┘             │              │
│                     │                    │              │
│                     ▼                    │              │
│         ┌────────────────────┐           │              │
│         │  should_continue   │───────────┘              │
│         │   (条件判断)       │                          │
│         └────────┬───────────┘                          │
│                  │                                      │
│                  ▼ (end)                                │
│                 END                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.3 代码结构分解

#### 步骤 1：定义状态

```python
class AgentState(TypedDict):
    messages: List[str]      # 对话历史
    current_task: str        # 当前任务
    plan: str                # 当前计划
    result: str              # 执行结果
    iteration: int           # 迭代次数 ← 防止无限循环
    final_answer: str        # 最终答案
```

**设计要点：**
- `iteration` 字段用于控制循环次数
- `plan` 和 `result` 分离，便于迭代优化

#### 步骤 2：定义节点

**规划者节点：**
```python
def planner_node(state: AgentState) -> dict:
    """分析任务，制定计划"""
    
    # 首次规划 vs 优化计划
    if state["iteration"] == 0:
        prompt = f"为任务制定计划：{state['current_task']}"
    else:
        prompt = f"根据结果优化计划：{state['result']}"
    
    plan = llm.invoke(prompt)
    return {"plan": plan, "iteration": state["iteration"] + 1}
```

**执行者节点：**
```python
def executor_node(state: AgentState) -> dict:
    """执行计划，产出结果"""
    result = llm.invoke(f"执行计划：{state['plan']}")
    return {"result": result}
```

#### 步骤 3：定义条件边

```python
def should_continue(state: AgentState) -> str:
    """决定是否继续迭代"""
    
    # 条件 1：超过最大迭代次数
    if state["iteration"] >= 3:
        return "end_workflow"
    
    # 条件 2：任务完成
    if "任务已完成" in state["plan"]:
        return "end_workflow"
    
    # 条件 3：继续循环
    return "continue_to_planner"
```

#### 步骤 4：构建图

```python
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

# 设置入口
workflow.set_entry_point("planner")

# 常规边
workflow.add_edge("planner", "executor")

# 条件边
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue_to_planner": "planner",
        "end_workflow": END
    }
)

# 编译
app = workflow.compile()
```

### 2.4 执行示例

```
输入: "帮我写一首关于人工智能的短诗"

执行过程:
┌──────────────────────────────────────────────────┐
│ 迭代 #1                                          │
│   [规划者] 制定计划: 1.确定主题 2.构思意象 3.创作 │
│   [执行者] 执行结果: (初版诗歌)                   │
│   [判断] → 继续迭代                              │
├──────────────────────────────────────────────────┤
│ 迭代 #2                                          │
│   [规划者] 优化计划: 增加韵律感，丰富意象         │
│   [执行者] 执行结果: (优化版诗歌)                 │
│   [判断] → 继续迭代                              │
├──────────────────────────────────────────────────┤
│ 迭代 #3                                          │
│   [规划者] 最终润色                              │
│   [执行者] 执行结果: (最终版诗歌)                 │
│   [判断] → 达到最大迭代次数，结束                 │
└──────────────────────────────────────────────────┘
```

---

## 3. Demo 2：三步问答助手

### 3.1 项目目标

展示 LangGraph 的实际应用：
- 线性工作流设计
- 外部工具集成（Tavily 搜索）
- 错误处理与降级策略

### 3.2 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    三步问答助手                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   START                                                 │
│     │                                                   │
│     ▼                                                   │
│   ┌─────────────┐                                       │
│   │  understand │  理解用户意图，优化搜索词            │
│   │   (理解)    │                                       │
│   └──────┬──────┘                                       │
│          │                                              │
│          ▼                                              │
│   ┌─────────────┐                                       │
│   │   search    │  调用 Tavily API 搜索                │
│   │   (搜索)    │                                       │
│   └──────┬──────┘                                       │
│          │                                              │
│          ▼                                              │
│   ┌─────────────┐                                       │
│   │   answer    │  基于搜索结果生成答案                │
│   │   (回答)    │  (搜索失败时降级到 LLM 知识)         │
│   └──────┬──────┘                                       │
│          │                                              │
│          ▼                                              │
│         END                                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.3 代码结构分解

#### 步骤 1：定义状态

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    # 使用 add_messages 注解实现消息自动追加
    messages: Annotated[list, add_messages]
    
    user_query: str          # 用户需求总结
    search_query: str        # 优化后的搜索词
    search_results: str      # 搜索结果
    final_answer: str        # 最终答案
    step: str                # 当前步骤标记
```

**关键设计：**
- `user_query` vs `search_query` 分离，支持查询优化
- `step` 字段用于错误处理和条件判断

#### 步骤 2：理解节点

```python
def understand_query_node(state: SearchState) -> dict:
    """理解用户意图，生成搜索关键词"""
    
    user_message = state["messages"][-1].content
    
    prompt = f"""分析用户查询："{user_message}"

请完成：
1. 总结用户需求
2. 生成搜索关键词

格式：
理解：[需求总结]
搜索词：[关键词]"""
    
    response = llm.invoke(prompt)
    
    # 解析搜索词
    search_query = extract_search_query(response)
    
    return {
        "user_query": response,
        "search_query": search_query,
        "step": "understood"
    }
```

#### 步骤 3：搜索节点

```python
def tavily_search_node(state: SearchState) -> dict:
    """调用 Tavily API 搜索"""
    
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        response = tavily.search(
            query=state["search_query"],
            search_depth="basic",
            max_results=5
        )
        
        # 格式化结果
        results = format_results(response)
        
        return {
            "search_results": results,
            "step": "searched"
        }
        
    except Exception as e:
        # 错误处理
        return {
            "search_results": f"搜索失败: {e}",
            "step": "search_failed"
        }
```

#### 步骤 4：回答节点（弹性设计）

```python
def generate_answer_node(state: SearchState) -> dict:
    """生成最终答案，支持降级策略"""
    
    if state["step"] == "search_failed":
        # 降级：使用 LLM 自身知识
        prompt = f"搜索失败，基于你的知识回答：{state['user_query']}"
    else:
        # 正常：基于搜索结果回答
        prompt = f"""基于搜索结果回答：
用户问题：{state['user_query']}
搜索结果：{state['search_results']}"""
    
    answer = llm.invoke(prompt)
    
    return {
        "final_answer": answer,
        "step": "completed"
    }
```

### 3.4 执行示例

```
输入: "2024年诺贝尔物理学奖得主是谁？"

执行过程:
┌──────────────────────────────────────────────────┐
│ [理解节点]                                       │
│   用户问题: 2024年诺贝尔物理学奖得主是谁？        │
│   LLM 分析:                                      │
│     理解：用户想了解2024年诺贝尔物理学奖获得者    │
│     搜索词：2024 Nobel Prize Physics winner      │
├──────────────────────────────────────────────────┤
│ [搜索节点]                                       │
│   搜索词: 2024 Nobel Prize Physics winner        │
│   结果: ✅ 获取 5 条搜索结果                      │
├──────────────────────────────────────────────────┤
│ [回答节点]                                       │
│   基于搜索结果生成答案...                         │
│   答案: 2024年诺贝尔物理学奖授予...              │
└──────────────────────────────────────────────────┘
```

---

## 4. 对比分析

### 4.1 与其他框架对比

| 特性 | LangGraph | AutoGen | AgentScope |
|------|-----------|---------|------------|
| **范式** | 状态机 | 多智能体对话 | 多智能体对话 |
| **控制流** | 显式图结构 | 隐式消息传递 | 轮询机制 |
| **循环支持** | 原生支持 | 需要手动实现 | 需要手动实现 |
| **可调试性** | 高（可视化图） | 中 | 中 |
| **学习曲线** | 中等 | 低 | 低 |
| **适用场景** | 复杂工作流 | 角色扮演对话 | 游戏仿真 |

### 4.2 两个 Demo 对比

| 特性 | Demo 1（基础） | Demo 2（问答助手） |
|------|---------------|-------------------|
| **图结构** | 循环图 | 线性图 |
| **边类型** | 条件边 + 常规边 | 仅常规边 |
| **外部依赖** | 仅 LLM | LLM + Tavily API |
| **复杂度** | 中等 | 较高 |
| **学习价值** | 理解核心概念 | 实际应用场景 |

---

## 5. 进阶技巧

### 5.1 状态注解

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # add_messages: 自动追加消息而非覆盖
    messages: Annotated[list, add_messages]
    
    # 自定义 reducer 函数
    count: Annotated[int, lambda x, y: x + y]
```

### 5.2 检查点（Checkpointing）

```python
from langgraph.checkpoint.memory import InMemorySaver

# 添加检查点支持（支持暂停/恢复）
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

# 带线程 ID 运行（支持多会话）
config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke(inputs, config)
```

### 5.3 可视化图

```python
from IPython.display import Image, display

# 生成图的可视化
display(Image(app.get_graph().draw_mermaid_png()))
```

### 5.4 流式输出

```python
# stream() 返回每个节点的输出
for event in app.stream(inputs):
    for node_name, node_output in event.items():
        print(f"[{node_name}]: {node_output}")

# astream() 异步流式输出
async for event in app.astream(inputs):
    print(event)
```

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 运行 Demo 1
python src/demo1_basic_workflow.py

# 运行 Demo 2
python src/demo2_search_assistant.py

# Demo 2 交互模式
python src/demo2_search_assistant.py --interactive
```
