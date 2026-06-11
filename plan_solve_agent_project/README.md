# Plan-and-Solve 智能体项目

基于 Plan-and-Solve（先规划，后执行）范式构建的智能体项目，通过两阶段策略解决多步骤推理问题。

## 项目结构

```
plan_solve_agent_project/
├── .env                    # 环境变量配置
├── .env.example            # 环境变量示例
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明
├── main.py                 # 主程序入口
└── src/
    ├── __init__.py
    ├── llm_client.py           # LLM客户端封装
    ├── planner.py              # 规划器（将问题分解为子步骤）
    ├── executor.py             # 执行器（逐步执行计划）
    └── plan_solve_agent.py     # Plan-and-Solve智能体核心实现
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的API密钥：

```bash
cp .env.example .env
```

需要配置：
- **API_KEY**: 大语言模型的API密钥（支持OpenAI兼容接口）
- **BASE_URL**: API基础URL
- **MODEL_NAME**: 模型名称

### 3. 运行示例

```bash
python main.py
```

进入交互模式：

```bash
python main.py --interactive
```

## Plan-and-Solve 范式说明

Plan-and-Solve = Planning + Solving，是一种"先谋后动"的智能体范式：

1. **规划阶段 (Planning Phase)**: 智能体接收完整问题，将其分解为清晰、分步骤的行动计划
2. **执行阶段 (Solving Phase)**: 严格按照计划中的步骤逐一执行，每步结果作为下一步的上下文

与 ReAct 范式的对比：
- **ReAct**: 思考与行动交替进行，适合需要实时反馈的任务
- **Plan-and-Solve**: 先制定完整计划再执行，适合结构性强、可分解的复杂任务

## 适用场景

- 多步数学应用题
- 需要整合多个信息源的报告撰写
- 代码生成任务（先构思结构，再逐一实现）
- 任何可以被清晰分解为子任务的复杂问题

## 示例对话

```
问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。
      周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？

--- 正在生成计划 ---
计划:
["计算周一卖出的苹果数量：15个",
 "计算周二卖出的苹果数量：周一数量 × 2 = 30个",
 "计算周三卖出的苹果数量：周二数量 - 5 = 25个",
 "计算三天总销量：15 + 30 + 25 = 70个"]

--- 正在执行计划 ---
步骤 1/4: 计算周一卖出的苹果数量 → 15
步骤 2/4: 计算周二卖出的苹果数量 → 30
步骤 3/4: 计算周三卖出的苹果数量 → 25
步骤 4/4: 计算三天总销量 → 70

最终答案: 70
```
