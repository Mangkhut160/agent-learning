# LangGraph 学习项目

基于教材《Hello Agents》第 6.5 节内容构建的 LangGraph 学习项目。

## 项目结构

```
langgraph_project/
├── main.py                 # 主入口
├── requirements.txt        # 依赖
├── .env.example           # 环境变量模板
├── src/
│   ├── demo1_basic_workflow.py    # Demo 1: 基础状态机
│   └── demo2_search_assistant.py  # Demo 2: 三步问答助手
└── docs/
    └── langgraph_guide.md         # 详细讲解文档
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3. 运行
python main.py
```

## Demo 介绍

### Demo 1: 基础状态机工作流

展示 LangGraph 核心概念：
- 状态（State）的定义与更新
- 节点（Nodes）之间的协作
- 条件边（Conditional Edges）实现循环

```
START → planner → executor → [判断] → planner（循环）
                                ↓
                               END
```

### Demo 2: 三步问答助手

展示实际应用场景：
- 线性工作流设计
- 外部工具集成（Tavily 搜索）
- 错误处理与降级策略

```
START → understand → search → answer → END
```

## 环境变量说明

| 变量名 | 说明 | 获取方式 |
|--------|------|----------|
| `LLM_API_KEY` | LLM API 密钥 | OpenAI / 其他兼容服务 |
| `LLM_MODEL_ID` | 模型 ID | 默认 gpt-4o-mini |
| `LLM_BASE_URL` | API 地址 | 默认 OpenAI 地址 |
| `TAVILY_API_KEY` | Tavily 搜索 API | https://tavily.com 免费注册 |

## 学习路径

1. 阅读 `docs/langgraph_guide.md` 理解核心概念
2. 运行 Demo 1 理解状态机工作流
3. 运行 Demo 2 学习工具集成
4. 修改代码，尝试自己的工作流设计
