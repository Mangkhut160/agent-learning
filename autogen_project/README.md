# AutoGen 多智能体项目

基于 AutoGen 0.7.5 框架的多智能体协作系统，模拟真实软件开发团队的协作流程。

## 项目结构

```
autogen_project/
├── .env                    # 环境变量配置
├── .env.example            # 环境变量示例
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明
├── main.py                 # 主程序入口
└── src/
    ├── __init__.py
    ├── model_client.py     # 模型客户端配置
    ├── agents.py           # 四个智能体角色定义
    └── team.py             # 团队协作模式定义
```

## 团队角色

| 角色 | 职责 | 结束语 |
|------|------|--------|
| ProductManager | 需求分析、技术规划、风险评估 | "请工程师开始实现" |
| Engineer | 代码编写、技术实现、错误处理 | "请代码审查员检查" |
| CodeReviewer | 代码质量审查、安全检查、最佳实践 | "请用户代理测试" |
| UserProxy | 验证功能、反馈结果、发出终止令 | "TERMINATE" |

## 协作流程

```
用户代理发起任务
       ↓
产品经理：需求分析 → "请工程师开始实现"
       ↓
工程师：代码实现 → "请代码审查员检查"
       ↓
代码审查员：质量审查 → "请用户代理测试"
       ↓
用户代理：验证测试 → "TERMINATE"（终止）
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

### 3. 运行示例

```bash
cd autogen_project
py main.py
```

## 技术要点

### 模型客户端

使用 `OpenAIChatCompletionClient` 兼容任何 OpenAI API 规范的服务：

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="MiniMax-M2.7",
    api_key="your-api-key",
    base_url="https://api.minimaxi.com/v1"
)
```

### 团队模式

采用 `RoundRobinGroupChat`（轮询群聊），按顺序让每个智能体发言，直到收到 `TERMINATE` 信号。

### 终止条件

`TextMentionTermination("TERMINATE")` —— 当任何消息中包含 "TERMINATE" 时，对话结束。

## 扩展练习

1. **增加角色**：添加 QA 工程师、运维工程师等新角色
2. **修改终止条件**：使用 `MaxMessageTermination` 限制最大消息数
3. **切换团队模式**：尝试 `FunctionCallGroupChat` 实现更灵活的工具调用
4. **更换任务**：修改 main.py 中的 task 变量，尝试其他软件开发场景