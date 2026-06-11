# ReAct 智能体项目

基于 ReAct (Reasoning + Acting) 范式构建的智能体项目，能够调用外部工具回答实时性问题。

## 项目结构

```
react_agent_project/
├── .env                    # 环境变量配置
├── .env.example            # 环境变量示例
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明
├── main.py                 # 主程序入口
└── src/
    ├── __init__.py
    ├── llm_client.py       # LLM客户端封装
    ├── tools.py            # 工具定义(搜索工具)
    ├── tool_executor.py    # 工具执行器
    └── agent.py            # ReAct智能体核心实现
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
- **SERPAPI_API_KEY**: SerpApi密钥（前往 https://serpapi.com/ 注册获取）

### 3. 运行示例

```bash
python main.py
```

## ReAct 范式说明

ReAct = Reasoning + Acting，是一种让LLM交替进行"思考"和"行动"的范式：

1. **Thought**: LLM分析问题、规划下一步
2. **Action**: 调用外部工具获取信息
3. **Observation**: 观察工具返回结果
4. 循环以上步骤直到获得最终答案

## 示例对话

```
用户: 华为最新手机是什么？

--- 第 1 步 ---
思考: 我需要搜索华为最新的手机型号
🎬 行动: Search[华为最新手机型号 2024]
🔍 正在执行网页搜索...
👀 观察: 华为Mate 70系列是最新旗舰...

--- 第 2 步 ---
思考: 我已经获得了足够的信息来回答问题
🎉 最终答案: 华为最新的手机是Mate 70系列...
```

## 扩展工具

在 `src/tools.py` 中添加新工具，然后在 `main.py` 中注册：

```python
tool_executor.registerTool(
    "Calculator", 
    "计算器工具，用于数学运算",
    calculator
)
```
