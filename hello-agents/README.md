# HelloAgents - 轻量级智能体框架

## 📖 概述

HelloAgents 是一个模块化、可扩展的 AI 智能体开发框架，基于 Python 实现。本框架遵循"分层解耦、职责单一、接口统一"的核心原则，为开发者提供了构建智能体应用的基础设施。

### 主要特性

- 🌍 **多LLM支持**：支持 OpenAI、ModelScope、智谱AI、VLLM、Ollama 等多种提供商
- 🔧 **统一工具接口**：标准化的工具注册与执行机制
- 🧠 **多种Agent范式**：SimpleAgent、ReActAgent、ReflectionAgent、PlanAndSolveAgent
- 📝 **标准化消息系统**：统一的消息格式，便于上下文管理
- ⚡ **流式响应支持**：实时逐字输出，提升用户体验
- 🔄 **自动检测机制**：智能识别LLM服务商，简化配置

### 学习路径

```
第7章：框架基础 - HelloAgentsLLM、Agent基类、工具系统
第8章：记忆与RAG - 基于第7章的架构扩展Agent能力边界
第9章：上下文工程 - 深入已建立的消息处理机制
第10章：智能体协议 - 扩展新的工具接口
```

---

## 📦 安装

### 方式一：pip 安装（推荐）

```bash
pip install "hello-agents==0.1.1"
```

### 方式二：从源码安装

```bash
cd hello-agents
pip install -e .
```

### 环境要求

- Python 版本 >= 3.10
- 需要配置 LLM API（通过环境变量或代码参数）

---

## 🚀 快速开始

### 1. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# OpenAI 配置
OPENAI_API_KEY=your-openai-api-key
LLM_MODEL_ID=gpt-3.5-turbo

# 或使用其他提供商，例如 ModelScope：
# MODELSCOPE_API_KEY=your-modelscope-key
# LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
```

### 2. 基础对话示例

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例 - 框架自动检测provider
llm = HelloAgentsLLM()

# 创建SimpleAgent
agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手"
)

# 基础对话
response = agent.run("你好！请介绍一下自己")
print(response)

# 查看对话历史
print(f"历史消息数: {len(agent.get_history())}")
```

### 3. 带工具的Agent示例

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
from hello_agents.tools.builtin import CalculatorTool

load_dotenv()

# 创建LLM和工具注册表
llm = HelloAgentsLLM()
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

# 创建带工具的Agent
agent = SimpleAgent(
    name="数学助手",
    llm=llm,
    tool_registry=registry,
    enable_tool_calling=True
)

# 使用工具
response = agent.run("请帮我计算 2 + 3 * 4")
print(response)
```

---

## 🏗️ 项目结构

```
hello-agents/
├── hello_agents/                 # 主包
│   ├── __init__.py              # 包导出
│   ├── core/                     # 核心框架层
│   │   ├── __init__.py
│   │   ├── agent.py             # Agent基类
│   │   ├── llm.py               # HelloAgentsLLM统一接口
│   │   ├── message.py           # 消息系统
│   │   ├── config.py            # 配置管理
│   │   └── exceptions.py        # 异常体系
│   ├── agents/                   # Agent实现层
│   │   ├── __init__.py
│   │   ├── simple_agent.py      # SimpleAgent实现
│   │   ├── react_agent.py       # ReActAgent实现
│   │   ├── reflection_agent.py  # ReflectionAgent实现
│   │   └── plan_solve_agent.py  # PlanAndSolveAgent实现
│   └── tools/                    # 工具系统层
│       ├── __init__.py
│       ├── base.py              # 工具基类
│       ├── registry.py          # 工具注册机制
│       └── builtin/             # 内置工具集
│           ├── __init__.py
│           ├── calculator.py    # 计算工具
│           └── search.py         # 搜索工具
├── examples/                     # 示例代码
├── tests/                        # 测试文件
├── README.md                     # 本文档
└── setup.py                      # 安装配置
```

---

## 📚 Agent范式详解

### 1. SimpleAgent - 基础对话

最简单的Agent实现，适合日常对话任务。

```python
from hello_agents import SimpleAgent

agent = SimpleAgent(
    name="助手",
    llm=llm,
    system_prompt="你是一个友好的助手"
)

response = agent.run("今天天气怎么样？")
```

### 2. ReActAgent - 推理与行动

通过思考-行动-观察的循环，解决复杂问题。

```python
from hello_agents import ReActAgent, ToolRegistry

registry = ToolRegistry()
registry.register_function("search", "搜索工具", search_func)

agent = ReActAgent(
    name="研究者",
    llm=llm,
    tool_registry=registry,
    max_steps=5
)

result = agent.run("帮我查一下Python的最新版本")
```

### 3. ReflectionAgent - 自我反思

通过迭代改进提升生成质量。

```python
from hello_agents import ReflectionAgent

agent = ReflectionAgent(
    name="写作助手",
    llm=llm,
    max_iterations=3
)

# 使用默认通用提示词
result = agent.run("写一篇关于AI的文章")

# 使用自定义提示词
code_prompts = {
    "initial": "你是Python专家，请编写函数:{task}",
    "reflect": "请审查代码的算法效率",
    "refine": "请根据反馈优化代码"
}
code_agent = ReflectionAgent(name="代码助手", llm=llm, custom_prompts=code_prompts)
```

### 4. PlanAndSolveAgent - 规划与执行

先规划后执行，处理多步骤复杂问题。

```python
from hello_agents import PlanAndSolveAgent

agent = PlanAndSolveAgent(name="规划助手", llm=llm)

result = agent.run("一个水果店周一卖了15个苹果，周二卖了周一的2倍，"
                   "周三比周二少5个，三天共卖了多少苹果？")
```

---

## 🔧 工具系统

### 内置工具

#### CalculatorTool - 数学计算器

```python
from hello_agents.tools.builtin import CalculatorTool

calc = CalculatorTool()
result = calc.run({"expression": "sqrt(16) + 2 * 3"})
# 结果: "10.0"
```

#### SearchTool - 搜索工具

```python
from hello_agents.tools.builtin import SearchTool

search = SearchTool(backend="hybrid")  # 自动选择最佳后端
result = search.run({"query": "Python教程"})
```

### 自定义工具

#### 方式一：注册函数

```python
from hello_agents import ToolRegistry

def my_calculator(expression):
    return str(eval(expression))

registry = ToolRegistry()
registry.register_function(
    name="calc",
    description="计算器",
    func=my_calculator
)

result = registry.execute_tool("calc", "2+3")
```

#### 方式二：继承Tool类

```python
from hello_agents.tools.base import Tool, ToolParameter
from typing import List, Dict, Any

class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="我的自定义工具"
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        # 实现工具逻辑
        return "结果"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="输入参数"
            )
        ]

# 注册和使用
registry.register_tool(MyTool())
```

---

## 🌐 多LLM提供商支持

### OpenAI

```python
# 方式一：环境变量
# OPENAI_API_KEY=your-key

llm = HelloAgentsLLM(provider="openai")
```

### ModelScope

```python
# 方式一：环境变量
# MODELSCOPE_API_KEY=your-key

llm = HelloAgentsLLM(provider="modelscope")
```

### 本地部署

#### VLLM

```python
llm = HelloAgentsLLM(
    provider="vllm",
    model="Qwen/Qwen1.5-0.5B-Chat",
    base_url="http://localhost:8000/v1",
    api_key="vllm"
)
```

#### Ollama

```python
llm = HelloAgentsLLM(
    provider="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
```

### 自动检测

框架会根据环境变量自动检测提供商：

```python
# 设置环境变量
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL_ID=llama3

llm = HelloAgentsLLM()  # 自动检测为ollama
```

---

## 📁 文件清单

| 文件路径 | 说明 |
|---------|------|
| `hello_agents/__init__.py` | 包入口，导出所有公开API |
| `hello_agents/core/__init__.py` | 核心模块初始化 |
| `hello_agents/core/agent.py` | Agent抽象基类 |
| `hello_agents/core/llm.py` | HelloAgentsLLM统一接口 |
| `hello_agents/core/message.py` | Message消息类 |
| `hello_agents/core/config.py` | Config配置类 |
| `hello_agents/core/exceptions.py` | 异常体系定义 |
| `hello_agents/agents/__init__.py` | Agent实现层初始化 |
| `hello_agents/agents/simple_agent.py` | SimpleAgent实现 |
| `hello_agents/agents/react_agent.py` | ReActAgent实现 |
| `hello_agents/agents/reflection_agent.py` | ReflectionAgent实现 |
| `hello_agents/agents/plan_solve_agent.py` | PlanAndSolveAgent实现 |
| `hello_agents/tools/__init__.py` | 工具系统初始化 |
| `hello_agents/tools/base.py` | Tool基类和ToolParameter |
| `hello_agents/tools/registry.py` | ToolRegistry注册表 |
| `hello_agents/tools/builtin/__init__.py` | 内置工具初始化 |
| `hello_agents/tools/builtin/calculator.py` | 计算器工具 |
| `hello_agents/tools/builtin/search.py` | 搜索工具 |
| `README.md` | 本文档 |
| `setup.py` | pip安装配置 |

---

## 🎯 核心设计原则

### 1. 分层解耦

```
┌─────────────────────────────────────┐
│         Agent实现层                  │
│  (SimpleAgent, ReActAgent, etc.)    │
├─────────────────────────────────────┤
│         工具系统层                   │
│   (Tool, ToolRegistry, Builtin)     │
├─────────────────────────────────────┤
│         核心框架层                   │
│  (Agent, LLM, Message, Config)      │
└─────────────────────────────────────┘
```

### 2. 单一职责

每个模块只关注自己的职责：
- `Agent`: 定义智能体的行为接口
- `LLM`: 处理与语言模型的通信
- `Tool`: 提供特定功能
- `Registry`: 管理工具的注册和发现

### 3. 接口统一

所有Agent都实现相同的`run()`方法签名：
```python
def run(self, input_text: str, **kwargs) -> str
```

所有Tool都实现相同的`run()`方法签名：
```python
def run(self, parameters: Dict[str, Any]) -> str
```

---

## 📖 扩展阅读

本框架的设计思想受以下工作启发：
- ReAct: Synergizing Reasoning and Acting in Language Models
- Plan-and-Solve: Planning and Solving Problems
- Reflexion: Language Agents with Self-Reflection

---

## 📄 许可证

本项目基于 MIT 许可证开源。

---

## 🙏 致谢

HelloAgents 由 HelloAgents Team 开发维护。

如有问题或建议，请访问 GitHub 仓库：
https://github.com/jjyaoao/HelloAgents