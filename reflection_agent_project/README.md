# Reflection 智能体项目

基于 Reflection（反思）范式构建的智能体项目，通过"执行-反思-优化"的迭代循环来持续提升解决方案质量。

## 项目结构

```
reflection_agent_project/
├── .env                    # 环境变量配置
├── .env.example            # 环境变量示例
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明
├── main.py                 # 主程序入口
└── src/
    ├── __init__.py
    ├── llm_client.py           # LLM客户端封装
    ├── memory.py               # 短期记忆模块
    └── reflection_agent.py     # Reflection智能体核心实现
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
py main.py
```

进入交互模式：

```bash
py main.py --interactive
```

## Reflection 范式说明

Reflection = 执行 + 反思 + 优化，是一种让智能体通过自我批判来迭代改进的范式：

1. **初始执行 (Execution)**: 根据任务生成初版解决方案
2. **反思 (Reflection)**: 以评审专家身份批判性分析，找出瓶颈
3. **优化 (Refinement)**: 根据反馈改进方案
4. **循环** 步骤2-3，直到评审认为"无需改进"或达到最大迭代次数

与 ReAct 和 Plan-and-Solve 的对比：
- **ReAct**: 边想边做，适合需要外部信息反馈的任务
- **Plan-and-Solve**: 先规划再执行，适合可分解的结构化任务
- **Reflection**: 迭代自我改进，适合有明确质量标准的生成任务

## 核心组件

### Memory 模块
短期记忆模块，存储每次"执行-反思"循环的完整轨迹：
- `add_record(type, content)`: 记录执行结果或反思反馈
- `get_trajectory()`: 将完整轨迹序列化为文本
- `get_last_execution()`: 获取最新的代码版本

### ReflectionAgent
智能体核心，协调三个角色的提示词：
- **INITIAL_PROMPT**: 程序员角色，生成初版代码
- **REFLECT_PROMPT**: 评审专家角色，批判性分析
- **REFINE_PROMPT**: 程序员角色，根据反馈优化

## 示例输出

```
任务: 编写一个Python函数，找出1到n之间所有的素数

--- 阶段1: 初始执行 ---
生成初版代码（试除法，O(n√n)）

--- 阶段2: 第1轮迭代 ---
反思: 试除法效率低，建议使用埃拉托斯特尼筛法
优化: 实现筛法版本（O(n log log n)）

--- 阶段2: 第2轮迭代 ---
反思: 筛法已高效，无需改进
✅ 优化完成
```
