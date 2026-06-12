# Agent Learning

<p align="right">
  <a href="#english">English</a> | <a href="#chinese">中文</a>
</p>

<a id="english"></a>
<details open>
<summary><strong>English</strong></summary>

## Overview
A hands-on learning repository for LLM agents, covering classic reasoning paradigms, mainstream multi-agent frameworks, and a small self-built agent framework.

## Highlights
- Reimplements three classic agent paradigms from scratch: ReAct, Plan-and-Solve, and Reflection.
- Includes framework-based demos with AutoGen, AgentScope, and LangGraph.
- Ships a custom `HelloAgents` package to compare common abstractions across multiple agent styles.
- Preserves detailed study notes that connect conceptual understanding with practical implementation.
- Works both as a learning log and as a compact portfolio of agent-engineering experiments.

## Repository Structure
- `react_agent_project/`: hand-written ReAct loop and tool execution flow.
- `plan_solve_agent_project/`: planner / executor split for task decomposition.
- `reflection_agent_project/`: execution-review-improvement loop.
- `autogen_project/`: role-based multi-agent collaboration with AutoGen.
- `agentscope_project/`: asynchronous multi-agent simulation with AgentScope.
- `langgraph_project/`: graph-based workflow orchestration demos.
- `hello-agents/`: custom lightweight agent framework packaged for local installation.
- `docs/` and the Chinese study notes: supporting theory and framework comparison write-ups.

## Getting Started
Each subproject contains its own `requirements.txt` and README. A typical workflow is:

```bash
cd react_agent_project
pip install -r requirements.txt
python main.py
```

Repeat the same pattern for the other subprojects depending on the paradigm or framework you want to explore.

</details>

<a id="chinese"></a>
<details>
<summary><strong>中文</strong></summary>

## 项目简介
这是一个面向大模型智能体的实践型学习仓库，覆盖经典推理范式、主流多智能体框架，以及一个自建的轻量级 Agent 框架。

## 项目亮点
- 从零实现三种经典 Agent 范式：ReAct、Plan-and-Solve、Reflection。
- 包含基于 AutoGen、AgentScope 和 LangGraph 的框架化示例。
- 提供自定义 `HelloAgents` 包，用来比较多种 Agent 抽象方式。
- 保留了较完整的学习笔记，把概念理解和工程实现串联起来。
- 既可以作为学习记录，也可以作为一份比较紧凑的 Agent Engineering 作品集。

## 仓库结构
- `react_agent_project/`：手写 ReAct 循环与工具执行流。
- `plan_solve_agent_project/`：规划器 / 执行器分离的任务拆解实现。
- `reflection_agent_project/`：执行 - 复盘 - 改进循环。
- `autogen_project/`：基于 AutoGen 的角色协作示例。
- `agentscope_project/`：基于 AgentScope 的异步多智能体模拟。
- `langgraph_project/`：图工作流编排示例。
- `hello-agents/`：可本地安装的自建轻量 Agent 框架。
- `docs/` 与中文学习笔记：理论梳理与框架比较材料。

## 快速开始
每个子项目都有自己的 `requirements.txt` 和 README，典型使用方式如下：

```bash
cd react_agent_project
pip install -r requirements.txt
python main.py
```

根据你想探索的范式或框架，对其他子项目重复同样的步骤即可。

</details>