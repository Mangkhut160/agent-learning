# Agent Learning

A hands-on learning repository for LLM agents, covering classic reasoning paradigms, mainstream multi-agent frameworks, and a small self-built agent framework.

## Highlights
- Reimplements three classic agent paradigms from scratch: ReAct, Plan-and-Solve, and Reflection
- Includes framework-based demos with AutoGen, AgentScope, and LangGraph
- Ships a custom `HelloAgents` package to compare common abstractions across multiple agent styles
- Preserves detailed study notes that connect conceptual understanding with practical implementation
- Useful both as a learning log and as a compact portfolio of agent-engineering experiments

## Repository Structure
The repository is organized as a set of focused subprojects rather than a single application.

Core components:
- `react_agent_project/`: hand-written ReAct loop and tool execution flow
- `plan_solve_agent_project/`: planner/executor split for task decomposition
- `reflection_agent_project/`: execution-review-improvement loop
- `autogen_project/`: role-based multi-agent collaboration with AutoGen
- `agentscope_project/`: asynchronous multi-agent game simulation with AgentScope
- `langgraph_project/`: graph-based workflow orchestration demos
- `hello-agents/`: custom lightweight agent framework packaged for local installation
- `docs/` and Chinese study notes: supporting theory and framework comparison write-ups

## Why This Repository Matters
This repo shows not just framework usage, but comparative understanding. It captures the difference between implementing agent control loops manually and using framework-level abstractions for orchestration, memory, tools, and collaboration.

## Getting Started
Each subproject contains its own `requirements.txt` and README. A typical workflow is:

```bash
cd react_agent_project
pip install -r requirements.txt
python main.py
```

Repeat the same pattern for the other subprojects depending on the paradigm or framework you want to explore.