"""
Agent实现层
===========

本模块包含HelloAgents框架的所有具体Agent实现。

已实现的Agent类型:
- SimpleAgent: 基础对话Agent，最简单的Agent实现
- ReActAgent: 推理与行动结合的Agent，融合思考和工具调用
- ReflectionAgent: 自我反思Agent，通过迭代改进生成质量
- PlanAndSolveAgent: 规划与执行Agent，先规划后逐步执行

每个Agent都继承自core.Agent基类，遵循统一的接口规范。

使用示例:
    from hello_agents.agents import SimpleAgent, ReActAgent

    # 简单对话
    simple = SimpleAgent(name="助手", llm=llm)

    # 推理Agent
    react = ReActAgent(name="研究者", llm=llm, tool_registry=registry)
"""

from .simple_agent import SimpleAgent
from .react_agent import ReActAgent
from .reflection_agent import ReflectionAgent
from .plan_solve_agent import PlanAndSolveAgent

__all__ = [
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
]