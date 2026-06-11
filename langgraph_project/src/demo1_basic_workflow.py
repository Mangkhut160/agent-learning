"""
Demo 1: 基础状态机工作流
展示 LangGraph 核心概念：状态(State)、节点(Nodes)、边(Edges)
实现规划者-执行者的循环迭代工作流
"""

import os
import sys
import re

# 设置 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

# 加载环境变量
load_dotenv()


# ============================================================
# 第一步：定义全局状态 (State)
# ============================================================

class AgentState(TypedDict):
    """全局状态定义"""
    messages: List[str]
    current_task: str
    plan: str
    result: str
    iteration: int
    final_answer: str


# ============================================================
# 第二步：定义节点 (Nodes)
# ============================================================

def planner_node(state: AgentState) -> dict:
    """规划者节点"""
    print(f"\n{'='*50}")
    print(f"🎯 [规划者节点] 迭代 #{state['iteration'] + 1}")
    print(f"{'='*50}")

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.7
    )

    current_task = state["current_task"]
    previous_result = state.get("result", "")
    iteration = state.get("iteration", 0)

    if iteration == 0:
        prompt = f"""分析任务并制定执行计划。

任务：{current_task}

要求：
1. 列出具体执行步骤（3-5步）
2. 每步简洁明确
3. 最后说"计划制定完成"

直接输出计划："""
    else:
        prompt = f"""根据执行反馈优化计划。

原任务：{current_task}
之前的计划：{state.get('plan', '')}
执行结果：{previous_result}

如果结果已经很好，输出"任务已完成"。
否则输出优化后的计划："""

    response = llm.invoke([
        SystemMessage(content="你是专业的任务规划者。"),
        HumanMessage(content=prompt)
    ])
    new_plan = response.content

    # 清理 thinking 标签
    new_plan = re.sub(r'<think>.*?</think>', '', new_plan, flags=re.DOTALL).strip()

    print(f"📋 计划:\n{new_plan[:200]}...")

    return {
        "plan": new_plan,
        "iteration": iteration + 1
    }


def executor_node(state: AgentState) -> dict:
    """执行者节点"""
    print(f"\n{'─'*50}")
    print(f"⚙️ [执行者节点] 执行计划")
    print(f"{'─'*50}")

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.5
    )

    current_plan = state["plan"]
    current_task = state["current_task"]

    prompt = f"""请执行以下计划。

任务目标：{current_task}
计划内容：{current_plan}

执行要求：
1. 按计划步骤逐一执行
2. 给出具体产出
3. 如有问题，说明需要改进的地方

输出执行结果："""

    response = llm.invoke([
        SystemMessage(content="你是高效的执行者。"),
        HumanMessage(content=prompt)
    ])
    result = response.content

    # 清理 thinking 标签
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

    print(f"📤 结果:\n{result[:200]}...")

    return {"result": result}


# ============================================================
# 第三步：定义条件边 (Conditional Edges)
# ============================================================

def should_continue(state: AgentState) -> str:
    """条件判断函数"""
    iteration = state.get("iteration", 0)
    result = state.get("result", "")
    plan = state.get("plan", "")

    print(f"\n🔀 [条件判断] 迭代次数: {iteration}")

    MAX_ITERATIONS = 2
    if iteration >= MAX_ITERATIONS:
        print(f"   → 达到最大迭代次数 ({MAX_ITERATIONS})，结束流程")
        state["final_answer"] = result
        return "end_workflow"

    if "任务已完成" in plan or "已完成" in result:
        print(f"   → 检测到完成信号，结束流程")
        state["final_answer"] = result
        return "end_workflow"

    print(f"   → 继续下一轮迭代")
    return "continue_to_planner"


# ============================================================
# 第四步：构建图 (Build Graph)
# ============================================================

def create_basic_workflow():
    """创建基础工作流图"""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")

    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "continue_to_planner": "planner",
            "end_workflow": END
        }
    )

    app = workflow.compile()
    return app


# ============================================================
# 第五步：运行应用
# ============================================================

def run_demo(task: str = "帮我设计一个简单的待办事项应用"):
    """运行 Demo 1"""
    print("\n" + "="*60)
    print("LangGraph Demo 1: 基础状态机工作流")
    print("="*60)
    print(f"\n📝 任务: {task}")

    app = create_basic_workflow()

    inputs = {
        "messages": [],
        "current_task": task,
        "plan": "",
        "result": "",
        "iteration": 0,
        "final_answer": ""
    }

    print("\n🚀 开始执行工作流...\n")

    for event in app.stream(inputs):
        for node_name, node_output in event.items():
            print(f"\n📦 节点 [{node_name}] 更新:")
            for key, value in node_output.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   {key}: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")

    print("\n" + "="*60)
    print("✅ 工作流执行完成")
    print("="*60)


if __name__ == "__main__":
    run_demo("帮我写一首关于人工智能的短诗")
