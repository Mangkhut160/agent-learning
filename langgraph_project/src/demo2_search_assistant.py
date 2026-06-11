"""
Demo 2: 三步问答助手
"""

import os
import sys
import re

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str


def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.7
    )


def clean_thinking(text):
    return re.sub(r"_flashdata.*?]", "", text, flags=re.DOTALL).strip()


def understand_query_node(state: SearchState) -> dict:
    print("\n" + "=" * 50)
    print("[理解节点] 分析用户意图")
    
    llm = get_llm()
    user_message = state["messages"][-1].content
    print("用户问题:", user_message)

    prompt = f"分析用户查询并生成搜索关键词。\n用户问题：{user_message}\n格式：\n理解：[需求总结]\n搜索词：[关键词]"

    response = llm.invoke([
        SystemMessage(content="你是查询分析专家。"),
        HumanMessage(content=prompt)
    ])
    response_text = clean_thinking(response.content)
    print("LLM 分析:", response_text[:200])

    search_query = user_message
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip().split("\n")[0]

    print("优化后的搜索词:", search_query)

    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content="搜索中: " + search_query)]
    }


def tavily_search_node(state: SearchState) -> dict:
    print("\n" + "-" * 50)
    print("[搜索节点] 执行搜索")
    
    search_query = state["search_query"]
    print("搜索词:", search_query)

    if TAVILY_AVAILABLE:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = tavily.search(query=search_query, search_depth="basic", max_results=3)
            results = []
            for r in response.get("results", []):
                results.append(r.get("title", "") + ": " + r.get("content", "")[:100])
            search_results = "\n".join(results)
            print("Tavily 搜索成功")
            return {"search_results": search_results, "step": "searched"}
        except Exception as e:
            print("Tavily 失败:", e)
            return {"search_results": "搜索失败", "step": "search_failed"}
    else:
        print("使用 LLM 模拟搜索")
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content="模拟搜索引擎结果。"),
            HumanMessage(content="模拟搜索: " + search_query)
        ])
        return {"search_results": clean_thinking(response.content), "step": "searched"}


def generate_answer_node(state: SearchState) -> dict:
    print("\n" + "-" * 50)
    print("[回答节点] 生成答案")
    
    llm = get_llm()

    prompt = f"基于搜索结果回答问题。\n用户问题：{state['user_query']}\n搜索结果：{state['search_results']}\n请综合回答。"

    response = llm.invoke([
        SystemMessage(content="你是问答助手。"),
        HumanMessage(content=prompt)
    ])
    answer = clean_thinking(response.content)
    print("答案:", answer[:200], "...")

    return {"final_answer": answer, "step": "completed"}


def create_search_assistant():
    workflow = StateGraph(SearchState)
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


def run_demo(query="最近AI有什么突破？"):
    print("\n" + "=" * 60)
    print("LangGraph Demo 2: 三步问答助手")
    print("=" * 60)
    print("用户问题:", query)

    app = create_search_assistant()
    inputs = {
        "messages": [HumanMessage(content=query)],
        "user_query": "", "search_query": "",
        "search_results": "", "final_answer": "", "step": ""
    }

    print("\n开始执行工作流...")
    result = app.invoke(inputs)
    
    print("\n" + "=" * 60)
    print("最终答案:")
    print("=" * 60)
    print(result["final_answer"])


if __name__ == "__main__":
    run_demo()
