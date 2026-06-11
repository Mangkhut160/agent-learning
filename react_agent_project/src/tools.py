"""
工具定义模块
包含智能体可调用的外部工具
"""

import os
from serpapi import SerpApiClient
from dotenv import load_dotenv

load_dotenv()


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。

    Args:
        query: 搜索查询字符串

    Returns:
        搜索结果的文本摘要
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")

    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误: SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析: 优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])

        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]

        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


def calculator(expression: str) -> str:
    """
    一个简单的计算器工具，用于执行基本的数学运算。

    Args:
        expression: 数学表达式字符串，如 "2 + 3 * 4"

    Returns:
        计算结果的字符串
    """
    print(f"🧮 正在计算: {expression}")
    try:
        # 安全地评估数学表达式（仅允许基本运算）
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含不允许的字符"

        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 工具描述常量，用于注册工具时使用
SEARCH_DESCRIPTION = (
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    "输入应该是一个搜索查询字符串。"
)

CALCULATOR_DESCRIPTION = (
    "一个计算器工具。当你需要进行数学运算时使用此工具。"
    "输入应该是一个数学表达式，如 '2 + 3 * 4'。"
)


if __name__ == "__main__":
    # 测试搜索工具
    result = search("Python最新版本")
    print(f"搜索结果:\n{result}")
