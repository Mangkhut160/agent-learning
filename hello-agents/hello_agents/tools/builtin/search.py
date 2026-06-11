"""
SearchTool - 智能搜索工具
==========================

内置的多源搜索工具，支持整合多个搜索引擎后端。
自动选择可用的最佳搜索源，并提供统一的搜索结果格式。

支持的搜索后端:
1. Tavily - AI优化的专业搜索服务
2. SerpAPI - 传统Google搜索API

使用示例:
    from hello_agents.tools.builtin import SearchTool

    # 使用混合模式（自动选择最佳后端）
    search = SearchTool(backend="hybrid")

    # 指定使用Tavily
    search = SearchTool(backend="tavily")

    # 执行搜索
    result = search.run({"query": "Python编程语言"})
    print(result)
"""

import os
from typing import Dict, Any, List, Optional

from ..base import Tool, ToolParameter


class SearchTool(Tool):
    """
    智能混合搜索工具

    支持多种搜索引擎后端，智能选择最佳搜索源。
    自动检测可用的API密钥并选择合适的后端。

    继承自:
        Tool: 工具基类

    后端优先级:
        1. Tavily (AI优化搜索)
        2. SerpAPI (传统Google搜索)
        3. 无可用后端时返回配置提示
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None
    ):
        """
        初始化搜索工具

        Args:
            backend: 搜索后端模式，可选值:
                    - "hybrid": 智能选择（默认）
                    - "tavily": 仅使用Tavily
                    - "serpapi": 仅使用SerpAPI
            tavily_key: Tavily API密钥，如果为None则从环境变量读取
            serpapi_key: SerpAPI密钥，如果为None则从环境变量读取
        """
        super().__init__(
            name="search",
            description="智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源（ Tavily 或 Google ）。"
        )

        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

        self.available_backends: List[str] = []
        self._setup_backends()

    def _setup_backends(self) -> None:
        """
        设置可用的搜索后端

        根据配置的API密钥自动检测可用的后端。
        """
        # 检查Tavily
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                print("✅ Tavily搜索源已启用")
            except ImportError:
                print("⚠️ Tavily库未安装，请运行: pip install tavily")

        # 检查SerpAPI
        if self.serpapi_key:
            try:
                import serpapi
                self.available_backends.append("serpapi")
                print("✅ SerpAPI搜索源已启用")
            except ImportError:
                print("⚠️ SerpAPI库未安装，请运行: pip install serpapi")

        if self.available_backends:
            print(f"🔧 可用搜索源: {', '.join(self.available_backends)}")
        else:
            print("⚠️ 没有可用的搜索源，请配置 TAVILY_API_KEY 或 SERPAPI_API_KEY")

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行搜索

        Args:
            parameters: 包含 "query" 键的参数字典

        Returns:
            搜索结果的格式化字符串
        """
        query = parameters.get("query", "")

        if not query or not query.strip():
            return "❌ 错误: 搜索查询不能为空"

        print(f"🔍 开始搜索: {query}")

        # 根据后端模式选择搜索方法
        if self.backend == "tavily":
            return self._search_tavily(query)
        elif self.backend == "serpapi":
            return self._search_serpapi(query)
        else:  # hybrid
            return self._search_hybrid(query)

    def get_parameters(self) -> List[ToolParameter]:
        """
        获取工具参数定义

        Returns:
            包含查询参数定义的列表
        """
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词，例如: Python编程语言、人工智能最新发展"
            )
        ]

    def _search_hybrid(self, query: str) -> str:
        """
        混合搜索 - 智能选择最佳搜索源

        优先使用Tavily，如果失败则降级到SerpAPI。
        如果都不可用，返回配置提示。

        Args:
            query: 搜索查询

        Returns:
            搜索结果字符串
        """
        # 优先使用Tavily
        if "tavily" in self.available_backends:
            try:
                result = self._search_tavily(query)
                if "未找到" not in result and "错误" not in result:
                    return result
            except Exception as e:
                print(f"⚠️ Tavily搜索失败: {e}")
                if "serpapi" in self.available_backends:
                    print("🔄 切换到SerpAPI搜索")

        # 如果Tavily不可用或失败，使用SerpAPI
        if "serpapi" in self.available_backends:
            try:
                return self._search_serpapi(query)
            except Exception as e:
                print(f"⚠️ SerpApi搜索失败: {e}")

        # 如果都不可用，返回配置提示
        return self._get_configuration_help()

    def _search_tavily(self, query: str) -> str:
        """
        使用Tavily搜索

        Args:
            query: 搜索查询

        Returns:
            格式化后的搜索结果
        """
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=3
            )

            result = f"🎯 Tavily AI搜索结果:\n"

            # 添加AI生成的直接答案（如果有）
            if response.get('answer'):
                result += f"\n💡 AI直接答案: {response['answer']}\n"

            # 添加搜索结果
            result += "\n🔗 相关结果:\n"
            for i, item in enumerate(response.get('results', [])[:3], 1):
                result += f"[{i}] {item.get('title', '无标题')}\n"
                content = item.get('content', '')
                result += f"    {content[:200]}...\n" if len(content) > 200 else f"    {content}\n"
                result += f"    来源: {item.get('url', '')}\n\n"

            return result

        except Exception as e:
            return f"❌ Tavily搜索错误: {str(e)}"

    def _search_serpapi(self, query: str) -> str:
        """
        使用SerpAPI搜索（Google）

        Args:
            query: 搜索查询

        Returns:
            格式化后的搜索结果
        """
        try:
            import serpapi

            search = serpapi.GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": 3
            })

            results = search.get_dict()

            result = "🌐 Google搜索结果:\n\n"

            if "organic_results" in results:
                for i, res in enumerate(results["organic_results"][:3], 1):
                    result += f"[{i}] {res.get('title', '无标题')}\n"
                    snippet = res.get('snippet', '')
                    result += f"    {snippet}\n\n"

                # 添加知识图谱（如果有）
                if "knowledge_graph" in results:
                    kg = results["knowledge_graph"]
                    result += "📚 知识图谱:\n"
                    if "title" in kg:
                        result += f"   {kg['title']}\n"
                    if "description" in kg:
                        result += f"   {kg['description']}\n"
            else:
                result += "未找到搜索结果\n"

            return result

        except Exception as e:
            return f"❌ SerpAPI搜索错误: {str(e)}"

    def _get_configuration_help(self) -> str:
        """
        获取API配置帮助信息

        Returns:
            配置提示字符串
        """
        return """❌ 没有可用的搜索源，请配置以下API密钥之一:

1. Tavily API (推荐):
   - 环境变量: TAVILY_API_KEY
   - 获取地址: https://tavily.com/

2. SerpAPI:
   - 环境变量: SERPAPI_API_KEY
   - 获取地址: https://serpapi.com/

配置后重新运行程序。"""