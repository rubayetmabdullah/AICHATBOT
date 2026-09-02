from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

search = DuckDuckGoSearchRun()


@tool("search")
def search_tool(query: str) -> str:
    """Search the web for up-to-date information."""
    return search.run(query)


response = search_tool.invoke({"query": "latest Python news"})