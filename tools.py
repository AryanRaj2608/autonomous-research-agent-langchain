"""
tools.py - Research Tools for Autonomous Research Agent

This module defines the tools used by the research agent:
1. Tavily Search - For web searching and gathering current information
2. Wikipedia - For querying encyclopedic knowledge and background information
"""

import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load environment variables
load_dotenv()


def get_search_tools():
    """
    Initialize and return the research tools for the agent.
    
    Returns:
        list: A list of LangChain tools [TavilySearchResults, WikipediaQueryRun]
    """

    # --- Tavily Search Tool ---
    # Tavily provides AI-optimized search results from the web
    tavily_search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        name="tavily_search",
        description=(
            "A web search engine powered by Tavily. "
            "Use this tool to search for current information, recent developments, "
            "news, and real-time data about any topic. "
            "Input should be a search query string."
        ),
    )

    # --- Wikipedia Tool ---
    # Wikipedia provides detailed encyclopedic information
    wiki_api_wrapper = WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000,
    )

    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=wiki_api_wrapper,
        name="wikipedia",
        description=(
            "A tool to search Wikipedia for encyclopedic information. "
            "Use this tool to get background knowledge, definitions, history, "
            "and detailed explanations about a topic. "
            "Input should be a search query string."
        ),
    )

    return [tavily_search, wikipedia_tool]


if __name__ == "__main__":
    # Quick test to verify tools are initialized correctly
    tools = get_search_tools()
    print("Tools initialized successfully!")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")
