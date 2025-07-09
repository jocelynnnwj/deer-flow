# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper
from pydantic import SecretStr

from src.config import SearchEngine, SELECTED_SEARCH_ENGINE
# from src.tools.tavily_search.tavily_search_api_wrapper import (
#     EnhancedTavilySearchAPIWrapper,
# )
# Remove: from src.tools.tavily_search.tavily_search_results_with_images import (
#     TavilySearchResultsWithImages,
# )
from src.tools.decorators import create_logged_tool

if TYPE_CHECKING:
    from src.config.tools import Tools

logger = logging.getLogger(__name__)

# Create logged versions of the search tools
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
LoggedBraveSearch = create_logged_tool(BraveSearch)
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)


# Get the selected search tool
def get_web_search_tool(max_search_results: int):
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        raise RuntimeError("Tavily search should be called via the async search_tavily or platform-specific wrappers, not get_web_search_tool.")
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        return LoggedDuckDuckGoSearch(
            name="web_search",
            num_results=max_search_results,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        return LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        return LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")
