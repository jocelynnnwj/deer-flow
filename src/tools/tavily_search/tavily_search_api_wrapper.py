# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import aiohttp
import os
import json
import re
from datetime import datetime
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

# Load API key from conf.yaml or environment
def get_tavily_api_key():
    # Try to load from conf.yaml first
    try:
        from src.config import load_yaml_config
        config_path = str((Path(__file__).parent.parent.parent.parent / "conf.yaml").resolve())
        config = load_yaml_config(config_path)
        if "TOOLS" in config and "search" in config["TOOLS"] and "tavily_api_key" in config["TOOLS"]["search"]:
            return config["TOOLS"]["search"]["tavily_api_key"]
    except Exception as e:
        logger.warning(f"Could not load Tavily API key from conf.yaml: {e}")
    
    # Fallback to environment variable
    return os.getenv("TAVILY_API_KEY")

TAVILY_API_KEY = get_tavily_api_key() or ""

PLATFORM_EMOJIS = {
    "twitter.com": "🐦",
    "x.com": "🐦",
    "linkedin.com": "💼",
    "instagram.com": "📸",
    "reddit.com": "👽",
}

def escape_md(text):
    if not text:
        return ""
    # Escape Markdown special characters
    return re.sub(r'([*_`\[\]()~>#+\-=|{}.!])', r'\\\1', str(text))

def normalize_date(date_str):
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return date_str

def dedup_results(results):
    seen_urls = set()
    seen_content = set()
    deduped = []
    for r in results:
        url = r.get("url")
        content = r.get("content", "").strip()
        if url in seen_urls or content in seen_content:
            continue
        seen_urls.add(url)
        seen_content.add(content)
        deduped.append(r)
    return deduped

async def search_tavily(query: str, max_results: int = 5, domain: str | None = None):
    if not TAVILY_API_KEY:
        return "Error: Tavily API key not found in conf.yaml or environment variables"
    
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    params = {
        "query": query,
        "max_results": max_results,
    }
    if domain:
        params["domain"] = domain
    logger.info(f"[TAVILY] Query: {query}, Params: {params}")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=params) as resp:
            data = await resp.json()
            logger.info(f"[TAVILY] Raw response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            results = data.get("results", [])
            results = dedup_results(results)
            md_blocks = []
            for r in results:
                title = escape_md(r.get("title", ""))
                url_ = r.get("url", "")
                content = escape_md(r.get("content", ""))
                image_url = r.get("image_url") or r.get("image")
                username = escape_md(r.get("author") or r.get("username") or "")
                timestamp = normalize_date(r.get("timestamp") or r.get("published_time") or "")
                platform = domain if domain is not None else (url_.split("/")[2] if url_ else "")
                platform = str(platform)
                badge = PLATFORM_EMOJIS.get(platform, "🌐")
                if len(content) > 400:
                    content = content[:400] + f"... [Read more]({url_})"
                # --- New prominent metadata block ---
                meta_top = []
                if username:
                    meta_top.append(f"👤 **User:** {username}")
                if timestamp:
                    meta_top.append(f"🕒 **Time:** {timestamp}")
                meta_top_str = "  ".join(meta_top)
                image_block = f"\n![Preview]({image_url})\n" if image_url else ""
                # ---
                raw_meta = json.dumps(r, ensure_ascii=False)
                md = f"{badge} **[{title}]({url_})**\n"
                if meta_top_str:
                    md += f"{meta_top_str}\n"
                if image_block:
                    md += f"{image_block}"
                md += f"\n{content}\n"
                md += f"\n🔗 [Open in {platform}]({url_}) 🏷️ **Source:** {platform}"
                md += f"\n<!-- RAW_METADATA: {raw_meta} -->\n"
                md_blocks.append(md)
            output = "\n\n".join(md_blocks) if md_blocks else "No results found."
            logger.info(f"[TAVILY] Markdown output: {output[:500]}")  # Truncate for log
            return output

async def search_linkedin(query: str, max_results: int = 5):
    """Search LinkedIn for results using Tavily API."""
    return await search_tavily(query, max_results=max_results, domain="linkedin.com")

async def search_instagram(query: str, max_results: int = 5):
    """Search Instagram for results using Tavily API."""
    return await search_tavily(query, max_results=max_results, domain="instagram.com")

async def search_reddit(query: str, max_results: int = 5):
    """Search Reddit for results using Tavily API."""
    return await search_tavily(query, max_results=max_results, domain="reddit.com")
