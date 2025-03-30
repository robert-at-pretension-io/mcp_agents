import os
import aiohttp
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from agents import function_tool, RunContextWrapper

class SearchResult(BaseModel):
    """Represents a single search result from Brave Search"""
    title: str = Field(description="The title of the search result")
    url: str = Field(description="The URL of the search result")
    description: Optional[str] = Field(None, description="A description of the content")
    page_age: Optional[str] = Field(None, description="Age indicator for the content")
    page_fetched: Optional[str] = Field(None, description="When the page was fetched")
    language: Optional[str] = Field(None, description="The language of the content")
    family_friendly: Optional[bool] = Field(None, description="Whether the content is family friendly")
    extra_snippets: Optional[List[str]] = Field(None, description="Additional snippets from the content")

class SearchResults(BaseModel):
    """Container for search results from Brave Search"""
    results: List[SearchResult] = Field(description="List of search results")
    query: str = Field(description="The query that was searched")
    more_results_available: Optional[bool] = Field(None, description="Whether more results are available")

async def _perform_brave_search(query: str, count: int = 10) -> SearchResults:
    """
    Internal function to perform the actual Brave Search API call
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY environment variable is not set")
    
    base_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    params = {
        "q": query,
        "count": min(count, 20),  # Maximum 20 results
        "safesearch": "moderate"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, headers=headers, params=params) as response:
            if not response.ok:
                error_text = await response.text()
                raise ValueError(f"API request failed with status {response.status}: {error_text}")
            
            data = await response.json()
            
            # Extract and format the results
            search_results = []
            web_results = data.get("web", {}).get("results", [])
            
            for result in web_results:
                search_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("description"),
                        page_age=result.get("page_age"),
                        page_fetched=result.get("page_fetched"),
                        language=result.get("language"),
                        family_friendly=result.get("family_friendly"),
                        extra_snippets=result.get("extra_snippets")
                    )
                )
            
            query_data = data.get("query", {})
            return SearchResults(
                results=search_results,
                query=query_data.get("original", query),
                more_results_available=query_data.get("more_results_available")
            )

@function_tool
async def brave_search(ctx: RunContextWrapper[Any], query: str, count: int) -> SearchResults:
    """Web search tool powered by Brave Search that retrieves relevant results from across the internet. Use this to:
            
    1. Find current information and facts from the web
    2. Research topics with results from multiple sources
    3. Verify claims or check information accuracy
    4. Discover recent news, trends, and developments
    5. Find specific websites, documentation, or resources
    
    Tips for effective searches:
    - Use specific keywords rather than full questions
    - Include important technical terms, names, or identifiers
    - Add date ranges for time-sensitive information
    - Use quotes for exact phrase matching
    
    Each result contains:
    - Title and URL of the webpage
    - Brief description of the content
    - Age indicators showing content freshness
    
    Args:
        query: The search query - be specific and include relevant keywords
        count: Number of results to return (max 20). Use more results for broad research, fewer for specific queries.
    
    Returns:
        A dictionary containing search results with titles, URLs, and descriptions
    """
    try:
        search_results = await _perform_brave_search(query, count)
        
        return search_results
    except Exception as e:
        return {"error": str(e), "results": []}