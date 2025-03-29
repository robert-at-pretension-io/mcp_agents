import asyncio
import os
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

from mcp.server.fastmcp import Context, FastMCP

# Helper Function to validate environment variables
def validate_env_var(var_name: str) -> str:
    """Gets an environment variable or raises ValueError if missing."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value

# --- Server State and Lifespan ---
@dataclass
class AppContext:
    """Context holding the shared HTTP client."""
    http_client: httpx.AsyncClient

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize and cleanup the HTTP client."""
    print("Brave Search Server Lifespan: Startup")
    # Standard timeout for API calls
    http_client = httpx.AsyncClient(timeout=30.0)
    try:
        yield AppContext(http_client=http_client)
    finally:
        print("Brave Search Server Lifespan: Shutdown")
        await http_client.aclose()
        print("Brave Search Server Shutdown Complete.")

# MCP Server Initialization with Lifespan
mcp = FastMCP(
    "Brave Search Tool",
    lifespan=app_lifespan,
    description="Searches the web using the Brave Search API.",
)


@mcp.tool()
async def brave_search(ctx: Context, query: str, count: Optional[int] = 10) -> str:
    """
    Web search tool powered by Brave Search that retrieves relevant results from across the internet. Use this to:

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

    The search defaults to returning 10 results but can provide up to 20 with the count parameter.

    NOTE: This tool requires the BRAVE_API_KEY environment variable to be set.
    """
    app_context: AppContext = ctx.lifespan
    try:
        api_key = validate_env_var("BRAVE_API_KEY")
    except ValueError as e:
        return f"Error: {e}"

    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    # Clamp count between 1 and 20 (API max)
    num_results = max(1, min(count or 10, 20))

    base_api_url = "https://api.search.brave.com/res/v1/web/search"
    params = {
        "q": query,
        "count": str(num_results),
        "safesearch": "moderate", # Or "strict" or "off"
        # "country": "us", # Optional: specify country
        # "search_lang": "en", # Optional: specify language
    }
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    try:
        print(f"Performing Brave search for: '{query}' (count: {num_results})")
        response = await app_context.http_client.get(base_api_url, params=params, headers=headers)
        response.raise_for_status() # Check for 4xx/5xx errors

        data = response.json()
        web_results = data.get("web", {}).get("results", [])

        if not web_results:
            return f"No web results found for query: '{query}'"

        # Format results nicely
        output = f"# Brave Search Results for '{query}'\n\n"
        for i, result in enumerate(web_results):
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            description = result.get("description", "No Description")
            # Clean description slightly
            description = description.replace("\n", " ").strip()
            page_age = result.get("page_age")

            output += f"## {i + 1}. {title}\n"
            output += f"   URL: {url}\n"
            if page_age:
                output += f"   Age: {page_age}\n"
            output += f"   Description: {description}\n\n"

        return output.strip()

    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            # Check if the error response is JSON
            error_json = json.loads(error_detail)
            if isinstance(error_json, dict): # Simple check
                 error_detail = json.dumps(error_json, indent=2) # Pretty print JSON error
        except json.JSONDecodeError:
             pass # Keep original text if not JSON
        return (f"Error: Brave Search API request failed with status {e.response.status_code}. "
                f"Detail: {error_detail}")
    except httpx.TimeoutException:
         return f"Error: Request to Brave Search API timed out."
    except httpx.RequestError as e:
        return f"Error: Network request to Brave Search API failed: {e}"
    except json.JSONDecodeError:
         return f"Error: Failed to parse JSON response from Brave Search API."
    except Exception as e:
        return f"An unexpected error occurred during Brave search: {e}"


if __name__ == "__main__":
    print("Starting Brave Search MCP Server...")
    mcp.run()