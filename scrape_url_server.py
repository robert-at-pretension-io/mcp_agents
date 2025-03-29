import asyncio
import os
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

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
    print("Scraping Server Lifespan: Startup")
    # Increased timeout slightly for potentially slow scraping targets
    http_client = httpx.AsyncClient(timeout=45.0)
    try:
        yield AppContext(http_client=http_client)
    finally:
        print("Scraping Server Lifespan: Shutdown")
        await http_client.aclose()
        print("Scraping Server Shutdown Complete.")


# MCP Server Initialization with Lifespan
mcp = FastMCP(
    "Scraping Tool",
    lifespan=app_lifespan,
    description="Scrapes web pages using ScrapingBee.",
)


# --- HTML Cleaning Helper ---
def _clean_html_for_markdown(html: str, base_url: Optional[str] = None) -> str:
    """Cleans HTML and prepares it for Markdown conversion."""
    if not html or not html.strip():
        return ""

    try:
        # Use 'html.parser' for built-in, no extra deps needed
        soup = BeautifulSoup(html, 'html.parser')

        # Remove common non-content tags
        tags_to_remove = ['script', 'style', 'img', 'svg', 'canvas', 'noscript',
                          'iframe', 'header', 'footer', 'nav', 'aside', 'button',
                          'form', 'input', 'select', 'textarea', 'link', 'meta']
        for tag in soup(tags_to_remove):
            tag.decompose()

        # Optional: Add base URL handling here if needed for relative links in output

        # Convert remaining HTML to Markdown
        markdown_text = md(str(soup), heading_style="ATX", bullets="*")

        return markdown_text.strip()

    except Exception as e:
        print(f"Error cleaning HTML for URL {base_url or ''}: {e}")
        # Fallback: return plain text extraction if cleaning/markdown fails
        try:
            return soup.get_text(separator=' ', strip=True) if 'soup' in locals() else "Error processing HTML structure."
        except Exception:
             return "Error processing HTML content."

@mcp.tool()
async def scrape_url(ctx: Context, url: str, render_js: bool = True) -> str:
    """
    Web scraping tool that extracts and processes content from websites, now with improved performance. Use this to:

    1. Extract text from webpages (news, articles, documentation)
    2. Gather product information from e-commerce sites
    3. Retrieve data from sites with JavaScript-rendered content
    4. Access content behind cookie notifications or simple overlays

    Important notes:
    - Always provide complete URLs including protocol (e.g., 'https://example.com')
    - JavaScript rendering is enabled by default for compatibility
    - Content is automatically processed to extract readable text
    - Set render_js=false for static sites to get faster responses

    Example queries:
    - News article: 'https://news.site.com/article/12345'
    - Product page: 'https://shop.example.com/products/item-name'
    - Documentation: 'https://docs.domain.org/tutorial'

    NOTE: This tool requires the SCRAPINGBEE_API_KEY environment variable to be set.
    """
    app_context: AppContext = ctx.lifespan
    try:
        api_key = validate_env_var("SCRAPINGBEE_API_KEY")
    except ValueError as e:
        return f"Error: {e}"

    if not url or not url.strip().startswith(('http://', 'https://')):
         return "Error: Invalid URL provided. Must start with http:// or https://"


    base_api_url = "https://app.scrapingbee.com/api/v1/"
    # Adjust timeout based on JS rendering
    # ScrapingBee API timeout is in ms, client timeout in seconds
    api_timeout_ms = 25000 if render_js else 10000
    client_timeout_sec = (api_timeout_ms / 1000) + 10 # Give client slightly longer

    params = {
        "api_key": api_key,
        "url": url,
        "render_js": str(render_js).lower(),
        "block_ads": "true",
        "block_resources": "true", # Continue blocking resources
        "premium_proxy": "true", # Assume premium proxy is desired
        "timeout": str(api_timeout_ms),
        # Consider adding 'return_page_source': 'true' if needed
        # "country_code": "us", # Optional: specify country
    }

    try:
        print(f"Scraping URL: {url} (render_js: {render_js}, timeout: {client_timeout_sec}s)")
        # Use the shared client from the context
        response = await app_context.http_client.get(base_api_url, params=params, timeout=client_timeout_sec)
        response.raise_for_status() # Check for 4xx/5xx errors

        content_type = response.headers.get("content-type", "").lower()

        # Check if response is HTML
        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            html_content = response.text
            markdown_content = _clean_html_for_markdown(html_content, base_url=url)
            if not markdown_content:
                 return f"Scraped content from {url} resulted in empty text after cleaning."

            # Add source info safely
            source_info = f"\n\nSource: {url}"
            try:
                parsed_url = httpx.URL(url)
                if parsed_url.host:
                    source_info += f"\nDomain: {parsed_url.host}"
            except Exception as url_parse_e:
                 print(f"Could not parse domain from URL {url}: {url_parse_e}")

            return markdown_content + source_info

        elif "text/" in content_type or "application/json" in content_type:
             # Return plain text or JSON as is, maybe wrap in markdown code block
             return f"Received non-HTML content type '{content_type}' from {url}:\n```\n{response.text}\n```"
        else:
             # Handle binary or unknown types
            return f"Error: Received unsupported content type '{content_type}' from {url}. Cannot process."

    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            error_json = json.loads(error_detail)
            if isinstance(error_json, dict) and 'message' in error_json:
                error_detail = error_json['message']
        except json.JSONDecodeError:
             pass # Keep original text if not JSON
        print(f"Caught exception in scrape_url: {e}") # Added logging
        return (f"Error: ScrapingBee API request for {url} failed with status {e.response.status_code}. "
                f"Detail: {error_detail}")
    except httpx.TimeoutException as e: # Added variable e
         print(f"Caught exception in scrape_url: {e}") # Added logging
         return f"Error: Request to ScrapingBee timed out after {client_timeout_sec} seconds for URL: {url}"
    except httpx.RequestError as e:
        # Catch network errors, DNS errors, etc.
        print(f"Caught exception in scrape_url: {e}") # Added logging
        return f"Error: Network request to ScrapingBee failed for {url}: {e}"
    except ValueError as e: # Catch URL parsing errors etc.
         print(f"Caught exception in scrape_url: {e}") # Added logging
         return f"Error processing URL or parameters for {url}: {e}"
    except Exception as e:
        # Catch-all for unexpected errors during processing
        print(f"Caught exception in scrape_url: {e}") # Added logging
        return f"An unexpected error occurred during scraping {url}: {e}"


if __name__ == "__main__":
    print("Starting Scrape URL MCP Server...")
    mcp.run()
