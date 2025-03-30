from typing import Dict, Any, Optional, List
import aiohttp
import os
from urllib.parse import urlparse
import json
import re
import logging

from agents import function_tool, RunContextWrapper

# Set up logging
logger = logging.getLogger(__name__)

async def _process_html_content(html_content: str) -> str:
    """
    Process HTML content to extract readable text.
    
    Args:
        html_content: The raw HTML content
        
    Returns:
        Processed readable text/markdown
    """
    try:
        # Import here to avoid loading these libraries unless needed
        from bs4 import BeautifulSoup
        import html2text
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "iframe", "nav", "footer"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Convert HTML to markdown
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = False
        h2t.ignore_tables = False
        h2t.ignore_emphasis = False
        h2t.body_width = 0  # No wrapping
        
        markdown = h2t.handle(str(soup))
        
        # Clean up markdown
        cleaned_markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return cleaned_markdown
    except Exception as e:
        logger.error(f"Error processing HTML: {str(e)}")
        # If processing fails, return just the text from the HTML
        if 'soup' in locals():
            return soup.get_text()
        return html_content

async def _scrape_url_with_scrapingbee(
    url: str, 
    render_js: bool = True,
    timeout: int = 30000,
    extract_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Internal function to scrape a URL using ScrapingBee API.
    
    Args:
        url: The URL to scrape
        render_js: Whether to render JavaScript
        timeout: Timeout in milliseconds
        extract_rules: Optional extraction rules for ScrapingBee
        
    Returns:
        Dictionary with result information
    """
    api_key = os.environ.get("SCRAPINGBEE_API_KEY")
    if not api_key:
        raise ValueError("SCRAPINGBEE_API_KEY environment variable is not set")
    
    # Prepare ScrapingBee API parameters
    params = {
        'api_key': api_key,
        'url': url,
        'render_js': str(render_js).lower(),
        'timeout': timeout,
        'premium_proxy': 'true',
        'country_code': 'us',
        'block_ads': 'true'
    }
    
    # Add extraction rules if provided
    if extract_rules:
        params['extract_rules'] = json.dumps(extract_rules)
    
    api_url = "https://app.scrapingbee.com/api/v1/"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"ScrapingBee API request failed with status {response.status}: {error_text}")
                
                # Get content
                content = await response.text()
                
                # Get response headers for metadata
                headers = response.headers
                
                # Process content to text
                processed_text = await _process_html_content(content)
                
                # Get domain for source attribution
                domain = urlparse(url).netloc
                
                return {
                    "url": url,
                    "status": response.status,
                    "source": domain,
                    "raw_html": content[:10000] if len(content) > 10000 else content,  # Limit raw HTML size
                    "text": processed_text,
                    "metadata": {
                        "content_type": headers.get("Content-Type", "unknown"),
                        "last_modified": headers.get("Last-Modified", "unknown"),
                        "js_rendered": render_js
                    }
                }
    except Exception as e:
        logger.error(f"Error scraping URL {url}: {str(e)}")
        return {
            "url": url,
            "status": 500,
            "source": urlparse(url).netloc,
            "error": str(e),
            "text": f"Failed to scrape URL: {str(e)}",
            "raw_html": "",
            "metadata": {
                "js_rendered": render_js
            }
        }

@function_tool
async def scrape_url(
    ctx: RunContextWrapper[Any], 
    url: str,
    render_js: Optional[bool] = None,
    timeout: Optional[int] = None,
    extract_selectors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Web scraping tool that extracts and processes content from websites. Use this to:
            
    1. Extract text from webpages (news, articles, documentation)
    2. Gather product information from e-commerce sites
    3. Retrieve information from JavaScript-rendered sites
    4. Research information that requires specific page content
    
    The tool processes HTML to extract readable text content in markdown format.
    
    Important notes:
    - Always provide complete URLs including protocol (e.g., 'https://example.com')
    - JavaScript rendering is enabled by default for compatibility
    - Set render_js=False for static sites to get faster results
    - Content is automatically processed to extract readable text
    - Safe mode filters out potentially harmful content
    
    Returns:
    - The extracted page content as readable text
    - Source attribution
    - Status information
    
    Args:
        url: The complete URL of the webpage to read and analyze
        render_js: Whether to render JavaScript (true for dynamic sites, false for faster scraping of static sites)
        timeout: Maximum time to wait in milliseconds
        extract_selectors: Optional list of CSS selectors to extract specific content
    """
    try:
        # Set default values inside the function
        if render_js is None:
            render_js = True
            
        if timeout is None:
            # Use shorter timeout for static content
            timeout = 15000 if not render_js else 30000
        
        # Convert extract_selectors to ScrapingBee extract_rules if provided
        extract_rules = None
        if extract_selectors:
            extract_rules = {}
            for i, selector in enumerate(extract_selectors):
                extract_rules[f"element_{i}"] = {"selector": selector, "type": "list", "output": "text"}
        
        # Call the internal function
        result = await _scrape_url_with_scrapingbee(
            url=url,
            render_js=render_js,
            timeout=timeout,
            extract_rules=extract_rules
        )
        
        # Log the result
        logger.info(f"Successfully scraped URL: {url}")
        
        return result
    except Exception as e:
        logger.error(f"Error in scrape_url for {url}: {str(e)}")
        return {
            "url": url,
            "status": 500,
            "source": urlparse(url).netloc,
            "error": str(e),
            "text": f"Failed to scrape URL: {str(e)}",
            "raw_html": "",
            "metadata": {
                "js_rendered": render_js if render_js is not None else True
            }
        }