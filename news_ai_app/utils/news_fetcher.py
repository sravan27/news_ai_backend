"""
Utilities for fetching news content from URLs.
"""
import logging
import re
from typing import Dict, Optional

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from news_ai_app.config import settings

logger = logging.getLogger(__name__)


async def fetch_news_from_url(url: str) -> Dict[str, str]:
    """
    Fetch and extract news content from a URL.
    
    Args:
        url: URL of the news article.
        
    Returns:
        Dictionary containing title and text of the article.
    """
    try:
        # Make HTTP request
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.title.string.strip() if soup.title else ""
            
            # Try to extract main article content based on common patterns
            article_content = extract_article_content(soup)
            
            return {
                "title": title,
                "text": article_content,
                "url": url
            }
            
    except httpx.RequestError as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return {"title": "", "text": "", "url": url}
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return {"title": "", "text": "", "url": url}


def extract_article_content(soup: BeautifulSoup) -> str:
    """
    Extract the main article content from a BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup object of the page HTML.
        
    Returns:
        Extracted article text.
    """
    # Try different common article content patterns
    
    # 1. Look for article tag
    article = soup.find("article")
    if article:
        return clean_text(article.get_text())
    
    # 2. Look for main content div with common class names
    content_classes = [
        "article-content", "post-content", "entry-content", "content-body",
        "article-body", "story-body", "story-content", "main-content"
    ]
    
    for class_name in content_classes:
        content = soup.find(class_=re.compile(class_name, re.I))
        if content:
            return clean_text(content.get_text())
    
    # 3. Look for main tag
    main = soup.find("main")
    if main:
        return clean_text(main.get_text())
    
    # 4. Fall back to extracting all paragraphs within the body
    paragraphs = []
    for p in soup.find_all("p"):
        # Filter out short paragraphs that are likely navigation, ads, etc.
        text = p.get_text().strip()
        if len(text) > 40:  # Minimal length for a valid paragraph
            paragraphs.append(text)
    
    if paragraphs:
        return "\n\n".join(paragraphs)
    
    # 5. Last resort: just get the body text
    body = soup.find("body")
    if body:
        return clean_text(body.get_text())
    
    # 6. If everything fails, return empty string
    return ""


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excess whitespace, etc.
    
    Args:
        text: Text to clean.
        
    Returns:
        Cleaned text.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple newlines with a double newline
    text = re.sub(r'\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


async def fetch_news_from_api(query: Optional[str] = None, category: Optional[str] = None) -> Dict:
    """
    Fetch news articles from News API.
    
    Args:
        query: Search query for articles.
        category: Category to filter articles by.
        
    Returns:
        Dictionary containing news API response.
    """
    try:
        # Build parameters
        params = {
            "apiKey": settings.api_keys.news_api_key,
            "language": "en",
            "pageSize": 20,
        }
        
        if query:
            params["q"] = query
            
        if category:
            params["category"] = category
        
        # Make API request
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://newsapi.org/v2/top-headlines" if not query else "https://newsapi.org/v2/everything",
                params=params
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"Error fetching news API: {e}")
        return {"status": "error", "articles": []}
    except Exception as e:
        logger.error(f"Error processing news API response: {e}")
        return {"status": "error", "articles": []}