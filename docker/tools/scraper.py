#!/usr/bin/env python3
"""
Web scraper helper for web container
Used by selenium_browse and playwright_browse tools
"""

import sys
import json

def scrape_with_requests(url: str, selector: str = None):
    """Simple scraper using requests + beautifulsoup"""
    import requests
    from bs4 import BeautifulSoup
    
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        if selector:
            elements = soup.select(selector)
            return json.dumps({
                'url': url,
                'status': response.status_code,
                'selector': selector,
                'matches': len(elements),
                'content': [el.get_text(strip=True) for el in elements[:10]]
            }, indent=2)
        else:
            # Return page title and text content
            title = soup.title.string if soup.title else 'No title'
            text = soup.get_text(separator=' ', strip=True)[:5000]
            return json.dumps({
                'url': url,
                'status': response.status_code,
                'title': title,
                'content': text
            }, indent=2)
            
    except Exception as e:
        return json.dumps({'error': str(e), 'url': url})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: scraper.py <url> [selector]")
        sys.exit(1)
    
    url = sys.argv[1]
    selector = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(scrape_with_requests(url, selector))
