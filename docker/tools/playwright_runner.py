#!/usr/bin/env python3
"""
Playwright browser automation helper
"""

import sys
import json
import asyncio

async def browse_with_playwright(url: str):
    """Browse URL with headless browser via Playwright"""
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            page = await browser.new_page(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            try:
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                title = await page.title()
                content = await page.content()
                
                # Take screenshot
                screenshot_path = '/web/screenshot.png'
                await page.screenshot(path=screenshot_path, full_page=False)
                
                # Get text content
                text = await page.inner_text('body')
                
                return json.dumps({
                    'url': page.url,
                    'title': title,
                    'screenshot': screenshot_path,
                    'content_length': len(content),
                    'text_preview': text[:2000] if text else ''
                }, indent=2)
                
            finally:
                await browser.close()
                
    except Exception as e:
        return json.dumps({'error': str(e), 'url': url})


def browse(url: str):
    """Synchronous wrapper"""
    return asyncio.run(browse_with_playwright(url))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: playwright_runner.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(browse(url))
