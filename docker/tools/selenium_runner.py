#!/usr/bin/env python3
"""
Selenium browser automation helper
"""

import sys
import json
import os

def browse_with_selenium(url: str):
    """Browse URL with headless Chrome via Selenium"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # Chrome options for headless
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get(url)
            
            # Wait for page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page info
            title = driver.title
            current_url = driver.current_url
            page_source = driver.page_source[:10000]
            
            # Take screenshot
            screenshot_path = '/web/screenshot.png'
            driver.save_screenshot(screenshot_path)
            
            return json.dumps({
                'url': current_url,
                'title': title,
                'screenshot': screenshot_path,
                'content_length': len(page_source),
                'content_preview': page_source[:2000]
            }, indent=2)
            
        finally:
            driver.quit()
            
    except Exception as e:
        return json.dumps({'error': str(e), 'url': url})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: selenium_runner.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(browse_with_selenium(url))
