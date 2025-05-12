import requests
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from newsapi import NewsApiClient
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import aiohttp
import asyncio
from crawl4ai import AsyncWebCrawler  # Importing the AsyncWebCrawler

# Load environment variables from .env
load_dotenv()

# API keys and headers
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

HEADERS = {
    "User-Agent": "MyApp/1.0 (contact@example.com)"  # Customize this with your contact
}

async def fetch_url(session, url):
    """Fetch the URL asynchronously and return the page content."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            else:
                return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def crawl_websites(urls):
    """Crawl the top 3 relevant websites from Google Search asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
        
        crawled_results = []
        for idx, content in enumerate(responses):
            if content:
                # Parse HTML content with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                title = soup.title.string if soup.title else 'No title found'
                description = soup.find('meta', attrs={'name': 'description'})
                description = description['content'] if description else 'No description found'
                crawled_results.append(f"[Crawled Website {idx + 1}] {title}\nDescription: {description}")
            else:
                crawled_results.append(f"[Crawled Website {idx + 1}] Error fetching content")

        return crawled_results

async def crawl_with_async_webcrawler(urls):
    """Use AsyncWebCrawler to crawl through URLs and return markdown content."""
    async with AsyncWebCrawler() as crawler:
        crawl_results = []
        for url in urls:
            try:
                result = await crawler.arun(url=url)  # Run the crawler
                crawl_results.append(f"[Crawled Website (Markdown)] URL: {url}\n{result.markdown}\n")
            except Exception as e:
                crawl_results.append(f"[Crawling Error] URL: {url} Error: {str(e)}")
        return crawl_results

def search_google(query):
    try:
        formatted_results = []

        # --- Google Custom Search ---
        google_search_url = "https://www.googleapis.com/customsearch/v1"
        google_params = {
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": 36,
        }
        response = requests.get(google_search_url, params=google_params)
        if response.status_code == 200:
            data = response.json()
            google_urls = []
            for i, item in enumerate(data.get("items", [])):
                formatted_results.append(
                    f"[Google Result {i + 1}] {item['title']} - {item['displayLink']}\n{item['snippet']}"
                )
                google_urls.append(item['link'])

            # --- Crawl the top 3 websites (asynchronously) using AsyncWebCrawler ---
            crawled_data = asyncio.run(crawl_with_async_webcrawler(google_urls[:3]))
            formatted_results.extend(crawled_data)

        else:
            formatted_results.append(f"Google Search Error: {response.status_code}")

        # --- ArXiv Search ---
        encoded_query = urllib.parse.quote(query)
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=3"
        req = urllib.request.Request(arxiv_url, headers=HEADERS)
        with urllib.request.urlopen(req) as response:
            xml_data = response.read().decode("utf-8")
            root = ET.fromstring(xml_data)
            ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('arxiv:entry', ns)
            for i, entry in enumerate(entries):
                title = entry.find('arxiv:title', ns)
                summary = entry.find('arxiv:summary', ns)
                title_text = title.text.strip() if title is not None else "No title"
                summary_text = summary.text.strip()[:300] + "..." if summary is not None else "No summary"
                formatted_results.append(f"[ArXiv Result {i + 1}] {title_text}\nSummary: {summary_text}")

        # --- NewsAPI Search ---
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
        for i, article in enumerate(articles.get("articles", [])):
            formatted_results.append(
                f"[News {i + 1}] {article['title']} ({article['source']['name']})\n{article['description']}\nURL: {article['url']}"
            )

        # --- SEC API (General Data Search) ---
        sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={urllib.parse.quote(query)}&action=getcompany"
        sec_response = requests.get(sec_url, headers=HEADERS)
        
        if sec_response.status_code == 200:
            # Check for data availability, such as the latest filings or company overview
            if "No matching companies" in sec_response.text:
                formatted_results.append(f"SEC API: No filings found for '{query}'.")
            else:
                # Extract links or a summary of the filings
                formatted_results.append(f"SEC API: Filings and data retrieved for {query}. Check SEC's website for details.")
        else:
            formatted_results.append(f"SEC API Error: {sec_response.status_code} - Unable to retrieve data from SEC.")

        # --- Wikipedia Extract ---
        wikipedia_url = "https://en.wikipedia.org/w/api.php"
        wiki_params = {
            "action": "query",
            "prop": "extracts",
            "titles": query,
            "format": "json",
            "exintro": True,
            "explaintext": True
        }
        wiki_response = requests.get(wikipedia_url, params=wiki_params)
        if wiki_response.status_code == 200:
            wiki_data = wiki_response.json()
            pages = wiki_data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                extract = page.get("extract")
                if extract:
                    formatted_results.append(f"[Wikipedia]\n{extract}")
        else:
            formatted_results.append(f"Wikipedia Error: {wiki_response.status_code}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"