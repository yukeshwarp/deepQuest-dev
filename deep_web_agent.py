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
import logging
from crawl4ai import AsyncWebCrawler
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables from .env
load_dotenv()

# API keys and headers
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

HEADERS = {
    "User-Agent": "MyApp/1.0 (contact@example.com)"  # Customize this with your contact
}

# --- Retry Decorator ---
def retry_on_exception(max_retries=2, backoff=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logging.warning(
                        f"Attempt {attempt+1} failed for {func.__name__}: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(backoff)
            logging.error(f"All retries failed for {func.__name__}: {last_exception}")
            raise last_exception
        return wrapper
    return decorator

# --- Asynchronous Retry Helper ---
async def async_retry_on_exception(func, *args, max_retries=2, backoff=2, **kwargs):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logging.warning(
                f"Async attempt {attempt+1} failed for {func.__name__}: {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)
    logging.error(f"All async retries failed for {func.__name__}: {last_exception}")
    raise last_exception

# --- Asynchronous Utilities ---

async def fetch_url(session, url, timeout=10):
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                return await response.text()
            else:
                logging.warning(f"Non-200 response for {url}: {response.status}")
                return None
    except asyncio.TimeoutError:
        logging.error(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

async def crawl_websites(urls, timeout=10):
    crawled_results = []
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                async_retry_on_exception(fetch_url, session, url, timeout=timeout)
                for url in urls
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, content in enumerate(responses):
                if isinstance(content, Exception):
                    logging.error(f"Exception during crawling {urls[idx]}: {content}")
                    crawled_results.append(
                        f"[Crawled Website {idx + 1}] Error fetching content: {content}"
                    )
                elif content:
                    soup = BeautifulSoup(content, "html.parser")
                    title = soup.title.string if soup.title else "No title found"
                    description = soup.find("meta", attrs={"name": "description"})
                    description = (
                        description["content"]
                        if description
                        else "No description found"
                    )
                    crawled_results.append(
                        f"[Crawled Website {idx + 1}] {title}\nDescription: {description}"
                    )
                else:
                    crawled_results.append(
                        f"[Crawled Website {idx + 1}] Error fetching content"
                    )
    except Exception as e:
        logging.error(f"Error in crawl_websites: {e}")
    return crawled_results

async def crawl_with_async_webcrawler(urls, timeout=20):
    crawl_results = []
    try:
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await asyncio.wait_for(
                        async_retry_on_exception(
                            crawler.arun, url=url, max_retries=2, backoff=2
                        ),
                        timeout=timeout,
                    )
                    crawl_results.append(
                        f"[Crawled Website (Markdown)] URL: {url}\n{result.markdown}\n"
                    )
                except asyncio.TimeoutError:
                    logging.error(f"Timeout crawling {url} with AsyncWebCrawler")
                    crawl_results.append(f"[Crawling Error] URL: {url} Error: Timeout")
                except Exception as e:
                    logging.error(f"Error crawling {url} with AsyncWebCrawler: {e}")
                    crawl_results.append(f"[Crawling Error] URL: {url} Error: {str(e)}")
    except Exception as e:
        logging.error(f"Error initializing AsyncWebCrawler: {e}")
    return crawl_results

# --- Synchronous Main Search Function ---

@retry_on_exception(max_retries=2, backoff=2)
def google_search_api_call(google_search_url, google_params):
    try:
        response = requests.get(google_search_url, params=google_params, timeout=15)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.error(f"Error in google_search_api_call: {e}")
        raise

@retry_on_exception(max_retries=2, backoff=2)
def arxiv_api_call(arxiv_url):
    try:
        req = urllib.request.Request(arxiv_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logging.error(f"Error in arxiv_api_call: {e}")
        raise

@retry_on_exception(max_retries=2, backoff=2)
def sec_api_call(sec_url):
    try:
        return requests.get(sec_url, headers=HEADERS, timeout=15)
    except Exception as e:
        logging.error(f"Error in sec_api_call: {e}")
        raise

@retry_on_exception(max_retries=2, backoff=2)
def wikipedia_api_call(wikipedia_url, wiki_params):
    try:
        return requests.get(wikipedia_url, params=wiki_params, timeout=10)
    except Exception as e:
        logging.error(f"Error in wikipedia_api_call: {e}")
        raise

@retry_on_exception(max_retries=2, backoff=2)
def newsapi_call(newsapi, query):
    try:
        return newsapi.get_everything(
            q=query, language="en", sort_by="relevancy", page_size=5
        )
    except Exception as e:
        logging.error(f"Error in newsapi_call: {e}")
        raise

def google_search(query):
    google_search_url = "https://www.googleapis.com/customsearch/v1"
    google_params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }
    try:
        response = google_search_api_call(google_search_url, google_params)
        data = response.json()
        google_urls = []
        formatted_results = []
        for i, item in enumerate(data.get("items", [])):
            formatted_results.append(
                f"[Google Result {i + 1}] {item['title']} - {item['displayLink']}\n{item['snippet']}"
            )
            google_urls.append(item["link"])
        return formatted_results, google_urls
    except Exception as e:
        logging.error(f"Google Search Error: {e}")
        return [f"Google Search Error: {str(e)}"], []


def arxiv_search(query):
    try:
        encoded_query = urllib.parse.quote(query)
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=3"
        xml_data = arxiv_api_call(arxiv_url)
        root = ET.fromstring(xml_data)
        ns = {"arxiv": "http://www.w3.org/2005/Atom"}
        entries = root.findall("arxiv:entry", ns)
        results = []
        for i, entry in enumerate(entries):
            title = entry.find("arxiv:title", ns)
            summary = entry.find("arxiv:summary", ns)
            title_text = title.text.strip() if title is not None else "No title"
            summary_text = (
                summary.text.strip()[:300] + "..."
                if summary is not None
                else "No summary"
            )
            results.append(
                f"[ArXiv Result {i + 1}] {title_text}\nSummary: {summary_text}"
            )
        return results
    except Exception as e:
        logging.error(f"ArXiv Search Error: {e}")
        return [f"ArXiv Search Error: {str(e)}"]


def newsapi_search(query):
    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        articles = newsapi_call(newsapi, query)
        results = []
        for i, article in enumerate(articles.get("articles", [])):
            results.append(
                f"[News {i + 1}] {article['title']} ({article['source']['name']})\n{article['description']}\nURL: {article['url']}"
            )
        return results
    except Exception as e:
        logging.error(f"NewsAPI Error: {e}")
        return [f"NewsAPI Error: {str(e)}"]


def sec_search(query):
    try:
        sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={urllib.parse.quote(query)}&action=getcompany"
        sec_response = sec_api_call(sec_url)
        if sec_response.status_code == 200:
            if "No matching companies" in sec_response.text:
                return [f"SEC API: No filings found for '{query}'."]
            else:
                return [f"SEC API: Filings and data retrieved for {query}. Check SEC's website for details."]
        else:
            return [f"SEC API Error: {sec_response.status_code} - Unable to retrieve data from SEC."]
    except Exception as e:
        logging.error(f"SEC API Error: {e}")
        return [f"SEC API Error: {str(e)}"]


def wikipedia_extract(query):
    try:
        wikipedia_url = "https://en.wikipedia.org/w/api.php"
        wiki_params = {
            "action": "query",
            "prop": "extracts",
            "titles": query,
            "format": "json",
            "exintro": True,
            "explaintext": True,
        }
        wiki_response = wikipedia_api_call(wikipedia_url, wiki_params)
        if wiki_response.status_code == 200:
            wiki_data = wiki_response.json()
            pages = wiki_data.get("query", {}).get("pages", {})
            results = []
            for _, page in pages.items():
                extract = page.get("extract")
                if extract:
                    results.append(f"[Wikipedia]\n{extract}")
            return results
        else:
            return [f"Wikipedia Error: {wiki_response.status_code}"]
    except Exception as e:
        logging.error(f"Wikipedia Error: {e}")
        return [f"Wikipedia Error: {str(e)}"]


def search_google(query):
    try:
        formatted_results, google_urls = google_search(query)
        crawled_data = []
        # Only crawl Google results using deep webcrawler with depth=2
        if google_urls:
            try:
                async def deep_crawl(urls):
                    async with AsyncWebCrawler(depth=2) as crawler:
                        results = []
                        for url in urls[:3]:
                            try:
                                result = await asyncio.wait_for(
                                    async_retry_on_exception(
                                        crawler.arun, url=url, max_retries=2, backoff=2
                                    ),
                                    timeout=20,
                                )
                                results.append(
                                    f"[Crawled Website (Markdown)] URL: {url}\n{result.markdown}\n"
                                )
                            except asyncio.TimeoutError:
                                logging.error(f"Timeout crawling {url} with AsyncWebCrawler")
                                results.append(f"[Crawling Error] URL: {url} Error: Timeout")
                            except Exception as e:
                                logging.error(f"Error crawling {url} with AsyncWebCrawler: {e}")
                                results.append(f"[Crawling Error] URL: {url} Error: {str(e)}")
                        return results
                crawled_data = asyncio.run(deep_crawl(google_urls))
                logging.info(f"Deep crawled URLs: {google_urls[:3]}")
            except Exception as e:
                logging.error(f"Error running deep async crawler: {e}")
                crawled_data.append(f"Async Crawler Error: {str(e)}")
        arxiv_results = arxiv_search(query)
        newsapi_results = newsapi_search(query)
        sec_results = sec_search(query)
        wiki_results = wikipedia_extract(query)
        all_results = formatted_results + crawled_data + arxiv_results + newsapi_results + sec_results + wiki_results
        all_results = [r for r in all_results if r and r.strip()]
        return "\n\n".join(all_results)
    except Exception as e:
        logging.critical(f"Unexpected error occurred in search_google: {e}")
        return "An unexpected error occurred. Please try again later."

def search_google_api(query):
    """Searches Google and returns relevant web results for a query."""
    formatted_results, google_urls = google_search(query)
    return "\n\n".join(formatted_results)

def search_arxiv_api(query):
    """Searches ArXiv and returns relevant results for a query."""
    arxiv_results = arxiv_search(query)
    return "\n\n".join(arxiv_results)

def search_newsapi_api(query):
    """Searches NewsAPI and returns relevant news articles for a query."""
    newsapi_results = newsapi_search(query)
    return "\n\n".join(newsapi_results)

def search_sec_api(query):
    """Searches SEC and returns relevant filings for a query."""
    sec_results = sec_search(query)
    return "\n\n".join(sec_results)

def search_wikipedia_api(query):
    """Searches Wikipedia and returns relevant extracts for a query."""
    wiki_results = wikipedia_extract(query)
    return "\n\n".join(wiki_results)