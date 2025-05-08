import requests
import os
import urllib, urllib.request
from newsapi import NewsApiClient
from dotenv import load_dotenv
load_dotenv()

BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT")
BING_API_KEY = os.getenv("BING_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")

def search_bing(query):
    try:
        headers = {
            "Ocp-Apim-Subscription-Key": BING_API_KEY
        }

        total_results = []
        count_per_request = 50  # Maximum per request supported by Bing
        total_to_fetch = 100    # Total number of results to retrieve

        for offset in range(0, total_to_fetch, count_per_request):
            params = {
                "q": query,
                "textDecorations": True,
                "textFormat": "HTML",
                "count": count_per_request,
                "offset": offset
            }

            response = requests.get(BING_SEARCH_ENDPOINT, headers=headers, params=params)
            if response.status_code == 200:
                search_results = response.json()
                if 'webPages' in search_results:
                    results = search_results['webPages']['value']
                    total_results.extend(results)
                else:
                    break  # No more results found
            else:
                return f"Error during search: {response.status_code}"

        if not total_results:
            return "Sorry, I couldn't find any relevant search results."

        # Format Bing search results
        formatted_results = [
            f"{i+1}. {item['name']} - {item['url']} - {item['snippet']}"
            for i, item in enumerate(total_results[:total_to_fetch])
        ]

        # Fetch results from ArXiv API
        url = f'http://export.arxiv.org/api/query?query=all:{query}&start=0&max_results=3'
        data = urllib.request.urlopen(url)
        formatted_results.append(f"Arxiv API response: {data.read().decode('utf-8')}")

        # Fetch results from News API
        newsapi = NewsApiClient(api_key=newsapi_key)
        all_articles = newsapi.get_everything(q=query,
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=5)

        result = ""
        for i in range(0, 5):
            result += f"Source: {all_articles['articles'][i]['source']['name']}\n"
            result += f"Title: {all_articles['articles'][i]['title']}\n"
            result += f"Description: {all_articles['articles'][i]['description']}\n"
            result += f"URL: {all_articles['articles'][i]['url']}\n\n"

        formatted_results.append(f"News API response: {result}")

        # Fetch results from SEC API
        sec_api_url = f"https://data.sec.gov/api/xbrl/companyfacts/{query}.json"
        sec_headers = {
            "User-Agent": "YourAppName (your_email@example.com)"
        }
        sec_response = requests.get(sec_api_url, headers=sec_headers)
        if sec_response.status_code == 200:
            sec_data = sec_response.json()
            formatted_results.append(f"SEC API response: {sec_data}")
        else:
            formatted_results.append(f"SEC API response: Error {sec_response.status_code}")

        # Fetch results from Wikipedia API
        wikipedia_api_url = "https://en.wikipedia.org/w/api.php"
        wikipedia_params = {
            "action": "query",
            "prop": "extracts",
            "titles": query,
            "format": "json",
            "exintro": True,
            "explaintext": True
        }
        wikipedia_response = requests.get(wikipedia_api_url, params=wikipedia_params)
        if wikipedia_response.status_code == 200:
            wikipedia_data = wikipedia_response.json()
            pages = wikipedia_data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if "extract" in page_info:
                    formatted_results.append(f"Wikipedia API response: {page_info['extract']}")
        else:
            formatted_results.append(f"Wikipedia API response: Error {wikipedia_response.status_code}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        print(e)
        return f"Error: {str(e)}"