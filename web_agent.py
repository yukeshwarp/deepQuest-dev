import requests
import os

# Bing Search API configuration
BING_API_KEY = os.getenv("BING_API_KEY")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT")

def search_bing(query):
    try:
        """
        Function to search the web using Azure Bing Search API when the user requests it.
        """
        headers = {
            "Ocp-Apim-Subscription-Key": BING_API_KEY
        }

        params = {
            "q": query,
            "textDecorations": True,
            "textFormat": "HTML",
        }

        response = requests.get(BING_SEARCH_ENDPOINT, headers=headers, params=params)
        if response.status_code == 200:
            search_results = response.json()
            # Extract top search result titles and URLs
            if 'webPages' in search_results:
                results = search_results['webPages']['value']
                top_result = results[0]
                return f"Here is a top result I found: {top_result['name']} - {top_result['url']}"
            else:
                return "Sorry, I couldn't find any relevant search results."
        else:
            return f"Error during search: {response.status_code}"
    except Exception as e:
        print(e)
        return f"Error: {str(e)}"
