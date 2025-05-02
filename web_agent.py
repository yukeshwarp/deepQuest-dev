import requests
import os

BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_API_KEY = "afc632c209584311956e12239969f498"


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

        # Return the top 100 search result titles and URLs
        formatted_results = [
            f"{i+1}. {item['name']} - {item['url']} - {item['snippet']}"
            for i, item in enumerate(total_results[:total_to_fetch])
        ]
        return "\n\n".join(formatted_results)

    except Exception as e:
        print(e)
        return f"Error: {str(e)}"
    
# print(search_bing("Python programming"))