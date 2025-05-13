from web_agent import search_google
from core import rewrite_query
data = search_google(rewrite_query("Can you tell me the parent company of BMAC WH 1 LLC and its corporate structure"))
print(data)