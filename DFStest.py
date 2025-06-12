import streamlit as st
from DFS_Deep_web_agent import *
from dotenv import load_dotenv
from config import client

load_dotenv()


def execute_step(step, context):
    """Execute a single research step using function calling and web search."""
    exec_prompt = (
        f"You are an autonomous research agent. Execute the following research step:\n\n"
        f"Step: {step}\n\n"
        f"Context so far: {context}\n\n"
        "Include even the most minor details in your response. "
        "Always search over the internet regarding the relevant details and include content from that, use the search_google function."
    )
    functions = [
        {
            "name": "search_google_api",
            "description": "Searches Google and returns relevant web results for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Google.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_arxiv_api",
            "description": "Searches ArXiv and returns relevant results for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for ArXiv.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_newsapi_api",
            "description": "Searches NewsAPI and returns relevant news articles for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for NewsAPI.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_sec_api",
            "description": "Searches SEC and returns relevant filings for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for SEC.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_wikipedia_api",
            "description": "Searches Wikipedia and returns relevant extracts for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Wikipedia.",
                    }
                },
                "required": ["query"],
            },
        },
    ]
    messages = [
        {"role": "system", "content": "You are a research execution agent."},
        {"role": "user", "content": exec_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-4.1", messages=messages, functions=functions, function_call="auto"
    )
    msg = response.choices[0].message

    import json
    if msg.function_call:
        fn_name = msg.function_call.name
        search_args = json.loads(msg.function_call.arguments)
        if fn_name == "search_google_api":
            web_results = search_google_api(search_args["query"])
        elif fn_name == "search_arxiv_api":
            web_results = search_arxiv_api(search_args["query"])
        elif fn_name == "search_newsapi_api":
            web_results = search_newsapi_api(search_args["query"])
        elif fn_name == "search_sec_api":
            web_results = search_sec_api(search_args["query"])
        elif fn_name == "search_wikipedia_api":
            web_results = search_wikipedia_api(search_args["query"])
        else:
            web_results = "[Function not implemented]"
        messages.append(
            {"role": "function", "name": fn_name, "content": web_results}
        )
        response2 = client.chat.completions.create(model="gpt-4.1", messages=messages)
        return response2.choices[0].message.content
    else:
        return msg.content
