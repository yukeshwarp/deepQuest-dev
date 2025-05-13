import streamlit as st
from web_agent import search_bing
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import redis
import hashlib
from typing import Union, List, Tuple
import numpy as np
from scipy.spatial.distance import cosine

load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-01-preview",
)

st.title("deepQuest")

# Collect research inputs
research_topic = st.text_input("Research Topic", "Enter your research topic here")
description = st.text_area("Description", "Enter a description of your research topic here")
redis_host = os.getenv("HOST_NAME")
redis_pass = os.getenv("PASSWORD")
redis_client = redis.Redis(
    host=redis_host,
    port=6379,
    password=redis_pass,
)

# Helper functions for embeddings and Redis
def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a given text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def store_embedding_in_redis(key: str, embedding: np.ndarray):
    """Store embedding in Redis."""
    redis_client.set(key, embedding.tobytes())

def retrieve_all_embeddings() -> dict:
    """Retrieve all embeddings from Redis."""
    keys = redis_client.keys()
    embeddings = {}
    for key in keys:
        embedding_bytes = redis_client.get(key)
        embeddings[key.decode()] = np.frombuffer(embedding_bytes, dtype=np.float32)
    return embeddings

def similarity_search(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Perform similarity search for a query."""
    query_embedding = generate_embedding(query)
    all_embeddings = retrieve_all_embeddings()

    # Compute cosine similarity
    similarities = []
    for key, embedding in all_embeddings.items():
        similarity = 1 - cosine(query_embedding, embedding)
        similarities.append((key, similarity))

    # Sort by similarity score (descending) and return top_k results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def hash_text(text: str) -> str:
    """Generate a unique hash for the text."""
    return hashlib.sha256(text.encode()).hexdigest()

# Example usage in your workflow
def search_and_store_web_results(task: str, result_text: str):
    """Search web, generate embedding, and store in Redis."""
    web_result = search_bing(task + result_text[:100])  # Get web result
    combined_text = f"{task}\n{result_text}\n{web_result}"
    embedding = generate_embedding(combined_text)
    key = hash_text(combined_text)
    store_embedding_in_redis(key, embedding)
    return web_result

# Process when topic is submitted
if st.button("Generate Report") and research_topic:
    st.subheader("Step 1: Planning Report Structure")
    planning_prompt = f"""
    Based on the following inputs:
    - Topic: {research_topic}
    - Description: {description}

    Propose a report structure with an ordered list of logical sections. Each section should build on the previous one, culminating in a synthesized conclusion.
    """

    plan_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert research planner."},
            {"role": "user", "content": planning_prompt},
        ]
    )
    section_plan = plan_response.choices[0].message.content
    st.markdown(f"**Planned Sections:**\n{section_plan}")

    st.subheader("Step 2: Dynamic Section Synthesis and Writing")
    sections = [line.strip("- ").strip() for line in section_plan.strip().split("\n") if line.strip()]
    final_report = ""
    context_memory = ""

    for idx, section in enumerate(sections):
        st.markdown(f"### {section}")

        search_query = f"{section} {research_topic}"
        search_results = search_bing(search_query)

        # Fixing TypeError by ensuring proper result parsing
        if isinstance(search_results, list) and all(isinstance(result, dict) for result in search_results):
            research_context = "\n".join([result.get('snippet', '') for result in search_results[:5]])
        else:
            research_context = "No valid search results found."

        dynamic_prompt = f"""
        You are a research assistant. Use the context below and the ongoing research to write the section titled "{section}".

        Previous Context:
        {context_memory}

        New Web Context:
        {research_context}
        """

        section_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant writing a report section based on ongoing synthesis."},
                {"role": "user", "content": dynamic_prompt},
            ]
        )

        section_content = section_response.choices[0].message.content
        st.markdown(section_content)

        # Store embeddings for similarity search
        combined_text = f"{section}\n{section_content}"
        key = hash_text(combined_text)
        embedding = generate_embedding(combined_text)
        store_embedding_in_redis(key, embedding)

        # Perform similarity search
        similar_items = similarity_search(section)
        st.markdown("**Similar Sections:**")
        for item_key, score in similar_items:
            st.markdown(f"- Key: {item_key}, Similarity: {score:.4f}")

        final_report += f"\n\n## {section}\n{section_content}"
        context_memory += f"\n\n{section_content}"

    st.subheader("Final Compiled Report")
    st.markdown(final_report)
    st.download_button("Download Report", data=final_report, file_name="research_report.md")