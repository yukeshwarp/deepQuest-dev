from openai import AzureOpenAI
import os
from web_agent import search_bing  # Assuming you have a proper search function
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Tuple, Union
from typing_extensions import TypedDict
import streamlit as st
import operator
import redis
import hashlib
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from core import *
load_dotenv()

# Initialize Redis connection
redis_host = os.getenv("HOST_NAME")
redis_pass = os.getenv("PASSWORD")
redis_client = redis.Redis(
    host=redis_host,
    port=6379,
    password=redis_pass,
)

st.title("Agentic Research Assistant")
# Azure OpenAI client setup
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",
)

if "plan" not in st.session_state:
    st.session_state["plan"] = []

if "past_steps" not in st.session_state:
    st.session_state["past_steps"] = []

if "current_step_index" not in st.session_state:
    st.session_state["current_step_index"] = 0

with st.sidebar:
    st.header("📋 Current Research Plan")
    if st.session_state["plan"]:
        for idx, step in enumerate(st.session_state["plan"], 1):
            st.markdown(f"**Step {idx}:** {step}")
    else:
        st.write("Plan will appear here after planning.")

    st.subheader("✅ Executed Steps")
    if st.session_state["past_steps"]:
        for idx, (task, result) in enumerate(st.session_state["past_steps"], 1):
            st.markdown(f"**Step {idx}:** {task}\n\n_Result:_ {result}")
    else:
        st.write("No steps executed yet.")

# Helper functions for Redis and embeddings
def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a given text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def create_redis_index():
    """Create a Redisearch index for vector similarity search."""
    try:
        redis_client.execute_command("FT.CREATE", "embedding_index",
            "ON", "HASH",
            "PREFIX", "1", "embedding:",
            "SCHEMA",
            "vector", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "COSINE"
        )
    except redis.exceptions.ResponseError:
        # Index already exists
        pass

def store_embedding_in_redis(key: str, embedding: np.ndarray):
    """Store embedding in Redis for KNN search."""
    redis_client.hset(f"embedding:{key}", mapping={
        "vector": embedding.tobytes()
    })

def knn_search(query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """Perform KNN search using Redisearch."""
    query = f"*=>[KNN {top_k} @vector $vec AS score]"
    params = {"vec": query_embedding.tobytes()}
    try:
        result = redis_client.execute_command(
            "FT.SEARCH", "embedding_index", query,
            "PARAMS", "2", "vec", params["vec"],
            "SORTBY", "score", "ASC",
            "RETURN", "1", "vector"
        )
        return [(result[i], float(result[i + 1]["score"])) for i in range(1, len(result), 2)]
    except redis.exceptions.ResponseError as e:
        return []
    
def store_in_redis(key: str, value: str):
    """Store a value in Redis."""
    redis_client.set(key, value)

def retrieve_from_redis(key: str) -> Union[str, None]:
    """Retrieve a value from Redis."""
    value = redis_client.get(key)
    return value.decode() if value else None

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

# Initialize Redis index for KNN search
create_redis_index()

def search_and_store_web_results(task: str, result_text: str) -> str:
    """Search web, generate embedding, and store in Redis."""
    key = hash_text(task + result_text[:100])
    cached_result = retrieve_from_redis(key)
    if cached_result:
        return cached_result  # Return cached result if available

    # If not cached, perform search and store result
    web_result = search_bing(task + result_text[:100])
    store_in_redis(key, web_result)
    combined_text = f"{task}\n{result_text}\n{web_result}"
    embedding = generate_embedding(combined_text)
    store_embedding_in_redis(key, embedding)
    return web_result

# Modified wrapper functions to properly update session state
def wrapped_plan_step(state: dict) -> dict:
    """Wrapper around plan_step to update session state"""
    result = plan_step(state)
    if "plan" in result:
        st.session_state["plan"] = result["plan"]
    return result

def wrapped_execute_step(state: dict) -> dict:
    """Wrapper around execute_step to update session state"""
    result = execute_step(state)
    # Make sure we preserve the full plan in state
    if "plan" not in result and "plan" in state:
        result["plan"] = state["plan"]
    
    # Update past steps
    if "past_steps" in result:
        current_past_steps = st.session_state.get("past_steps", [])
        new_steps = result.get("past_steps", [])
        st.session_state["past_steps"] = current_past_steps + new_steps
    
    return result

def wrapped_replan_step(state: dict) -> dict:
    """Wrapper around replan_step to update session state"""
    # Create a copy of the state to modify before passing to replan_step
    replan_state = state.copy()
    
    # Mark current step as completed by advancing the index
    st.session_state["current_step_index"] += 1
    
    # Only remove the first plan item in our modified state, preserving the original
    if "plan" in replan_state and replan_state["plan"]:
        replan_state["plan"] = replan_state["plan"][1:]
    
    # Call the original replan_step
    result = replan_step(replan_state)
    
    # Update session state with new plan if provided
    if "plan" in result:
        st.session_state["plan"] = result["plan"]
    
    return result

# Workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", wrapped_plan_step)
workflow.add_node("agent", wrapped_execute_step)
workflow.add_node("replan", wrapped_replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])
app = workflow.compile()

# Inputs
inputs = {
    "input": "",
    "plan": [],
    "past_steps": [],
    "response": ""
}

def run_sync_app(Topic: str):
    # Reset session state for a new run
    st.session_state["plan"] = []
    st.session_state["past_steps"] = []
    st.session_state["current_step_index"] = 0
    
    state = {
        "input": Topic,
        "plan": [],
        "past_steps": [],
        "response": ""
    }
    
    planner_prompt = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Describe the qualities and domain expertise of a research assistant best suited for the following research topic:\n\n{Topic}\n\nRespond only with a short paragraph."}
        ]
    ).choices[0].message.content

    current_node = START
    result = {}

    while current_node != END:
        if current_node == START:
            current_node = "planner"
            result = wrapped_plan_step(state)
            
        elif current_node == "planner":
            state.update(result)
            current_node = "agent"
            result = wrapped_execute_step(state)
            
        elif current_node == "agent":
            # Update state with result from agent
            state.update(result)
            # Do NOT remove items from the plan here - that's handled in wrapped_replan_step
            current_node = "replan"
            result = wrapped_replan_step(state)
            
        elif current_node == "replan":
            state.update(result)
            current_node = should_end(state)

    st.write("\n✅ Final Output:\n")
    st.write(state["response"])

    # Perform KNN search for the final topic
    query_embedding = generate_embedding(Topic)
    similar_items = knn_search(query_embedding)
    st.subheader("🔍 Similar Topics Found:")
    for item_key, score in similar_items:
        st.markdown(f"- **Key:** {item_key}, **Similarity Score:** {score:.4f}")

if __name__ == "__main__":
    Topic = st.text_area("Research Topic")
    if st.button("Run Research"):
        if Topic.strip():  # Only run if there's valid input
            run_sync_app(Topic)
        else:
            st.warning("Please enter a research topic.")
