import os
import redis
import numpy as np
from dotenv import load_dotenv
from config import client
from typing import List, Tuple
load_dotenv()

redis_host = os.getenv("HOST_NAME")
redis_pass = os.getenv("PASSWORD")
redis_client = redis.Redis(
    host=redis_host,
    port=6379,
    password=redis_pass,
)

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