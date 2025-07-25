import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dotenv import load_dotenv
import os
import numpy as np
from ask_llm import ask_llm


load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def run_query():
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    # Load FAISS index
    index = faiss.read_index("outputs/vector_store/index.faiss")

    # Load chunk metadata
    with open("outputs/vector_store/meta.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Get user query
    query = input("Enter your query: ")

    # Embed query
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query])[0]

    # Search FAISS
    top_k = 5
    _, indices = index.search(np.array([query_embedding]), top_k)

    # Retrieve chunks
    retrieved_chunks = [metadata[i]["text"] for i in indices[0]]

    # Combine context
    context = "\n---\n".join(retrieved_chunks)

    # Ask LLM
    response = ask_llm(query, context)

    print("\nLLM Response:\n")
    print(response)
