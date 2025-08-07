import requests
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
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

    # Use the same model as in embedding for consistency
    embedder = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = embedder.encode([query])[0]

    # Enhanced retrieval with more chunks and re-ranking
    top_k = 15  # Increased from 5 for better coverage
    _, indices = index.search(np.array([query_embedding]), top_k)

    # Retrieve chunks with metadata
    retrieved_chunks = []
    for i in indices[0]:
        chunk_info = metadata[i]
        retrieved_chunks.append({
            'text': chunk_info["text"],
            'doc': chunk_info["doc"],
            'page': chunk_info["page"],
            'chunk_index': chunk_info["chunk_index"],
            'token_count': chunk_info.get("token_count", 0)
        })

    # Re-rank chunks using a cross-encoder for better relevance
    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)
        
        # Sort chunks by relevance score
        chunk_scores = list(zip(retrieved_chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        retrieved_chunks = [chunk for chunk, score in chunk_scores]
        
        print(f"🔍 Retrieved {len(retrieved_chunks)} chunks, re-ranked by relevance")
    except Exception as e:
        print(f"⚠️ Cross-encoder re-ranking failed: {e}")
        print("📄 Using original FAISS ranking")

    # Create enhanced context with metadata
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks[:10]):  # Use top 10 after re-ranking
        context_parts.append(f"[Document: {chunk['doc']}, Page: {chunk['page']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)

    print(f"📄 Context length: {len(context)} characters")
    print(f"📄 Number of chunks in context: {len(context_parts)}")

    # Ask LLM
    response = ask_llm(query, context)

    print("\nLLM Response:\n")
    print(response)
