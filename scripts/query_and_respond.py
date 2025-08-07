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

def run_query_with_context(query, num_chunks=15, temperature=0.1):
    """Enhanced version for Streamlit frontend with configurable parameters"""
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    if not TOGETHER_API_KEY:
        raise EnvironmentError("❌ TOGETHER_API_KEY not found in environment variables.")

    # Load FAISS index
    index = faiss.read_index("outputs/vector_store/index.faiss")

    # Load chunk metadata
    with open("outputs/vector_store/meta.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Use the same model as in embedding for consistency
    embedder = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = embedder.encode([query])[0]

    # Enhanced retrieval with configurable number of chunks
    _, indices = index.search(np.array([query_embedding]), num_chunks)

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
        
    except Exception as e:
        # If cross-encoder fails, continue with original ranking
        pass

    # Create enhanced context with metadata
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks[:10]):  # Use top 10 after re-ranking
        context_parts.append(f"[Document: {chunk['doc']}, Page: {chunk['page']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)

    # Ask LLM with configurable temperature
    response = ask_llm_with_temperature(query, context, temperature)
    
    return response

def ask_llm_with_temperature(query, context, temperature=0.1):
    """Modified ask_llm function that accepts temperature parameter"""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ TOGETHER_API_KEY not found in environment variables.")

    # Load system prompt
    with open("prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    # Call the LLM with configurable temperature
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    user_message = f"""Context:
{context}

Question: {query}

IMPORTANT: Respond ONLY with a valid JSON object in this exact format:
{{
  "answer": "YES|NO|UNKNOWN",
  "justification": "Brief explanation with context reference",
  "source_clause": "Exact clause/section reference or null",
  "confidence": 0.0-1.0
}}

Do NOT add any text outside the JSON object."""

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": 500
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        raw_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            if all(key in parsed for key in ["answer", "justification", "source_clause", "confidence"]):
                return json.dumps(parsed, indent=2)
        
        # Fallback response
        fallback_response = {
            "answer": "UNKNOWN",
            "justification": "Unable to process the query due to technical issues.",
            "source_clause": None,
            "confidence": 0.0
        }
        return json.dumps(fallback_response, indent=2)
        
    except Exception as e:
        fallback_response = {
            "answer": "UNKNOWN",
            "justification": f"Error during processing: {str(e)}",
            "source_clause": None,
            "confidence": 0.0
        }
        return json.dumps(fallback_response, indent=2)
