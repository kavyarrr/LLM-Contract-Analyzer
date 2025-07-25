import os
import requests

def ask_llm(query, context):
    api_key = os.getenv("TOGETHER_API_KEY")

    system_prompt = (
    "You are a highly accurate legal assistant specializing in interpreting insurance contracts. "
    "You must answer user queries strictly using the provided context, which includes clauses, definitions, and terms from real insurance documents.\n\n"
    
    "Your goal is to give a direct YES or NO answer to the user’s question. Always justify your answer in 1–2 concise lines using evidence from the context "
    "and mention where in the context (page, section, clause) the justification was found.\n\n"
    
    "You must also rate your confidence in the answer from 0 to 1. Use high confidence (0.8–1.0) only when the answer is explicitly stated or clearly implied. "
    "Use low confidence (0.0–0.5) when the context is vague or incomplete.\n\n"
    
    "If the answer cannot be determined from the given context, respond with:\n"
    '{\n'
    '  "answer": "UNKNOWN",\n'
    '  "justification": "The answer could not be found in the provided context.",\n'
    '  "source_clause": null,\n'
    '  "confidence": 0.0\n'
    '}\n\n'

    "🛑 Do not make assumptions or hallucinate.\n\n"
    
    "✅ Your response must always be a strict JSON object in the following format:\n"
    '{\n'
    '  "answer": "<YES or NO or UNKNOWN>",\n'
    '  "justification": "<1–2 line explanation referencing the context>",\n'
    '  "source_clause": "<Exact clause, section, or line if available, else null>",\n'
    '  "confidence": <float between 0.0 and 1.0>\n'
    '}'
)


    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    # TEMP: debug print to see actual structure
    import json
    print("🔍 Response structure:")
    print(json.dumps(result, indent=2))

    # Check if choices exists, else fall back
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError("Unexpected response format from Together API")
