import os
import requests
import json
import re
import time

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

def extract_json(text):
    """Attempts to extract and clean a JSON object from raw text."""
    try:
        # First, try to find JSON blocks with various patterns
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
            r'\{.*?\}',  # Simple JSON
            r'\{[^}]*"answer"[^}]*"justification"[^}]*"source_clause"[^}]*"confidence"[^}]*\}',  # Specific format
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean common formatting issues
                    cleaned = match.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                    # Remove any trailing commas before closing braces
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    
                    parsed = json.loads(cleaned)
                    # Validate required fields
                    if all(key in parsed for key in ["answer", "justification", "source_clause", "confidence"]):
                        return cleaned
                except json.JSONDecodeError:
                    continue
        
        return None
    except Exception as e:
        print(f"⚠️ extract_json error: {e}")
        return None


def call_llm(api_key, system_prompt, query, context):
    """Calls the Together API and returns raw LLM output."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Improved user message with clearer instructions
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
        "temperature": 0.1,  # Lower temperature for more consistent JSON
        "max_tokens": 500
    }

    try:
        response = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None

def ask_llm(query, context):
    """Main function to get clean JSON from the LLM with retry."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ TOGETHER_API_KEY not found in environment variables.")

    # Load system prompt
    with open("prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    max_retries = 3
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1}/{max_retries}")
        
        result = call_llm(api_key, system_prompt, query, context)
        
        if not result:
            print("❌ API call failed")
            time.sleep(2)
            continue

        # Debugging: show raw response
        print("🔍 Raw API response:")
        print(json.dumps(result, indent=2))

        raw_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        json_str = extract_json(raw_output)

        if json_str:
            try:
                parsed = json.loads(json_str)
                # Validate the response
                if all(key in parsed for key in ["answer", "justification", "source_clause", "confidence"]):
                    print(f"✅ Successfully parsed JSON with all required fields")
                    return json.dumps(parsed, indent=2)
                else:
                    print(f"⚠️ Missing required fields in JSON: {parsed}")
                    print(f"🧪 Available fields: {list(parsed.keys())}")
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse cleaned JSON: {e}")
                print(f"🧪 Cleaned JSON candidate:\n{json_str}")

        # Retry with stronger instruction
        time.sleep(2)
        if attempt < max_retries - 1:
            system_prompt += "\n\n🛑 CRITICAL: You MUST respond with ONLY a valid JSON object. No other text."

    # If all retries fail, return a fallback response
    print("❌ All retries failed, returning fallback response")
    fallback_response = {
        "answer": "UNKNOWN",
        "justification": "Unable to process the query due to technical issues.",
        "source_clause": None,
        "confidence": 0.0
    }
    return json.dumps(fallback_response, indent=2)
