import os
import requests
import json
import re
import time

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

def extract_json(text):
    """Attempts to extract and clean a JSON object from raw text."""
    try:
        # Find the first JSON-looking block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        raw = match.group(0)

        # Clean common formatting issues
        cleaned = raw.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')

        return cleaned
    except Exception as e:
        print(f"⚠️ extract_json error: {e}")
        return None


def call_llm(api_key, system_prompt, query, context):
    """Calls the Together API and returns raw LLM output."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # ✅ Strong final user message to enforce JSON
    user_message = f"Context:\n{context}\n\nQuestion: {query}\n\n⚠️ Respond ONLY with a valid JSON object. Do NOT add any text outside the JSON."

    payload = {
        # "model": "Qwen/Qwen3-32B-Instruct",
        "model":"mistralai/Mixtral-8x7B-Instruct-v0.1",
        # mistralai/Mistral-7B-Instruct-v0.2
        # "model":"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2
    }

    response = requests.post(TOGETHER_URL, headers=headers, json=payload)
    return response.json()

def ask_llm(query, context):
    """Main function to get clean JSON from the LLM with retry."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ TOGETHER_API_KEY not found in environment variables.")

    # ✅ Load system prompt
    with open("prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    max_retries = 2
    for attempt in range(max_retries):
        result = call_llm(api_key, system_prompt, query, context)

        # Debugging: show raw response
        print("🔍 Raw API response:")
        print(json.dumps(result, indent=2))

        raw_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        json_str = extract_json(raw_output)

        if json_str:
            try:
                parsed = json.loads(json_str)
                return json.dumps(parsed)
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse cleaned JSON: {e}")
                print(f"🧪 Cleaned JSON candidate:\n{json_str}")


        # Retry with stronger instruction
        time.sleep(1)
        system_prompt += "\n\n🛑 STRICT RULE: Respond ONLY with a raw JSON object. No extra text."

    # If all retries fail
    raise ValueError("❌ Model failed to return valid JSON after retries.")
