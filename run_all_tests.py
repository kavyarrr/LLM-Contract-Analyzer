import json
import subprocess
import os
from prettytable import PrettyTable

# Path to your test cases file
TEST_FILE = "test_cases.json"
QUERY_SCRIPT = "scripts/query_and_respond.py"

def run_query(query):
    """Runs the query_and_respond.py script and captures its JSON output."""
    try:
        # Create a temporary input file for the query
        with open("temp_query.txt", "w") as f:
            f.write(query)
        
        # Run the script with input redirection
        result = subprocess.run(
            ["python", "main.py"],
            input=query + "\n",
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        output = result.stdout.strip()
        
        # Look for JSON in the output
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        
        if json_start != -1 and json_end != -1:
            json_str = output[json_start:json_end]
            try:
                parsed = json.loads(json_str)
                # Validate required fields
                if all(key in parsed for key in ["answer", "justification", "source_clause", "confidence"]):
                    return parsed
                else:
                    print(f"⚠️ Missing required fields in JSON: {parsed}")
                    return None
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error: {e}")
                print(f"🧪 JSON candidate: {json_str}")
                return None
        else:
            print(f"⚠️ No JSON found in output")
            print(f"🧪 Full output: {output}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for query: {query}")
        return None
    except Exception as e:
        print(f"❌ Error parsing output for query '{query}': {e}")
        return None

def compare(expected, actual):
    """Compares expected vs actual (only 'answer' for pass/fail)."""
    if not actual:
        return False
    return actual.get("answer") == expected.get("answer")

def main():
    # Load test cases
    with open(TEST_FILE, "r") as f:
        test_cases = json.load(f)

    table = PrettyTable(["Query", "Expected", "Got", "Result"])
    passed = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        query = case["query"]
        expected = case["expected"]

        print(f"\n🚀 Testing ({i+1}/{total}): {query}")
        actual = run_query(query)

        got_ans = actual.get("answer") if actual else "ERROR"
        result = "✅ PASS" if compare(expected, actual) else "❌ FAIL"
        
        if compare(expected, actual):
            passed += 1

        table.add_row([query[:40] + "...", expected["answer"], got_ans, result])

    print("\n" + table.get_string())
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    main()
