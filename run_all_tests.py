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
        result = subprocess.run(
            ["python", QUERY_SCRIPT, query],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        
        # Extract JSON safely
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON found in output.")
        
        json_str = output[json_start:json_end]
        return json.loads(json_str)
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

    for case in test_cases:
        query = case["query"]
        expected = case["expected"]

        print(f"\n🚀 Testing: {query}")
        actual = run_query(query)

        got_ans = actual.get("answer") if actual else "ERROR"
        result = "✅ PASS" if compare(expected, actual) else "❌ FAIL"

        table.add_row([query[:40] + "...", expected["answer"], got_ans, result])

    print("\n" + table.get_string())

if __name__ == "__main__":
    main()
