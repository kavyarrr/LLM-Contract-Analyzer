import sys
import os

# Add subfolders to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from query_and_respond import run_query

if __name__ == "__main__":
    # Test with a simple query
    test_query = "Does the Global Health Care policy cover AYUSH Day Care treatments?"
    print(f"Testing query: {test_query}")
    
    # We need to modify the input method for testing
    import builtins
    original_input = builtins.input
    
    def mock_input(prompt):
        return test_query
    
    builtins.input = mock_input
    
    try:
        run_query()
    finally:
        builtins.input = original_input 