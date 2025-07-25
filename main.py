# main.py

import sys
import os

# Add subfolders to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from query_and_respond import run_query

if __name__ == "__main__":
    run_query()
