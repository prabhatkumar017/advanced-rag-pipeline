"""
Stores query logs.

WHY?

To analyze failure cases
and improve retrieval.
"""

import json

def log_interaction(query, answer):

    log = {
        "query": query,
        "answer": answer
    }

    with open("logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")