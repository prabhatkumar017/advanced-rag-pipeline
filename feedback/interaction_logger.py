"""
Stores query logs.

WHY?

To analyze failure cases
and improve retrieval.
"""

import json
from datetime import datetime


def log_interaction(query, answer, context=""):

    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "context": context
    }

    with open("logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")
