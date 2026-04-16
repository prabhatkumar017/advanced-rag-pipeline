"""
Analyzes interaction logs to surface usage patterns
and potential retrieval failure areas.
"""

import json
from collections import Counter


def analyze_logs(log_file="logs.json"):

    entries = []

    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        print("No logs found. Run some queries first.")
        return

    if not entries:
        print("Log file is empty.")
        return

    print(f"Total interactions: {len(entries)}")

    # Most common query terms (excluding short stop words)
    stop_words = {"the", "a", "an", "is", "in", "of", "to", "and", "what", "how", "who", "why", "was"}
    words = []
    for entry in entries:
        for word in entry["query"].lower().split():
            if word not in stop_words and len(word) > 2:
                words.append(word)

    if words:
        print("\nTop 10 query terms:")
        for word, count in Counter(words).most_common(10):
            print(f"  {word}: {count}")

    # Show recent queries
    print("\nLast 5 queries:")
    for entry in entries[-5:]:
        ts = entry.get("timestamp", "unknown time")
        print(f"  [{ts}] {entry['query']}")
