"""
RAGAS evaluation.

Measures:

1 Faithfulness
2 Context precision
3 Answer relevance
"""

from ragas import evaluate

def evaluate_rag(dataset):

    results = evaluate(dataset)

    return results