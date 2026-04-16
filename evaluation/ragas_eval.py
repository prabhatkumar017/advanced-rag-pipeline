"""
RAGAS evaluation.

Measures:

1 Faithfulness
2 Context precision
3 Answer relevance
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


def evaluate_rag(dataset):

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    return results
