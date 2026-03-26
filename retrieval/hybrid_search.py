"""
Hybrid retrieval combines results from:

1 Vector search
2 Keyword search

This improves recall significantly.
"""

def hybrid_retrieval(vector_results, bm25_results):

    combined = list(set(vector_results + bm25_results))

    return combined[:5]