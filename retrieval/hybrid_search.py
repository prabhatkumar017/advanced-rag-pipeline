"""
Hybrid retrieval combines results from:

1 Vector search
2 Keyword search

This improves recall significantly.
"""

def hybrid_retrieval(vector_results, bm25_results, max_index):

    combined = list(set(vector_results.tolist() + bm25_results))

    # Filter invalid indices
    combined = [i for i in combined if i < max_index]

    return combined[:5]