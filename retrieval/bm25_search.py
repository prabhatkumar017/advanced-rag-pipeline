"""
WHY BM25?

Vector search finds semantic similarity
but sometimes exact keywords are important.

BM25 solves this.

Hybrid search = vector + BM25
"""

from rank_bm25 import BM25Okapi

class BM25Search:

    def __init__(self, texts):

        tokenized = [doc.split(" ") for doc in texts]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=5):

        tokenized_query = query.split(" ")

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        return ranked[:k]