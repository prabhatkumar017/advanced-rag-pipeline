"""
Vector database implementation.

FAISS is used because:
- fast
- local
- widely used
"""

import faiss
import numpy as np

class VectorSearch:

    def __init__(self, embeddings):

        dim = embeddings.shape[1]

        # IndexFlatL2 = Euclidean distance search
        self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

    def search(self, query_embedding, k=5):

        distances, indices = self.index.search(query_embedding, k)

        return indices[0]