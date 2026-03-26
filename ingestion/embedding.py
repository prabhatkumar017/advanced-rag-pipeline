"""
WHY EMBEDDINGS?

LLMs cannot search text efficiently.

Embeddings convert text into vector space where
similar texts are close together.

Vector search is then performed.
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en")

def embed_texts(texts):
    return model.encode(texts)