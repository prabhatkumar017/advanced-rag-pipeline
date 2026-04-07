"""
Main pipeline that connects all modules.

This file demonstrates full RAG workflow.
"""

from ingestion.document_loader import load_documents
from ingestion.chunking import chunk_documents
from ingestion.embedding import embed_texts

from retrieval.vector_search import VectorSearch
from retrieval.bm25_search import BM25Search
from retrieval.hybrid_search import hybrid_retrieval
from retrieval.reranker import rerank

from generation.answer_generator import generate_answer


print("Loading documents")

docs = load_documents()

print("Chunking")

chunks = chunk_documents(docs)

texts = [c["text"] for c in chunks]

print("Embedding")

embeddings = embed_texts(texts)

vector_db = VectorSearch(embeddings)

bm25 = BM25Search(texts)


def ask(question):

    query_embedding = embed_texts([question])

    vec_results = vector_db.search(query_embedding)

    bm25_results = bm25.search(question)

    combined = hybrid_retrieval(vec_results, bm25_results, len(texts))

    # docs = [texts[i] for i in combined]
    print("Total texts:", len(texts))
    print("Combined indices:", combined)
    docs = [texts[i] for i in combined if i < len(texts)]

    reranked = rerank(question, docs)

    context = "\n".join(reranked)

    answer = generate_answer(context, question)

    return answer


while True:

    q = input("Question: ")

    print(ask(q))