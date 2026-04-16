from fastapi import FastAPI

from ingestion.document_loader import load_documents
from ingestion.chunking import chunk_documents
from ingestion.embedding import embed_texts

from retrieval.vector_search import VectorSearch
from retrieval.bm25_search import BM25Search
from retrieval.hybrid_search import hybrid_retrieval
from retrieval.reranker import rerank

from generation.answer_generator import generate_answer
from feedback.interaction_logger import log_interaction

# Initialize pipeline once at startup
print("Initializing RAG pipeline...")
docs = load_documents()
chunks = chunk_documents(docs)
texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)
vector_db = VectorSearch(embeddings)
bm25 = BM25Search(texts)
print("Pipeline ready.")

app = FastAPI()


@app.get("/ask")
def ask(question: str):
    query_embedding = embed_texts([question])
    vec_results = vector_db.search(query_embedding)
    bm25_results = bm25.search(question)
    combined = hybrid_retrieval(vec_results, bm25_results, len(texts))
    retrieved_docs = [texts[i] for i in combined if i < len(texts)]
    reranked = rerank(question, retrieved_docs)
    context = "\n".join(reranked)
    answer = generate_answer(context, question)
    log_interaction(question, answer, context)
    return {"answer": answer}
