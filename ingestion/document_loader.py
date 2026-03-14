"""
Loads documents for RAG.

WHY THIS FILE EXISTS:
A RAG system must support multiple data sources
(pdf, csv, html, database).

Separating loading logic makes system scalable.
"""

from datasets import load_dataset

def load_documents():

    # Using open source dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:200]")

    documents = []

    for row in dataset:
        documents.append({
            "text": row["article"],
            "source": "cnn_dailymail"
        })

    return documents