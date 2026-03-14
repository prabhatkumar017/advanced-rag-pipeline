"""
WHY CHUNKING IS REQUIRED:

LLMs cannot handle extremely long documents.

Therefore we split documents into smaller pieces.

These pieces are called CHUNKS.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:

        split_texts = splitter.split_text(doc["text"])

        for text in split_texts:

            chunks.append({
                "text": text,
                "source": doc["source"]
            })

    return chunks