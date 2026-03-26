"""
Prompt engineering is critical in RAG.

LLM must be forced to answer ONLY from context.
"""

PROMPT_TEMPLATE = """
You are a news assistant trained on CNN/DailyMail-style articles.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""