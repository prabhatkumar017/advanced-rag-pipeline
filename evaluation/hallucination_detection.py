"""
Detect hallucinations.

Idea:
Ask LLM if answer is supported by context.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def check_hallucination(answer, context):

    prompt = f"""
    Check if the answer is supported by context.

    Context:
    {context}

    Answer:
    {answer}

    Respond YES or NO.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content