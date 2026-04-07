"""
Detect hallucinations.

Idea:
Ask LLM if answer is supported by context.
"""

from openai import OpenAI

client = OpenAI()

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