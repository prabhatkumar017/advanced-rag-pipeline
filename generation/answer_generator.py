"""
This module sends prompt to LLM.

Uses OpenAI API but can be replaced with
any LLM.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from .prompt_templates import PROMPT_TEMPLATE

# Load environment variables once
load_dotenv()

def get_openai_client():
    """
    Initialize and return OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    return OpenAI(api_key=api_key)


def generate_answer(context, question):
    """
    Generate answer from LLM using context + question.
    """
    client = get_openai_client()

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content