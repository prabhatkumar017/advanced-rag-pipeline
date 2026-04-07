from fastapi import FastAPI
from generation.answer_generator import generate_answer

app = FastAPI()

@app.get("/ask")

def ask(question: str):

    answer = generate_answer("", question)

    return {"answer": answer}