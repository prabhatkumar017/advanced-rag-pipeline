import streamlit as st
import requests

st.title("CNN/DailyMail News RAG Assistant")

question = st.text_input("Ask question")

if st.button("Submit"):

    response = requests.get(
        "http://localhost:8000/ask",
        params={"question":question}
    )

    st.write(response.json()["answer"])