import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


def init_llm():
    return init_chat_model("llama-3.3-70b-versatile", model_provider="groq")


def init_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
