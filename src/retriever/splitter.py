from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from data.loader import loader
from llm.model_init import init_embeddings
import streamlit as st


def chunking():
    if "chunked_docs" in st.session_state:
        return st.session_state.chunked_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        add_start_index=True
    )
    docs = loader()
    chunks = splitter.split_documents(docs)
    st.session_state.chunked_docs = chunks
    return st.session_state.chunked_docs


def vector():
    if "vector_store" in st.session_state:
        return st.session_state.vector_store

    embedding = init_embeddings()
    chunks = chunking()
    vector_store = FAISS.from_documents(chunks, embedding)
    st.session_state.vector_store = vector_store
    return st.session_state.vector_store
