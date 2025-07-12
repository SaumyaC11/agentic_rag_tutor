import streamlit as st
from retriever.splitter import vector, chunking

from llm.model_init import init_llm, init_embeddings

@st.cache_resource
def get_vector_store():
    return vector()


@st.cache_resource
def get_graph():
    from graph.build_graph import langchain_graph
    return langchain_graph()

@st.cache_resource
def get_llm():
    return init_llm()

@st.cache_resource
def get_chunk():
    return chunking()

@st.cache_resource
def get_embeddings():
    return init_embeddings()
