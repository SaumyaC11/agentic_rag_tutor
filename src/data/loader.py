from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
import streamlit as st
import os


def loader():
    if "documents" in st.session_state:
        return st.session_state.documents

    if "uploaded_pdf_paths" not in st.session_state:
        raise ValueError("No uploaded PDF please provide that")

    if "documents" in st.session_state:
        return st.session_state.documents

    docs = []
    for path in st.session_state.uploaded_pdf_paths:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(path)
        elif ext in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs.extend(loader.load())

    st.session_state.documents = docs
    return docs
