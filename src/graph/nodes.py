
from .my_state import MyState
from rag_initializer import get_vector_store, get_llm, get_chunk, get_embeddings
import time
SUMMARY_INTENT = "Give me a summary of the document"
summary_embedding = get_embeddings().embed_query(text=SUMMARY_INTENT)


def rag_generate(state: MyState):
    query = state.input
    result = get_vector_store().similarity_search_with_score(query, k=10)
    filtered = [doc for doc, score in result if score < 1.5]

    if not filtered:
        return {"output": "The document does not contain that information."}

    context = "\n\n".join(doc.page_content for doc in filtered)
    rag_prompt = f"""
    You are a helpful and knowledgeable assistant. Use the following document context to answer the user's question clearly and informatively.

    --------------------
    {context}
    --------------------

    Answer the question below using only the information from the document. It’s okay to rephrase, elaborate, or provide helpful context — as long as it is directly supported by the document.

    Do not guess or include any information not found in the context. If the answer is not present, respond:
    "The document does not contain that information."

    Be accurate, natural, and informative. Length can vary as needed to fully explain the answer.

    User Question: {query}
    """

    response = get_llm().invoke(rag_prompt)
    return {"output": response.content if hasattr(response, "content") else response}


def summary_generate(state: MyState):
    top_k = 5
    relevant_chunks = get_vector_store().similarity_search_by_vector(summary_embedding, k=top_k)

    joined_text = "\n\n".join([doc.page_content for doc in relevant_chunks])
    token_limit = 3000

    if len(joined_text) < token_limit:
        summary_prompt = f"""
           You are an expert assistant tasked with summarizing study material.

           Please generate a **detailed and structured summary** of the following content. 
           Highlight important definitions, concepts, and examples where applicable. 
           Avoid condensing too much—aim for clarity and depth.

           Content:
           {joined_text}
           """
        response = get_llm().invoke(summary_prompt)
        return {"output": response.content if hasattr(response, "content") else response}

    partial_summaries = []
    for doc in relevant_chunks:
        chunk_prompt = f"""
        Please summarize the following section in a **clear and informative** manner.
        Focus on covering key definitions, concepts, and any technical terms.
    
        Section:
        {doc.page_content}
        """
        partial_response = get_llm().invoke(chunk_prompt)
        partial_summaries.append(partial_response.content if hasattr(partial_response, "content") else partial_response)
        time.sleep(1.5)

    combined_summary_text = "\n".join(partial_summaries)
    final_prompt = f"""
    You are a study assistant. Combine the following section-wise summaries into a **detailed overview** of the document. 
    Maintain structure and include all key insights from the sections. Avoid limiting to a single paragraph.
    
    Section Summaries:
    {combined_summary_text}
    """
    final_response = get_llm().invoke(final_prompt)
    return {
            "output": final_response.content if hasattr(final_response, "content") else final_response
    }


def passthrough_response(state: MyState):
    final_output = state.output
    return {"output": final_output}


def router(state: MyState):
    tool_outputs = state.tool_messages
    if tool_outputs:
        result = tool_outputs[-1]["tool_output"]
        if result == "rag_generate":
            return "rag_generate"
        elif result == "summary_generate":
            return "summary_generate"
        else:
            return "default_response"
    return "default_response"
