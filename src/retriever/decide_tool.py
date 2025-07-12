from langchain_core.tools import tool
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.vectorstores.utils import _cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from rag_initializer import get_llm, get_vector_store, get_embeddings

embeddings = get_embeddings()

SUMMARY_INTENT = "Give me a summary of the document"
summary_embedding = embeddings.embed_query(text=SUMMARY_INTENT)


@tool
def decide_tool_to_call(question: str, chat_history: list = []) -> Literal["rag_generate", "default_response", "summary_generate"]:
    """
    Uses semantic similarity to decide whether the question requires document context.
    Returns:
    - 'rag_generate' if the question is semantically similar to document content
    - 'default_response' if the question is unrelated (like greetings or general small talk)
    - 'summary_generate' if the question is asking for summary
    """
    if chat_history:
        previous_user_turn = [msg.content for msg in chat_history if isinstance(msg, HumanMessage)]
        last_turn = previous_user_turn[-1] if previous_user_turn else ""
        question = last_turn + " " + question

    query_embedding = get_embeddings().embed_query(question)
    similarity_score = _cosine_similarity([query_embedding], [summary_embedding])
    print("similarity score was found to be", similarity_score)
    if similarity_score > 0.60:
        return "summary_generate"

    threshold = 1.5
    result = get_vector_store().similarity_search_with_score(question, k=10)
    filtered = [doc for doc, score in result if score < threshold]

    if len(filtered) > 1:
        return "rag_generate"

    return "default_response"


tools = [decide_tool_to_call]

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant that decide which tool to use based on the document provided to you"),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(
    llm=get_llm(),
    tools=tools,
    prompt=prompt,
)


def agent_executor():
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True

    )
    return agent_executor

