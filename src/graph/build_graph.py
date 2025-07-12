from langgraph.graph import StateGraph, END, START
from .nodes import rag_generate, passthrough_response, summary_generate, router
from graph.my_state import MyState
from langgraph.prebuilt import ToolNode
from retriever.decide_tool import decide_tool_to_call

tools = [decide_tool_to_call]
tool_node = ToolNode(tools)


def langchain_graph():
    graph = StateGraph(MyState)
    graph.add_node("call_tool", tool_node)
    graph.add_node("rag_generate", rag_generate)
    graph.add_node("default_response", passthrough_response)
    graph.add_node("summary_generate", summary_generate)

    graph.set_entry_point("call_tool")
    graph.add_conditional_edges("call_tool", router, {
        "rag_generate": "rag_generate",
        "default_response": "default_response",
        "summary_generate": "summary_generate"
    })

    graph.add_edge("rag_generate", END)
    graph.add_edge("default_response", END)
    graph.add_edge("summary_generate", END)

    compiled_graph = graph.compile()
    return compiled_graph
