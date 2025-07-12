from langchain_core.messages import HumanMessage, AIMessage
from memory.history import ChatHistoryBuffer
from graph.build_graph import langchain_graph
from retriever.decide_tool import agent_executor
from quiz.generate_mcq import generate_mcq_questions, parse_mcqs
from quiz.generate_short import generate_short_questions, parse_short_answers, evaluate_short_answer
from rag_initializer import get_llm, get_chunk
# history = ChatHistoryBuffer(max_turns=10)
compiled_graph = langchain_graph()
agent_executor = agent_executor()


def process_query(query: str):
    agent_result = agent_executor.invoke({"input": query, "chat_history": []})
    intermediate_steps = agent_result.get("intermediate_steps", [])

    tool_messages = []
    for action, output in intermediate_steps:
        tool_messages.append({
            "tool_output": output,
            "tool_call_id": action.tool_call_id
        })

    graph_input = {
        "input": query,
        "chat_history": [],
        # "tool_messages": [],
        "intermediate_steps": intermediate_steps,
        "tool_messages": tool_messages,
        "messages": [HumanMessage(content=query),
                     AIMessage(
                         content="",
                         additional_kwargs={
                             "tool_calls": [
                                 {
                                     "name": "decide_tool_to_call",
                                     "arguments": {"question": query, "chat_history": []},
                                     "id": "tool-call-id",
                                     "type": "function"
                                 }
                             ]
                         }
                     )
                     ],
        "output": agent_result["output"]
    }

    response = compiled_graph.invoke(graph_input)

    print("\nFinal Output:\n", response['output'])
    return response['output']



