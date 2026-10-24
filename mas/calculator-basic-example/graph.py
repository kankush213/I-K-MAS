from langgraph.graph import StateGraph, START, END
from agents.math_nodes import MessagesState, llm_call, tool_node, should_continue

def build_math_agent():
    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")

    return builder.compile()