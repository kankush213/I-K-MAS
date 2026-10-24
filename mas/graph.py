from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from agents.nodes import (
    TravelPlannerState, classify_query_parallel, dispatch_to_agents,
    search_agent_node, planner_agent_node, search_tool_node, planner_tool_node,
    synthesizer_node, should_continue_for_searchAgent, should_continue_for_plannerAgent
)

def build_parallel_travel_agent(enable_persistence: bool = True):
    builder = StateGraph(TravelPlannerState)

    builder.add_node("orchestrator", classify_query_parallel)
    builder.add_node("search_agent_node", search_agent_node)
    builder.add_node("planner_agent_node", planner_agent_node)
    builder.add_node("search_tool_node", search_tool_node)
    builder.add_node("planner_tool_node", planner_tool_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", dispatch_to_agents, ["search_agent_node", "planner_agent_node"])

    builder.add_conditional_edges("search_agent_node", should_continue_for_searchAgent, ["search_tool_node", "synthesizer"])
    builder.add_edge("search_tool_node", "search_agent_node")

    builder.add_conditional_edges("planner_agent_node", should_continue_for_plannerAgent, ["planner_tool_node", "synthesizer"])
    builder.add_edge("planner_tool_node", "planner_agent_node")

    builder.add_edge("synthesizer", END)

    checkpointer = InMemorySaver() if enable_persistence else None
    return builder.compile(checkpointer=checkpointer)