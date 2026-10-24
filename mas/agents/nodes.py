import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage, AnyMessage
from config import llm
from tools.search_tools import search_tools, planner_tools
from agents.prompts import flightHotelSearch_instructions, travel_scout_instructions

class AgentTask(BaseModel):
    source: Literal["search_agent_node", "planner_agent_node"]
    user_query: str
    focus: str = ""

class TravelPlannerState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    user_query: str
    tasks: List[AgentTask]
    requires_synthesis: bool
    agent_results: Annotated[List[dict], operator.add]
    final_answer: str
    search_messages: Annotated[List[AnyMessage], operator.add]
    planner_messages: Annotated[List[AnyMessage], operator.add]

class ClassificationResult(BaseModel):
    tasks: List[AgentTask] = Field(description="List of agents to invoke with targeted queries")
    requires_synthesis: bool = Field(default=False, description="Whether multiple agents are used")

search_model_with_tools = llm.bind_tools(search_tools)
planner_model_with_tools = llm.bind_tools(planner_tools)
search_tools_by_name = {tool.name: tool for tool in search_tools}
planner_tools_by_name = {tool.name: tool for tool in planner_tools}

def classify_query_parallel(state: TravelPlannerState) -> dict:
    structured_llm = llm.with_structured_output(ClassificationResult)
    result = structured_llm.invoke([
        {"role": "system", "content": "You are a Travel Request Router. Dispatch to search_agent_node (flights/hotels) or planner_agent_node (itinerary/general guidance), or both."},
        {"role": "user", "content": state["user_query"]}
    ])
    return {"tasks": result.tasks, "requires_synthesis": result.requires_synthesis}

def dispatch_to_agents(state: TravelPlannerState):
    from langgraph.types import Send
    tasks = state.get("tasks", [])
    sends = []
    for task in tasks:
        worker_state = {
            "messages": state.get("messages", []),
            "user_query": task.user_query,
            "tasks": state.get("tasks", []),
            "requires_synthesis": state.get("requires_synthesis", False),
            "agent_results": [],
            "final_answer": "",
            "search_messages": [],
            "planner_messages": [],
        }
        if task.source == "search_agent_node":
            sends.append(Send("search_agent_node", worker_state))
        elif task.source == "planner_agent_node":
            sends.append(Send("planner_agent_node", worker_state))
    return sends

def search_agent_node(state: TravelPlannerState):
    search_msgs = state.get("search_messages", [])
    messages = [SystemMessage(content=flightHotelSearch_instructions)] + search_msgs
    if not search_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))
    response = search_model_with_tools.invoke(messages)
    return {
        "search_messages": [response],
        "agent_results": [{"agent": "search_agent", "focus": "flights and hotels", "result": response.content}] if not response.tool_calls else []
    }

def search_tool_node(state: TravelPlannerState):
    search_msgs = state.get("search_messages", [])
    if not search_msgs or not hasattr(search_msgs[-1], 'tool_calls'):
        return {"search_messages": []}
    result = []
    for tool_call in search_msgs[-1].tool_calls:
        tool = search_tools_by_name.get(tool_call["name"])
        if tool:
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"search_messages": result}

def planner_agent_node(state: TravelPlannerState):
    planner_msgs = state.get("planner_messages", [])
    messages = [SystemMessage(content=travel_scout_instructions)] + planner_msgs
    if not planner_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))
    response = planner_model_with_tools.invoke(messages)
    return {
        "planner_messages": [response],
        "agent_results": [{"agent": "planner_agent", "focus": "travel planning", "result": response.content}] if not response.tool_calls else []
    }

def planner_tool_node(state: TravelPlannerState):
    planner_msgs = state.get("planner_messages", [])
    if not planner_msgs or not hasattr(planner_msgs[-1], 'tool_calls'):
        return {"planner_messages": []}
    result = []
    for tool_call in planner_msgs[-1].tool_calls:
        tool = planner_tools_by_name.get(tool_call["name"])
        if tool:
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"planner_messages": result}

def synthesizer_node(state: TravelPlannerState) -> dict:
    agent_results = state.get("agent_results", [])
    if not agent_results:
        return {"final_answer": "I couldn't process your request.", "messages": [AIMessage(content="I couldn't process your request.")]}
    if len(agent_results) == 1:
        result = agent_results[0]["result"]
        return {"final_answer": result, "messages": [AIMessage(content=result)]}

    results_formatted = "\n\n" + "="*50 + "\n\n".join([f"**{r['agent'].upper()}**\n{r['result']}" for r in agent_results])
    synthesis_prompt = "Combine these travel agent outputs into a single comprehensive response:\n{results}"
    response = llm.invoke([HumanMessage(content=synthesis_prompt.format(results=results_formatted))])
    return {"final_answer": response.content, "messages": [AIMessage(content=response.content)]}

def should_continue_for_searchAgent(state: TravelPlannerState):
    search_msgs = state.get("search_messages", [])
    if search_msgs and hasattr(search_msgs[-1], 'tool_calls') and search_msgs[-1].tool_calls:
        return "search_tool_node"
    return "synthesizer"

def should_continue_for_plannerAgent(state: TravelPlannerState):
    planner_msgs = state.get("planner_messages", [])
    if planner_msgs and hasattr(planner_msgs[-1], 'tool_calls') and planner_msgs[-1].tool_calls:
        return "planner_tool_node"
    return "synthesizer"