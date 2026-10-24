import operator
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from config import llm
from tools.math_tools import math_tools, math_tools_by_name

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

model_with_math_tools = llm.bind_tools(math_tools)

def llm_call(state: MessagesState):
    response = model_with_math_tools.invoke(
        [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
        + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = math_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "__end__"