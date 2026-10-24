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
    system_message = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )
    messages = [system_message] + state["messages"]
    response = model_with_math_tools.invoke(messages)

    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool = math_tools_by_name[tool_call["name"]]
        tool_result = tool.invoke(tool_call["args"])
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
        tool_messages.append(tool_message)

    return {"messages": tool_messages}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls:
        return "tool_node"
    return "__end__"