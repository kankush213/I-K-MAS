from graph import build_math_agent
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    agent = build_math_agent()
    config = {"configurable": {"thread_id": "math-local-01"}}
    
    messages = [HumanMessage(content="Add 3 and 4, then multiply the result by 5.")]
    result = agent.invoke({"messages": messages}, config=config)
    
    print("=" * 60)
    print("FINAL RESPONSE:")
    for m in result["messages"]:
        m.pretty_print()