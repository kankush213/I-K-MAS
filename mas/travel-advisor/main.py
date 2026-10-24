from graph import build_parallel_travel_agent

if __name__ == "__main__":
    agent = build_parallel_travel_agent()
    
    config = {"configurable": {"thread_id": "user-local-01"}}
    
    result = agent.invoke(
        {
            "user_query": "December 2026 around. In India, Suggest me 3 places to visit for small vacation break with my family.",
            "messages": [],
            "tasks": [],
            "requires_synthesis": False,
            "agent_results": [],
            "final_answer": "",
            "search_messages": [],
            "planner_messages": [],
        },
        config=config
    )

    print("=" * 60)
    print("FINAL RESPONSE:")
    print(result['final_answer'])