import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

load_dotenv()

# Initialize LLM for local usage
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
)

deep_research_model = init_chat_model(model="gpt-5.2", model_provider="openai", temperature=0.2)