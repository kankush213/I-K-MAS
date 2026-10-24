import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai_api_key:
    raise ValueError("Set OPENAI_API_KEY in calculator-basic-example/.env before running the calculator.")

# Initialize LLM for local usage
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=openai_api_key,
)

deep_research_model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0.2, api_key=openai_api_key)