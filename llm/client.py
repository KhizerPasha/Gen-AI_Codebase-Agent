# llm/client.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    Returns a GPT-4o-mini instance.
    We use mini for speed + cost. Easy to swap to gpt-4o later.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Core LLM call function.
    Returns the response as a plain string.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    return response.content