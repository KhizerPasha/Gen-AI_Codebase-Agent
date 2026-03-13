# agent/graph.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.tools import ALL_TOOLS
from dotenv import load_dotenv
import os

load_dotenv()

# ── System prompt for the agent ───────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """
You are an expert AI code analysis assistant with deep knowledge of software engineering.

You have access to a codebase that has been indexed and embedded. You can analyze it using these tools:
- explain_code    → explain what code does in plain English
- find_bugs       → find bugs, issues, and bad patterns
- refactor_code   → suggest refactoring and improvements
- find_usage      → find where functions/classes are used

Rules:
1. ALWAYS use a tool to retrieve actual code before answering
2. Never make up code — only analyze what the tools return
3. Be specific and cite the file names and line numbers in your answers
4. If a query needs multiple tools, call them in sequence
5. After using tools, give a clear, structured final answer

You are analyzing a RAG-based codebase agent project written in Python.
""".strip()


def build_agent():
    """
    Build and return the LangGraph ReAct agent.
    ReAct = Reasoning + Acting loop:
      Think → Choose Tool → Execute → Observe → Think again → Answer
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT
    )

    print("✅ LangGraph Agent built successfully!")
    return agent


def run_agent(agent, query: str) -> str:
    """
    Run a single query through the agent.
    Returns the final response as a string.
    """
    print(f"\n{'='*55}")
    print(f"👤 User: {query}")
    print(f"{'='*55}")

    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # Extract final message from agent response
    final_message = result["messages"][-1].content

    print(f"\n🤖 Agent: {final_message}")
    return final_message


def run_agent_chat(agent):
    """
    Interactive chat loop with the agent.
    Type 'exit' to quit.
    """
    print("\n" + "="*55)
    print("💬 CODEBASE AGENT — Interactive Chat")
    print("="*55)
    print("Commands: 'exit' to quit\n")
    print("Example queries:")
    print("  - explain how the file loader works")
    print("  - find bugs in the chunker")
    print("  - where is chunk_file used")
    print("  - suggest refactors for the embedder")
    print("="*55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            run_agent(agent, user_input)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue