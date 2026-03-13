# agent/tools.py
from langchain_core.tools import tool
from retrieval.retriever import CodeRetriever
from llm.client import call_llm
from llm.prompts import (
    build_explain_prompt,
    build_bug_prompt,
    build_refactor_prompt,
    build_usage_prompt
)

# Single shared retriever instance
_retriever = None

def get_retriever() -> CodeRetriever:
    """Lazy-load retriever so it's only loaded once."""
    global _retriever
    if _retriever is None:
        _retriever = CodeRetriever()
        _retriever.load()
    return _retriever


@tool
def explain_code(query: str) -> str:
    """
    Use this tool when the user wants to understand what code does.
    Input should be a description of the function or concept to explain.
    Examples: 'how does file loading work', 'explain the chunker'
    """
    print(f"\n🔧 [Tool Called] explain_code('{query}')")
    retriever = get_retriever()
    chunks = retriever.search(query, top_k=4)

    if not chunks:
        return "No relevant code found for this query."

    system, user = build_explain_prompt(query, chunks)
    return call_llm(system, user)


@tool
def find_bugs(query: str) -> str:
    """
    Use this tool when the user wants to find bugs, issues, or bad patterns.
    Input should describe what part of the code to analyze for bugs.
    Examples: 'bugs in file loader', 'issues with error handling'
    """
    print(f"\n🔧 [Tool Called] find_bugs('{query}')")
    retriever = get_retriever()
    chunks = retriever.search(query, top_k=4)

    if not chunks:
        return "No relevant code found for this query."

    system, user = build_bug_prompt(query, chunks)
    return call_llm(system, user)


@tool
def refactor_code(query: str) -> str:
    """
    Use this tool when the user wants refactoring suggestions or improvements.
    Input should describe what code to refactor.
    Examples: 'refactor the chunker', 'improve the embedding pipeline'
    """
    print(f"\n🔧 [Tool Called] refactor_code('{query}')")
    retriever = get_retriever()
    chunks = retriever.search(query, top_k=4)

    if not chunks:
        return "No relevant code found for this query."

    system, user = build_refactor_prompt(query, chunks)
    return call_llm(system, user)


@tool
def find_usage(query: str) -> str:
    """
    Use this tool when the user asks where something is used or called.
    Input should be the function/class name to find usages of.
    Examples: 'where is load_from_directory used', 'find usages of chunk_file'
    """
    print(f"\n🔧 [Tool Called] find_usage('{query}')")
    retriever = get_retriever()
    chunks = retriever.search(query, top_k=4)

    if not chunks:
        return "No relevant code found for this query."

    system, user = build_usage_prompt(query, chunks)
    return call_llm(system, user)


# Export all tools as a list for the agent
ALL_TOOLS = [explain_code, find_bugs, refactor_code, find_usage]