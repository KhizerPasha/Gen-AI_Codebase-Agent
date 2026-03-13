# llm/analyzer.py
from retrieval.retriever import CodeRetriever
from llm.client import call_llm
from llm.prompts import (
    build_explain_prompt,
    build_bug_prompt,
    build_refactor_prompt,
    build_usage_prompt
)


class CodeAnalyzer:
    """
    The main interface for analyzing code.
    Combines retrieval + LLM into clean task methods.
    """

    def __init__(self):
        self.retriever = CodeRetriever()
        self.retriever.load()
        print("✅ CodeAnalyzer ready!\n")

    def explain(self, query: str, top_k: int = 4) -> str:
        """Explain what a piece of code does."""
        print(f"💡 Explaining: '{query}'")
        chunks = self.retriever.search(query, top_k=top_k)
        system, user = build_explain_prompt(query, chunks)
        return call_llm(system, user)

    def find_bugs(self, query: str, top_k: int = 4) -> str:
        """Find bugs and bad patterns in relevant code."""
        print(f"🐛 Finding bugs for: '{query}'")
        chunks = self.retriever.search(query, top_k=top_k)
        system, user = build_bug_prompt(query, chunks)
        return call_llm(system, user)

    def refactor(self, query: str, top_k: int = 4) -> str:
        """Suggest refactoring improvements."""
        print(f"🔧 Refactoring: '{query}'")
        chunks = self.retriever.search(query, top_k=top_k)
        system, user = build_refactor_prompt(query, chunks)
        return call_llm(system, user)

    def find_usage(self, query: str, top_k: int = 4) -> str:
        """Find where something is used in the codebase."""
        print(f"🔍 Finding usage of: '{query}'")
        chunks = self.retriever.search(query, top_k=top_k)
        system, user = build_usage_prompt(query, chunks)
        return call_llm(system, user)