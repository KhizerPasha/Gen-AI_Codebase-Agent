# llm/prompts.py
"""
All prompt templates live here.
Each task type (explain, bugs, refactor, usage) has its own system prompt.
Keeping prompts separate makes them easy to tune later.
"""

# ── System Prompts ─────────────────────────────────────────────────────────

SYSTEM_EXPLAIN = """
You are an expert software engineer and code explainer.
You are given code chunks retrieved from a real codebase.
Your job is to explain what the code does in clear, simple language.

Rules:
- Start with a one-line summary
- Explain the purpose, inputs, outputs
- Mention any important logic or edge cases
- Use bullet points for clarity
- Do NOT repeat the code back, just explain it
""".strip()


SYSTEM_BUG_FINDER = """
You are a senior code reviewer and bug hunter.
You are given code chunks retrieved from a real codebase.
Your job is to find potential bugs, issues, and bad patterns.

Rules:
- List each issue with: [SEVERITY: HIGH/MEDIUM/LOW]
- Explain WHY it is a bug or bad practice
- Suggest a fix for each issue
- If no bugs found, say so clearly
- Be specific, not generic
""".strip()


SYSTEM_REFACTOR = """
You are a software architect specializing in clean code.
You are given code chunks retrieved from a real codebase.
Your job is to suggest concrete refactoring improvements.

Rules:
- Focus on readability, performance, and maintainability
- Show the improved version of the code where possible
- Explain WHY each change is better
- Keep the same functionality, only improve structure
""".strip()


SYSTEM_FIND_USAGE = """
You are a codebase analysis expert.
You are given code chunks retrieved from a real codebase.
Your job is to explain how and where specific functions or classes are used.

Rules:
- List all usages you can see in the retrieved chunks
- Explain the context of each usage
- Note any patterns in how it's being called
""".strip()


# ── Prompt Builder ─────────────────────────────────────────────────────────

def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clean block for the LLM prompt.
    """
    formatted = []

    for i, chunk in enumerate(chunks, 1):
        block = f"""
--- Chunk {i} ---
File: {chunk['file_path']}
Function/Class: {chunk['name']} ({chunk['chunk_type']})
Lines: {chunk['start_line']} to {chunk['end_line']}
Language: {chunk['language']}
Similarity Score: {chunk.get('similarity_score', 'N/A')}

Code:
{chunk['code']}
""".strip()
        formatted.append(block)

    return "\n\n".join(formatted)


def build_explain_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for explanation task."""
    context = format_chunks_for_prompt(chunks)
    user_prompt = f"""
The user asked: "{query}"

Here are the most relevant code chunks from the codebase:

{context}

Please explain this code clearly.
""".strip()
    return SYSTEM_EXPLAIN, user_prompt


def build_bug_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for bug finding task."""
    context = format_chunks_for_prompt(chunks)
    user_prompt = f"""
The user wants to find bugs related to: "{query}"

Here are the most relevant code chunks from the codebase:

{context}

Please analyze for bugs, issues, and bad patterns.
""".strip()
    return SYSTEM_BUG_FINDER, user_prompt


def build_refactor_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for refactor task."""
    context = format_chunks_for_prompt(chunks)
    user_prompt = f"""
The user wants refactoring suggestions for: "{query}"

Here are the most relevant code chunks from the codebase:

{context}

Please suggest concrete refactoring improvements.
""".strip()
    return SYSTEM_REFACTOR, user_prompt


def build_usage_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for usage finding task."""
    context = format_chunks_for_prompt(chunks)
    user_prompt = f"""
The user wants to know about usage of: "{query}"

Here are the most relevant code chunks from the codebase:

{context}

Please explain how and where this is used.
""".strip()
    return SYSTEM_FIND_USAGE, user_prompt