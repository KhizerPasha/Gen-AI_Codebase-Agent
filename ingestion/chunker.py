# ingestion/chunker.py
import ast
import re
from typing import Optional


def chunk_python_file(file_info: dict) -> list[dict]:
    """
    Parse a Python file using AST and extract functions + classes as chunks.
    Each chunk = one function or class with its full source code.
    """
    content = file_info["content"]
    file_path = file_info["file_path"]
    chunks = []

    try:
        tree = ast.parse(content)
        lines = content.splitlines()

        for node in ast.walk(tree):

            # Extract functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = _extract_node_chunk(
                    node=node,
                    lines=lines,
                    file_path=file_path,
                    language="python",
                    chunk_type="function"
                )
                chunks.append(chunk)

            # Extract classes (as a whole)
            elif isinstance(node, ast.ClassDef):
                chunk = _extract_node_chunk(
                    node=node,
                    lines=lines,
                    file_path=file_path,
                    language="python",
                    chunk_type="class"
                )
                chunks.append(chunk)

    except SyntaxError as e:
        print(f"⚠️  Could not parse {file_path}: {e}")
        # Fallback: treat whole file as one chunk
        chunks.append(_fallback_chunk(file_info))

    # If no functions/classes found (e.g. script files), fallback
    if not chunks:
        chunks.append(_fallback_chunk(file_info))

    return chunks


def chunk_generic_file(file_info: dict) -> list[dict]:
    """
    For non-Python files (JS, TS, Java, etc.)
    We use regex-based chunking to detect function boundaries.
    """
    content = file_info["content"]
    file_path = file_info["file_path"]
    language = file_info["language"]
    chunks = []

    # Regex patterns to detect function/method starts
    patterns = [
        r"((?:export\s+)?(?:async\s+)?function\s+\w+)",      # JS/TS functions
        r"((?:public|private|protected|static|\s)+\w+\s+\w+\s*\()",  # Java methods
        r"(func\s+\w+\s*\()",                                  # Go functions
        r"(def\s+\w+\s*\()",                                   # Ruby/Python fallback
        r"(fn\s+\w+\s*\()",                                    # Rust functions
    ]

    combined_pattern = "|".join(patterns)
    lines = content.splitlines()

    # Find all lines where a new function starts
    function_start_lines = []
    for i, line in enumerate(lines):
        if re.search(combined_pattern, line):
            function_start_lines.append(i)

    # If no functions found, return whole file as one chunk
    if not function_start_lines:
        chunks.append(_fallback_chunk(file_info))
        return chunks

    # Slice content between function starts
    for idx, start in enumerate(function_start_lines):
        end = function_start_lines[idx + 1] if idx + 1 < len(function_start_lines) else len(lines)
        chunk_lines = lines[start:end]
        chunk_code = "\n".join(chunk_lines)

        # Extract function name from first line
        name_match = re.search(r"(?:function|func|def|fn)\s+(\w+)", lines[start])
        func_name = name_match.group(1) if name_match else f"block_{start}"

        chunks.append({
            "chunk_id": f"{file_path}::{func_name}",
            "file_path": file_path,
            "language": language,
            "chunk_type": "function",
            "name": func_name,
            "start_line": start + 1,
            "end_line": end,
            "code": chunk_code,
            "char_count": len(chunk_code)
        })

    return chunks


def chunk_file(file_info: dict) -> list[dict]:
    """
    Main entry point. Routes to the right chunker based on language.
    """
    language = file_info["language"]

    if language == "python":
        return chunk_python_file(file_info)
    else:
        return chunk_generic_file(file_info)


def chunk_all_files(files: list[dict]) -> list[dict]:
    """
    Chunk all loaded files. Returns flat list of all chunks.
    """
    all_chunks = []

    for file_info in files:
        try:
            chunks = chunk_file(file_info)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️  Failed to chunk {file_info['file_path']}: {e}")

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _extract_node_chunk(node, lines, file_path, language, chunk_type) -> dict:
    """Extract a clean chunk dict from an AST node."""
    start = node.lineno - 1
    end = node.end_lineno
    code = "\n".join(lines[start:end])

    return {
        "chunk_id": f"{file_path}::{node.name}",
        "file_path": file_path,
        "language": language,
        "chunk_type": chunk_type,
        "name": node.name,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "code": code,
        "char_count": len(code)
    }


def _fallback_chunk(file_info: dict) -> dict:
    """When we can't parse properly, treat whole file as one chunk."""
    return {
        "chunk_id": f"{file_info['file_path']}::__module__",
        "file_path": file_info["file_path"],
        "language": file_info["language"],
        "chunk_type": "module",
        "name": "__module__",
        "start_line": 1,
        "end_line": len(file_info["content"].splitlines()),
        "code": file_info["content"],
        "char_count": len(file_info["content"])
    }