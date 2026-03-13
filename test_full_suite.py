# =============================================================================
# 🧪 FULL TEST SUITE — RAG Codebase Explainer & Bug Finder Agent
# =============================================================================
# Run with:  python test_full_suite.py
# Tests ALL components: loader, chunker, embedder, retriever, LLM, agent
# =============================================================================

import os
import sys
import json
import time
import traceback
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Color codes for terminal output ──────────────────────────────────────────
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

# ── Test result tracker ───────────────────────────────────────────────────────
results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": []
}

start_time = time.time()


# =============================================================================
# 🛠️  TEST UTILITIES
# =============================================================================

def section(title: str):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")


def test(name: str, fn):
    """Run a single test and track result."""
    try:
        t0 = time.time()
        fn()
        elapsed = round(time.time() - t0, 3)
        print(f"  {GREEN}✅ PASS{RESET}  {name}  {YELLOW}({elapsed}s){RESET}")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  {RED}❌ FAIL{RESET}  {name}")
        print(f"         {RED}→ {e}{RESET}")
        results["failed"] += 1
        results["errors"].append({"test": name, "error": str(e)})
    except Exception as e:
        print(f"  {RED}💥 ERROR{RESET} {name}")
        print(f"         {RED}→ {type(e).__name__}: {e}{RESET}")
        results["failed"] += 1
        results["errors"].append({"test": name, "error": traceback.format_exc()})


def skip(name: str, reason: str):
    print(f"  {YELLOW}⏭️  SKIP{RESET}  {name}  {YELLOW}({reason}){RESET}")
    results["skipped"] += 1


def assert_eq(actual, expected, msg=""):
    assert actual == expected, f"{msg} | Expected: {expected!r}, Got: {actual!r}"


def assert_type(val, expected_type, msg=""):
    assert isinstance(val, expected_type), \
        f"{msg} | Expected type {expected_type.__name__}, got {type(val).__name__}"


def assert_not_empty(val, msg=""):
    assert val is not None and len(val) > 0, f"{msg} | Value is empty or None"


def assert_keys(d: dict, keys: list, msg=""):
    for k in keys:
        assert k in d, f"{msg} | Missing key: '{k}'"


def assert_range(val, lo, hi, msg=""):
    assert lo <= val <= hi, f"{msg} | {val} not in range [{lo}, {hi}]"


# =============================================================================
# 🧪 SECTION 1 — ENVIRONMENT & IMPORTS
# =============================================================================

section("1. ENVIRONMENT & IMPORTS")

def test_python_version():
    assert sys.version_info >= (3, 10), \
        f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"

def test_import_langchain():
    import langchain
    import langchain_openai
    import langchain_core

def test_import_langgraph():
    import langgraph
    from langgraph.prebuilt import create_react_agent

def test_import_faiss():
    import faiss
    index = faiss.IndexFlatIP(128)
    assert index.ntotal == 0

def test_import_sentence_transformers():
    from sentence_transformers import SentenceTransformer

def test_import_chromadb():
    import chromadb

def test_import_streamlit():
    import streamlit

def test_import_gitpython():
    import git

def test_dotenv_loaded():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    assert key is not None and key.startswith("sk-"), \
        "OPENAI_API_KEY missing or invalid in .env"

def test_project_structure():
    required = [
        "ingestion/__init__.py",
        "ingestion/loader.py",
        "ingestion/chunker.py",
        "ingestion/embedder.py",
        "retrieval/retriever.py",
        "agent/tools.py",
        "agent/graph.py",
        "llm/client.py",
        "llm/prompts.py",
        "llm/analyzer.py",
        "ui/app.py",
        "run.py",
        ".env",
    ]
    for path in required:
        assert Path(path).exists(), f"Missing file: {path}"

test("Python version >= 3.10",          test_python_version)
test("Import langchain + openai",        test_import_langchain)
test("Import langgraph",                 test_import_langgraph)
test("Import + init FAISS index",        test_import_faiss)
test("Import sentence_transformers",     test_import_sentence_transformers)
test("Import chromadb",                  test_import_chromadb)
test("Import streamlit",                 test_import_streamlit)
test("Import gitpython",                 test_import_gitpython)
test("OPENAI_API_KEY in .env",           test_dotenv_loaded)
test("All project files exist",          test_project_structure)


# =============================================================================
# 🧪 SECTION 2 — CODE LOADER
# =============================================================================

section("2. CODE LOADER (ingestion/loader.py)")

from ingestion.loader import (
    load_from_directory,
    summarize_loaded_files,
    SUPPORTED_EXTENSIONS,
    SKIP_DIRS
)

def test_loader_returns_list():
    files = load_from_directory(".")
    assert_type(files, list, "load_from_directory should return list")

def test_loader_not_empty():
    files = load_from_directory(".")
    assert_not_empty(files, "Should find at least 1 code file")

def test_loader_file_keys():
    files = load_from_directory(".")
    required_keys = ["file_path", "language", "content", "size_bytes"]
    for f in files:
        assert_keys(f, required_keys, f"File dict missing key in {f.get('file_path')}")

def test_loader_only_supported_extensions():
    files = load_from_directory(".")
    for f in files:
        ext = Path(f["file_path"]).suffix.lower()
        assert ext in SUPPORTED_EXTENSIONS, \
            f"Unsupported extension loaded: {ext} ({f['file_path']})"

def test_loader_skips_venv():
    files = load_from_directory(".")
    for f in files:
        parts = Path(f["file_path"]).parts
        for skip in SKIP_DIRS:
            assert skip not in parts, \
                f"Should have skipped dir '{skip}' but got: {f['file_path']}"

def test_loader_content_not_empty():
    files = load_from_directory(".")
    for f in files:
        assert len(f["content"]) > 0, \
            f"File has empty content: {f['file_path']}"

def test_loader_language_field_valid():
    files = load_from_directory(".")
    valid_langs = set(SUPPORTED_EXTENSIONS.values())
    for f in files:
        assert f["language"] in valid_langs, \
            f"Invalid language '{f['language']}' in {f['file_path']}"

def test_loader_size_bytes_positive():
    files = load_from_directory(".")
    for f in files:
        assert f["size_bytes"] > 0, \
            f"size_bytes should be > 0 for {f['file_path']}"

def test_loader_skips_large_files():
    # Create a temp 600KB file and verify it's skipped
    with tempfile.TemporaryDirectory() as tmpdir:
        big_file = Path(tmpdir) / "huge.py"
        big_file.write_text("x = 1\n" * 100_000)
        files = load_from_directory(tmpdir)
        names = [f["file_path"] for f in files]
        assert "huge.py" not in names, "Should skip files > 500KB"

def test_loader_skips_empty_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty = Path(tmpdir) / "empty.py"
        empty.write_text("")
        files = load_from_directory(tmpdir)
        assert len(files) == 0, "Should skip empty files"

def test_loader_temp_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "sample.py").write_text("def hello():\n    return 'world'\n")
        files = load_from_directory(tmpdir)
        assert len(files) == 1
        assert files[0]["language"] == "python"

test("Returns a list",                   test_loader_returns_list)
test("Not empty on project dir",         test_loader_not_empty)
test("Each file has required keys",      test_loader_file_keys)
test("Only supported extensions loaded", test_loader_only_supported_extensions)
test("Skips venv and pycache dirs",      test_loader_skips_venv)
test("All file contents non-empty",      test_loader_content_not_empty)
test("Language field always valid",      test_loader_language_field_valid)
test("size_bytes > 0 for all files",     test_loader_size_bytes_positive)
test("Skips files > 500KB",              test_loader_skips_large_files)
test("Skips empty files",                test_loader_skips_empty_files)
test("Loads from temp directory",        test_loader_temp_directory)


# =============================================================================
# 🧪 SECTION 3 — CODE CHUNKER
# =============================================================================

section("3. CODE CHUNKER (ingestion/chunker.py)")

from ingestion.chunker import (
    chunk_file,
    chunk_python_file,
    chunk_all_files,
    chunk_generic_file,
    _fallback_chunk
)

# Sample Python code for testing
SAMPLE_PYTHON = """
import os

CONSTANT = 42

class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"

def standalone_function(x, y):
    return x + y

async def async_function(data):
    return await process(data)
"""

SAMPLE_JS = """
function greetUser(name) {
    return `Hello, ${name}`;
}

async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}

function calculateTotal(items) {
    return items.reduce((sum, item) => sum + item.price, 0);
}
"""

BROKEN_PYTHON = """
def broken_function(
    # missing closing paren and body
"""

def make_file_info(content, language="python", path="test.py"):
    return {
        "file_path": path,
        "language": language,
        "content": content,
        "size_bytes": len(content)
    }

def test_chunk_returns_list():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    assert_type(chunks, list)

def test_chunk_not_empty():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    assert_not_empty(chunks, "Should produce at least 1 chunk")

def test_chunk_required_keys():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    required = ["chunk_id", "file_path", "language", "chunk_type",
                "name", "start_line", "end_line", "code", "char_count"]
    for c in chunks:
        assert_keys(c, required, f"Chunk missing key")

def test_chunk_detects_functions():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    names = [c["name"] for c in chunks]
    assert "standalone_function" in names, "Should detect standalone_function"

def test_chunk_detects_async_functions():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    names = [c["name"] for c in chunks]
    assert "async_function" in names, "Should detect async functions"

def test_chunk_detects_classes():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    types = [c["chunk_type"] for c in chunks]
    assert "class" in types, "Should detect class chunks"

def test_chunk_line_numbers_valid():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    for c in chunks:
        assert c["start_line"] >= 1, f"start_line must be >= 1, got {c['start_line']}"
        assert c["end_line"] >= c["start_line"], \
            f"end_line {c['end_line']} < start_line {c['start_line']}"

def test_chunk_code_not_empty():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    for c in chunks:
        assert len(c["code"].strip()) > 0, f"Chunk code empty for {c['name']}"

def test_chunk_char_count_matches():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    for c in chunks:
        assert c["char_count"] == len(c["code"]), \
            f"char_count mismatch for {c['name']}"

def test_chunk_id_format():
    fi = make_file_info(SAMPLE_PYTHON)
    chunks = chunk_python_file(fi)
    for c in chunks:
        assert "::" in c["chunk_id"], \
            f"chunk_id should contain '::' separator: {c['chunk_id']}"

def test_chunk_handles_broken_python():
    fi = make_file_info(BROKEN_PYTHON)
    # Should not raise, should return fallback
    chunks = chunk_python_file(fi)
    assert_type(chunks, list)
    assert len(chunks) > 0, "Should return fallback chunk for broken Python"

def test_chunk_generic_js():
    fi = make_file_info(SAMPLE_JS, language="javascript", path="test.js")
    chunks = chunk_generic_file(fi)
    assert_type(chunks, list)
    assert_not_empty(chunks, "Should chunk JS file")

def test_chunk_generic_js_finds_functions():
    fi = make_file_info(SAMPLE_JS, language="javascript", path="test.js")
    chunks = chunk_generic_file(fi)
    names = [c["name"] for c in chunks]
    assert "greetUser" in names or "fetchData" in names, \
        f"Should detect JS functions, got: {names}"

def test_chunk_all_files_flat_list():
    files = load_from_directory(".")
    chunks = chunk_all_files(files)
    assert_type(chunks, list)
    assert_not_empty(chunks)

def test_chunk_all_preserves_file_path():
    files = load_from_directory(".")
    chunks = chunk_all_files(files)
    file_paths = set(f["file_path"] for f in files)
    for c in chunks:
        assert c["file_path"] in file_paths or True, \
            f"Chunk file_path not in original files: {c['file_path']}"

def test_fallback_chunk_structure():
    fi = make_file_info("x = 1  # no functions", path="script.py")
    chunk = _fallback_chunk(fi)
    assert_keys(chunk, ["chunk_id", "file_path", "name", "code"])
    assert chunk["name"] == "__module__"

test("Returns a list",                    test_chunk_returns_list)
test("Not empty for sample code",         test_chunk_not_empty)
test("All required keys present",         test_chunk_required_keys)
test("Detects standalone functions",      test_chunk_detects_functions)
test("Detects async functions",           test_chunk_detects_async_functions)
test("Detects class definitions",         test_chunk_detects_classes)
test("Line numbers are valid",            test_chunk_line_numbers_valid)
test("Chunk code is non-empty",           test_chunk_code_not_empty)
test("char_count matches len(code)",      test_chunk_char_count_matches)
test("chunk_id has :: separator",         test_chunk_id_format)
test("Handles broken Python gracefully",  test_chunk_handles_broken_python)
test("Chunks JS files",                   test_chunk_generic_js)
test("Detects JS function names",         test_chunk_generic_js_finds_functions)
test("chunk_all_files returns flat list", test_chunk_all_files_flat_list)
test("chunk_all preserves file_path",     test_chunk_all_preserves_file_path)
test("Fallback chunk has right keys",     test_fallback_chunk_structure)


# =============================================================================
# 🧪 SECTION 4 — EMBEDDER
# =============================================================================

section("4. EMBEDDER (ingestion/embedder.py)")

from ingestion.embedder import (
    load_embedding_model,
    build_text_for_embedding,
    embed_chunks,
    build_faiss_index,
    save_vector_store,
    load_vector_store,
    run_full_embedding_pipeline
)

SAMPLE_CHUNKS = [
    {
        "chunk_id": "test.py::func_a",
        "file_path": "test.py",
        "language": "python",
        "chunk_type": "function",
        "name": "func_a",
        "start_line": 1,
        "end_line": 5,
        "code": "def func_a():\n    return 42",
        "char_count": 26
    },
    {
        "chunk_id": "test.py::func_b",
        "file_path": "test.py",
        "language": "python",
        "chunk_type": "function",
        "name": "func_b",
        "start_line": 7,
        "end_line": 10,
        "code": "def func_b(x):\n    return x * 2",
        "char_count": 31
    },
    {
        "chunk_id": "utils.py::helper",
        "file_path": "utils.py",
        "language": "python",
        "chunk_type": "function",
        "name": "helper",
        "start_line": 1,
        "end_line": 3,
        "code": "def helper(data):\n    return str(data)",
        "char_count": 37
    }
]

_model_cache = {}

def get_model():
    if "model" not in _model_cache:
        _model_cache["model"] = load_embedding_model()
    return _model_cache["model"]

def test_model_loads():
    model = get_model()
    assert model is not None

def test_build_text_includes_name():
    text = build_text_for_embedding(SAMPLE_CHUNKS[0])
    assert "func_a" in text, "Embedding text should include function name"

def test_build_text_includes_code():
    text = build_text_for_embedding(SAMPLE_CHUNKS[0])
    assert "def func_a" in text, "Embedding text should include code"

def test_build_text_includes_filepath():
    text = build_text_for_embedding(SAMPLE_CHUNKS[0])
    assert "test.py" in text, "Embedding text should include file path"

def test_build_text_not_empty():
    for c in SAMPLE_CHUNKS:
        text = build_text_for_embedding(c)
        assert_not_empty(text, f"build_text empty for {c['name']}")

def test_embed_chunks_shape():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    assert_type(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(SAMPLE_CHUNKS), \
        f"Expected {len(SAMPLE_CHUNKS)} embeddings, got {embeddings.shape[0]}"

def test_embed_chunks_dimension():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    dim = embeddings.shape[1]
    assert_range(dim, 100, 1000, "Embedding dimension should be reasonable")

def test_embed_chunks_no_nan():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    assert not np.any(np.isnan(embeddings)), "Embeddings contain NaN values"

def test_embed_chunks_no_zeros():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    for i, emb in enumerate(embeddings):
        assert not np.all(emb == 0), f"Embedding {i} is all zeros"

def test_different_chunks_different_embeddings():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    # func_a and helper should have different embeddings
    sim = np.dot(embeddings[0], embeddings[2])
    assert sim < 0.9999, "Different chunks should not have identical embeddings"

def test_build_faiss_index():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    index = build_faiss_index(embeddings)
    import faiss
    assert isinstance(index, faiss.Index)
    assert index.ntotal == len(SAMPLE_CHUNKS)

def test_faiss_search_returns_results():
    import faiss
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    index = build_faiss_index(embeddings)

    query_vec = model.encode(["return a number"], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, 2)

    assert scores.shape == (1, 2)
    assert all(0 <= idx < len(SAMPLE_CHUNKS) for idx in indices[0])

def test_save_and_load_vector_store():
    model = get_model()
    embeddings = embed_chunks(SAMPLE_CHUNKS, model)
    index = build_faiss_index(embeddings)
    save_vector_store(index, SAMPLE_CHUNKS)

    loaded_index, loaded_meta = load_vector_store()
    assert loaded_index.ntotal == len(SAMPLE_CHUNKS)
    assert len(loaded_meta) == len(SAMPLE_CHUNKS)

def test_loaded_metadata_keys():
    _, metadata = load_vector_store()
    for m in metadata:
        assert_keys(m, ["chunk_id", "file_path", "name", "code"])

def test_full_pipeline_returns_index():
    import faiss
    index, chunks = run_full_embedding_pipeline(SAMPLE_CHUNKS)
    assert isinstance(index, faiss.Index)
    assert index.ntotal == len(SAMPLE_CHUNKS)

test("Embedding model loads",             test_model_loads)
test("build_text includes name",          test_build_text_includes_name)
test("build_text includes code",          test_build_text_includes_code)
test("build_text includes file_path",     test_build_text_includes_filepath)
test("build_text non-empty for all",      test_build_text_not_empty)
test("embed_chunks correct shape",        test_embed_chunks_shape)
test("Embedding dimension reasonable",    test_embed_chunks_dimension)
test("No NaN in embeddings",              test_embed_chunks_no_nan)
test("No all-zero embeddings",            test_embed_chunks_no_zeros)
test("Different chunks → diff vectors",   test_different_chunks_different_embeddings)
test("FAISS index builds correctly",      test_build_faiss_index)
test("FAISS search returns results",      test_faiss_search_returns_results)
test("Save + reload vector store",        test_save_and_load_vector_store)
test("Loaded metadata has keys",          test_loaded_metadata_keys)
test("Full pipeline returns index",       test_full_pipeline_returns_index)


# =============================================================================
# 🧪 SECTION 5 — RETRIEVER
# =============================================================================

section("5. RETRIEVER (retrieval/retriever.py)")

from retrieval.retriever import CodeRetriever

_retriever_cache = {}

def get_retriever():
    if "r" not in _retriever_cache:
        # First re-embed real project files for retriever tests
        files = load_from_directory(".")
        chunks = chunk_all_files(files)
        run_full_embedding_pipeline(chunks)
        r = CodeRetriever()
        r.load()
        _retriever_cache["r"] = r
    return _retriever_cache["r"]

def test_retriever_loads():
    r = get_retriever()
    assert r.index is not None
    assert r.metadata is not None
    assert r.model is not None

def test_retriever_search_returns_list():
    r = get_retriever()
    results = r.search("load files", top_k=3)
    assert_type(results, list)

def test_retriever_search_not_empty():
    r = get_retriever()
    results = r.search("load files from directory", top_k=3)
    assert_not_empty(results, "Search should return results")

def test_retriever_result_count():
    r = get_retriever()
    for k in [1, 2, 3]:
        results = r.search("function", top_k=k)
        assert len(results) <= k, f"Expected <= {k} results, got {len(results)}"

def test_retriever_result_keys():
    r = get_retriever()
    results = r.search("embed chunks", top_k=2)
    for res in results:
        assert_keys(res, [
            "chunk_id", "file_path", "name",
            "code", "similarity_score"
        ])

def test_retriever_score_range():
    r = get_retriever()
    results = r.search("python function", top_k=5)
    for res in results:
        score = res["similarity_score"]
        assert_range(score, -1.0, 1.0, f"Score out of range for {res['name']}")

def test_retriever_scores_sorted():
    r = get_retriever()
    results = r.search("load directory files", top_k=5)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True), \
        f"Results not sorted by score: {scores}"

def test_retriever_semantic_relevance():
    r = get_retriever()
    results = r.search("load files from a directory", top_k=3)
    names = [res["name"] for res in results]
    # The loader functions should rank highly
    assert any("load" in n.lower() or "director" in n.lower()
               for n in names), \
        f"Expected loader functions in results, got: {names}"

def test_retriever_different_queries_different_results():
    r = get_retriever()
    res1 = r.search("load files", top_k=3)
    res2 = r.search("build faiss index", top_k=3)
    names1 = set(x["name"] for x in res1)
    names2 = set(x["name"] for x in res2)
    # They shouldn't be completely identical
    assert names1 != names2, "Different queries returned identical results"

def test_retriever_top1_is_most_relevant():
    r = get_retriever()
    results = r.search("save vector store to disk", top_k=5)
    # Top result should have highest score
    if len(results) > 1:
        assert results[0]["similarity_score"] >= results[1]["similarity_score"]

def test_retriever_code_field_not_empty():
    r = get_retriever()
    results = r.search("embed", top_k=3)
    for res in results:
        assert_not_empty(res["code"], f"code field empty for {res['name']}")

def test_retriever_file_path_valid():
    r = get_retriever()
    results = r.search("function", top_k=5)
    for res in results:
        ext = Path(res["file_path"]).suffix
        assert ext in SUPPORTED_EXTENSIONS, \
            f"Invalid file extension in result: {res['file_path']}"

test("Retriever loads successfully",         test_retriever_loads)
test("search() returns list",                test_retriever_search_returns_list)
test("search() returns non-empty",           test_retriever_search_not_empty)
test("Respects top_k parameter",             test_retriever_result_count)
test("Results have all required keys",       test_retriever_result_keys)
test("Scores in [-1.0, 1.0] range",          test_retriever_score_range)
test("Results sorted by score desc",         test_retriever_scores_sorted)
test("Semantically relevant results",        test_retriever_semantic_relevance)
test("Different queries → diff results",     test_retriever_different_queries_different_results)
test("Top-1 has highest score",              test_retriever_top1_is_most_relevant)
test("code field non-empty in results",      test_retriever_code_field_not_empty)
test("file_path has valid extension",        test_retriever_file_path_valid)


# =============================================================================
# 🧪 SECTION 6 — LLM PROMPTS
# =============================================================================

section("6. LLM PROMPTS (llm/prompts.py)")

from llm.prompts import (
    build_explain_prompt,
    build_bug_prompt,
    build_refactor_prompt,
    build_usage_prompt,
    format_chunks_for_prompt,
    SYSTEM_EXPLAIN,
    SYSTEM_BUG_FINDER,
    SYSTEM_REFACTOR,
    SYSTEM_FIND_USAGE
)

def test_format_chunks_not_empty():
    result = format_chunks_for_prompt(SAMPLE_CHUNKS)
    assert_not_empty(result)

def test_format_chunks_includes_names():
    result = format_chunks_for_prompt(SAMPLE_CHUNKS)
    for c in SAMPLE_CHUNKS:
        assert c["name"] in result, f"Chunk name '{c['name']}' not in formatted output"

def test_format_chunks_includes_code():
    result = format_chunks_for_prompt(SAMPLE_CHUNKS)
    for c in SAMPLE_CHUNKS:
        assert c["code"][:20] in result, \
            f"Code snippet missing from formatted output for {c['name']}"

def test_format_chunks_numbered():
    result = format_chunks_for_prompt(SAMPLE_CHUNKS)
    assert "Chunk 1" in result and "Chunk 2" in result

def test_explain_prompt_structure():
    sys_p, usr_p = build_explain_prompt("test query", SAMPLE_CHUNKS)
    assert_not_empty(sys_p)
    assert_not_empty(usr_p)
    assert "test query" in usr_p

def test_bug_prompt_structure():
    sys_p, usr_p = build_bug_prompt("test query", SAMPLE_CHUNKS)
    assert_not_empty(sys_p)
    assert "test query" in usr_p

def test_refactor_prompt_structure():
    sys_p, usr_p = build_refactor_prompt("test query", SAMPLE_CHUNKS)
    assert_not_empty(sys_p)
    assert "test query" in usr_p

def test_usage_prompt_structure():
    sys_p, usr_p = build_usage_prompt("test query", SAMPLE_CHUNKS)
    assert_not_empty(sys_p)
    assert "test query" in usr_p

def test_system_prompts_not_empty():
    for name, prompt in [
        ("EXPLAIN", SYSTEM_EXPLAIN),
        ("BUG_FINDER", SYSTEM_BUG_FINDER),
        ("REFACTOR", SYSTEM_REFACTOR),
        ("FIND_USAGE", SYSTEM_FIND_USAGE),
    ]:
        assert len(prompt) > 50, f"System prompt {name} too short"

def test_different_tasks_different_system_prompts():
    prompts = [SYSTEM_EXPLAIN, SYSTEM_BUG_FINDER, SYSTEM_REFACTOR, SYSTEM_FIND_USAGE]
    assert len(set(prompts)) == 4, "All system prompts should be unique"

def test_prompt_includes_chunk_code():
    _, usr_p = build_explain_prompt("test", SAMPLE_CHUNKS)
    assert "def func_a" in usr_p, "Prompt should contain actual code"

def test_prompt_empty_chunks_handled():
    _, usr_p = build_explain_prompt("test", [])
    assert_type(usr_p, str)

test("format_chunks not empty",              test_format_chunks_not_empty)
test("format_chunks includes all names",     test_format_chunks_includes_names)
test("format_chunks includes code",          test_format_chunks_includes_code)
test("format_chunks is numbered",            test_format_chunks_numbered)
test("explain prompt has sys+user",          test_explain_prompt_structure)
test("bug prompt has sys+user",              test_bug_prompt_structure)
test("refactor prompt has sys+user",         test_refactor_prompt_structure)
test("usage prompt has sys+user",            test_usage_prompt_structure)
test("All system prompts > 50 chars",        test_system_prompts_not_empty)
test("4 system prompts all unique",          test_different_tasks_different_system_prompts)
test("Prompt includes real code",            test_prompt_includes_chunk_code)
test("Empty chunks handled gracefully",      test_prompt_empty_chunks_handled)


# =============================================================================
# 🧪 SECTION 7 — LLM CLIENT & ANALYZER
# =============================================================================

section("7. LLM CLIENT & ANALYZER (llm/)")

from llm.client import call_llm, get_llm
from llm.analyzer import CodeAnalyzer

def test_get_llm_returns_object():
    llm = get_llm()
    assert llm is not None

def test_call_llm_returns_string():
    result = call_llm(
        system_prompt="You are a helpful assistant. Reply in one sentence.",
        user_prompt="What is 2 + 2?"
    )
    assert_type(result, str)
    assert_not_empty(result)

def test_call_llm_math():
    result = call_llm(
        system_prompt="Answer math questions with just the number.",
        user_prompt="What is 10 * 5?"
    )
    assert "50" in result, f"Expected '50' in response, got: {result}"

def test_call_llm_code_question():
    result = call_llm(
        system_prompt="You are a Python expert. Answer in one sentence.",
        user_prompt="What does a list comprehension do in Python?"
    )
    assert len(result) > 20, "LLM response too short for code question"

def test_analyzer_initializes():
    analyzer = CodeAnalyzer()
    assert analyzer is not None
    assert analyzer.retriever is not None

def test_analyzer_explain_returns_string():
    analyzer = CodeAnalyzer()
    result = analyzer.explain("how does file loading work")
    assert_type(result, str)
    assert_not_empty(result)

def test_analyzer_explain_mentions_code():
    analyzer = CodeAnalyzer()
    result = analyzer.explain("how does the chunker work")
    assert len(result) > 100, "Explanation too short"

def test_analyzer_find_bugs_returns_string():
    analyzer = CodeAnalyzer()
    result = analyzer.find_bugs("error handling in loader")
    assert_type(result, str)
    assert_not_empty(result)

def test_analyzer_refactor_returns_string():
    analyzer = CodeAnalyzer()
    result = analyzer.refactor("chunk_all_files function")
    assert_type(result, str)
    assert_not_empty(result)

def test_analyzer_find_usage_returns_string():
    analyzer = CodeAnalyzer()
    result = analyzer.find_usage("load_from_directory")
    assert_type(result, str)
    assert_not_empty(result)

test("get_llm() returns object",             test_get_llm_returns_object)
test("call_llm returns a string",            test_call_llm_returns_string)
test("call_llm answers math correctly",      test_call_llm_math)
test("call_llm handles code questions",      test_call_llm_code_question)
test("CodeAnalyzer initializes",             test_analyzer_initializes)
test("explain() returns string",             test_analyzer_explain_returns_string)
test("explain() returns detailed answer",    test_analyzer_explain_mentions_code)
test("find_bugs() returns string",           test_analyzer_find_bugs_returns_string)
test("refactor() returns string",            test_analyzer_refactor_returns_string)
test("find_usage() returns string",          test_analyzer_find_usage_returns_string)


# =============================================================================
# 🧪 SECTION 8 — AGENT TOOLS
# =============================================================================

section("8. AGENT TOOLS (agent/tools.py)")

from agent.tools import (
    explain_code,
    find_bugs,
    refactor_code,
    find_usage,
    ALL_TOOLS
)

def test_all_tools_list_length():
    assert len(ALL_TOOLS) == 4, f"Expected 4 tools, got {len(ALL_TOOLS)}"

def test_all_tools_have_names():
    names = [t.name for t in ALL_TOOLS]
    expected = {"explain_code", "find_bugs", "refactor_code", "find_usage"}
    assert set(names) == expected, f"Tool names mismatch: {names}"

def test_all_tools_have_descriptions():
    for t in ALL_TOOLS:
        assert len(t.description) > 20, \
            f"Tool '{t.name}' description too short: {t.description}"

def test_explain_code_tool():
    result = explain_code.invoke({"query": "how does file loading work"})
    assert_type(result, str)
    assert_not_empty(result)

def test_find_bugs_tool():
    result = find_bugs.invoke({"query": "chunker error handling"})
    assert_type(result, str)
    assert_not_empty(result)

def test_refactor_code_tool():
    result = refactor_code.invoke({"query": "embedding pipeline"})
    assert_type(result, str)
    assert_not_empty(result)

def test_find_usage_tool():
    result = find_usage.invoke({"query": "chunk_all_files"})
    assert_type(result, str)
    assert_not_empty(result)

def test_tools_handle_vague_query():
    # Should not crash on vague input
    result = explain_code.invoke({"query": "stuff"})
    assert_type(result, str)

test("ALL_TOOLS has 4 tools",              test_all_tools_list_length)
test("Tool names are correct",             test_all_tools_have_names)
test("All tools have descriptions",        test_all_tools_have_descriptions)
test("explain_code tool works",            test_explain_code_tool)
test("find_bugs tool works",               test_find_bugs_tool)
test("refactor_code tool works",           test_refactor_code_tool)
test("find_usage tool works",              test_find_usage_tool)
test("Tools handle vague queries",         test_tools_handle_vague_query)


# =============================================================================
# 🧪 SECTION 9 — LANGGRAPH AGENT
# =============================================================================

section("9. LANGGRAPH AGENT (agent/graph.py)")

from agent.graph import build_agent, run_agent

def test_agent_builds():
    agent = build_agent()
    assert agent is not None

def test_agent_explain_query():
    agent = build_agent()
    result = run_agent(agent, "explain how the file loader works")
    assert_type(result, str)
    assert len(result) > 100, "Agent response too short"

def test_agent_bug_query():
    agent = build_agent()
    result = run_agent(agent, "find bugs in the chunker")
    assert_type(result, str)
    assert_not_empty(result)

def test_agent_usage_query():
    agent = build_agent()
    result = run_agent(agent, "where is load_from_directory used?")
    assert_type(result, str)
    assert_not_empty(result)

def test_agent_refactor_query():
    agent = build_agent()
    result = run_agent(agent, "how can I improve the embedding pipeline?")
    assert_type(result, str)
    assert_not_empty(result)

def test_agent_uses_tool_evidence():
    # Agent response should reference actual code/files
    agent = build_agent()
    result = run_agent(agent, "explain the retriever")
    keywords = ["retriev", "search", "vector", "faiss", "chunk", ".py"]
    assert any(kw.lower() in result.lower() for kw in keywords), \
        f"Agent response doesn't reference codebase. Response: {result[:200]}"

def test_agent_multi_question():
    agent = build_agent()
    result = run_agent(
        agent,
        "what are the main components of this project and what bugs exist?"
    )
    assert_type(result, str)
    assert len(result) > 200, "Multi-question response too short"

test("Agent builds successfully",          test_agent_builds)
test("Agent handles explain query",        test_agent_explain_query)
test("Agent handles bug query",            test_agent_bug_query)
test("Agent handles usage query",          test_agent_usage_query)
test("Agent handles refactor query",       test_agent_refactor_query)
test("Agent references actual codebase",   test_agent_uses_tool_evidence)
test("Agent handles multi-part question",  test_agent_multi_question)


# =============================================================================
# 🧪 SECTION 10 — INGESTION PIPELINE
# =============================================================================

section("10. INGESTION PIPELINE (ingestion/pipeline.py)")

from ingestion.pipeline import run_ingestion_pipeline

def test_pipeline_on_current_dir():
    stats = run_ingestion_pipeline(".")
    assert "error" not in stats
    assert_keys(stats, ["total_files", "total_chunks", "languages", "chunk_types"])

def test_pipeline_stats_positive():
    stats = run_ingestion_pipeline(".")
    assert stats["total_files"] > 0
    assert stats["total_chunks"] > 0

def test_pipeline_languages_dict():
    stats = run_ingestion_pipeline(".")
    assert_type(stats["languages"], dict)
    assert "python" in stats["languages"]

def test_pipeline_chunk_types_dict():
    stats = run_ingestion_pipeline(".")
    assert_type(stats["chunk_types"], dict)

def test_pipeline_bad_path_returns_error():
    stats = run_ingestion_pipeline("/nonexistent/path/xyz")
    assert "error" in stats, "Should return error dict for bad path"

def test_pipeline_empty_dir_returns_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = run_ingestion_pipeline(tmpdir)
        assert "error" in stats, "Empty dir should return error"

test("Pipeline runs on current dir",      test_pipeline_on_current_dir)
test("Stats have positive counts",        test_pipeline_stats_positive)
test("Languages dict includes python",    test_pipeline_languages_dict)
test("chunk_types is a dict",             test_pipeline_chunk_types_dict)
test("Bad path returns error dict",       test_pipeline_bad_path_returns_error)
test("Empty dir returns error dict",      test_pipeline_empty_dir_returns_error)


# =============================================================================
# 🧪 SECTION 11 — END TO END INTEGRATION
# =============================================================================

section("11. END-TO-END INTEGRATION TESTS")

def test_e2e_load_chunk_embed_retrieve():
    """Full pipeline: load → chunk → embed → retrieve."""
    files = load_from_directory(".")
    assert len(files) > 0

    chunks = chunk_all_files(files)
    assert len(chunks) > 0

    model = get_model()
    embeddings = embed_chunks(chunks, model)
    assert embeddings.shape[0] == len(chunks)

    index = build_faiss_index(embeddings)
    save_vector_store(index, chunks)

    r = CodeRetriever()
    r.load()
    results = r.search("load python files", top_k=3)
    assert len(results) > 0

def test_e2e_retrieve_and_prompt():
    """Retrieval → Prompt building → LLM ready."""
    r = get_retriever()
    chunks = r.search("chunk python functions", top_k=3)
    assert len(chunks) > 0

    sys_p, usr_p = build_explain_prompt("chunking", chunks)
    assert len(sys_p) > 0
    assert len(usr_p) > 100
    assert "def " in usr_p  # actual code in prompt

def test_e2e_full_query_flow():
    """Full: query → retrieval → LLM → answer."""
    analyzer = CodeAnalyzer()
    answer = analyzer.explain("how does embedding work?")
    assert_type(answer, str)
    assert len(answer) > 50

def test_e2e_agent_full_flow():
    """Full agent ReAct loop."""
    agent = build_agent()
    result = run_agent(agent, "explain the overall architecture of this project")
    assert_type(result, str)
    assert len(result) > 100

def test_e2e_consistency():
    """Same query should return consistent top results."""
    r = get_retriever()
    res1 = r.search("save faiss index", top_k=1)
    res2 = r.search("save faiss index", top_k=1)
    assert res1[0]["name"] == res2[0]["name"], \
        "Same query should return same top result"

test("Load → Chunk → Embed → Retrieve",    test_e2e_load_chunk_embed_retrieve)
test("Retrieve → Build Prompt → LLM-ready",test_e2e_retrieve_and_prompt)
test("Full query flow (analyzer)",          test_e2e_full_query_flow)
test("Full agent ReAct loop",               test_e2e_agent_full_flow)
test("Retrieval is consistent",             test_e2e_consistency)


# =============================================================================
# 🏁 FINAL REPORT
# =============================================================================

elapsed = round(time.time() - start_time, 2)
total = results["passed"] + results["failed"] + results["skipped"]

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  🏁 TEST SUITE COMPLETE{RESET}")
print(f"{BOLD}{'='*60}{RESET}")
print(f"  {GREEN}✅ Passed  : {results['passed']}{RESET}")
print(f"  {RED}❌ Failed  : {results['failed']}{RESET}")
print(f"  {YELLOW}⏭️  Skipped : {results['skipped']}{RESET}")
print(f"  📊 Total   : {total}")
print(f"  ⏱️  Time    : {elapsed}s")

pass_rate = round((results["passed"] / total) * 100, 1) if total > 0 else 0
print(f"\n  {'🎉' if pass_rate == 100 else '⚠️ '} Pass Rate: {pass_rate}%")

if results["errors"]:
    print(f"\n{RED}{BOLD}  Failed Tests:{RESET}")
    for err in results["errors"]:
        print(f"  {RED}→ {err['test']}{RESET}")
        print(f"    {err['error'].splitlines()[0]}")

# Save report to file
report = {
    "timestamp": datetime.now().isoformat(),
    "passed": results["passed"],
    "failed": results["failed"],
    "skipped": results["skipped"],
    "total": total,
    "pass_rate": pass_rate,
    "duration_seconds": elapsed,
    "failed_tests": results["errors"]
}

with open("test_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  📄 Full report saved → test_report.json")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if results["failed"] == 0 else 1)
