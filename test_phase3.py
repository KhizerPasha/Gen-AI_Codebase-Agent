# test_phase3.py
from llm.analyzer import CodeAnalyzer

print("=" * 55)
print("🤖 PHASE 3 TEST — LLM Integration")
print("=" * 55)

analyzer = CodeAnalyzer()

# ── Test 1: Explain ───────────────────────────────────────
print("\n" + "="*55)
print("TEST 1: EXPLAIN")
print("="*55)
result = analyzer.explain("how does the file loader work?")
print(result)

# ── Test 2: Bug Finder ────────────────────────────────────
print("\n" + "="*55)
print("TEST 2: BUG FINDER")
print("="*55)
result = analyzer.find_bugs("file loading and chunking")
print(result)

# ── Test 3: Refactor ──────────────────────────────────────
print("\n" + "="*55)
print("TEST 3: REFACTOR")
print("="*55)
result = analyzer.refactor("chunk_all_files function")
print(result)

# ── Test 4: Find Usage ────────────────────────────────────
print("\n" + "="*55)
print("TEST 4: FIND USAGE")
print("="*55)
result = analyzer.find_usage("load_from_directory")
print(result)

print("\n✅ Phase 3 complete! LLM is analyzing code.")