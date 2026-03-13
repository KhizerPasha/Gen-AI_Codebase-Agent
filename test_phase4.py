# test_phase4.py
from agent.graph import build_agent, run_agent

print("=" * 55)
print("🧠 PHASE 4 TEST — LangGraph Agent")
print("=" * 55)

# Build the agent
agent = build_agent()

# ── Test 1: Explain ───────────────────────────────────────
print("\n📌 TEST 1: Explanation Query")
run_agent(agent, "explain how the file loader works")

# ── Test 2: Bug Finding ───────────────────────────────────
print("\n📌 TEST 2: Bug Finding Query")
run_agent(agent, "find any bugs in the chunker code")

# ── Test 3: Find Usage ────────────────────────────────────
print("\n📌 TEST 3: Usage Query")
run_agent(agent, "where is chunk_all_files used in the project?")

# ── Test 4: Refactor ──────────────────────────────────────
print("\n📌 TEST 4: Refactor Query")
run_agent(agent, "how can I improve the embedding pipeline?")

print("\n✅ Phase 4 complete! Agent is reasoning and acting.")