# ui/app.py
import streamlit as st
import sys
import os

# Make sure imports work from ui/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import run_ingestion_pipeline
from agent.graph import build_agent, run_agent

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Codebase Agent",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .stat-box {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .user-msg {
        background-color: #2d2d3f;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .agent-msg {
        background-color: #1a2f1a;
        border-left: 3px solid #4CAF50;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "codebase_stats" not in st.session_state:
    st.session_state.codebase_stats = None

if "codebase_loaded" not in st.session_state:
    st.session_state.codebase_loaded = False


# ── Header ────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🧠 Codebase Explainer & Bug Finder</div>',
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Powered by RAG + GPT-4 + LangGraph</p>",
    unsafe_allow_html=True
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Codebase")

    source_type = st.radio(
        "Source type:",
        ["Local Folder", "GitHub URL"],
        horizontal=True
    )

    if source_type == "Local Folder":
        source_input = st.text_input(
            "Folder path:",
            placeholder="C:/projects/my-app  or  .",
            help="Enter the full path to your project folder. Use '.' for current directory."
        )
    else:
        source_input = st.text_input(
            "GitHub URL:",
            placeholder="https://github.com/username/repo",
            help="Public GitHub repositories only"
        )

    analyze_btn = st.button(
        "🚀 Analyze Codebase",
        type="primary",
        use_container_width=True
    )

    # ── Handle Analyze Button ─────────────────────────────
    if analyze_btn:
        if not source_input.strip():
            st.error("Please enter a path or URL!")
        else:
            with st.spinner("🔍 Loading & indexing codebase..."):
                try:
                    is_github = source_type == "GitHub URL"
                    stats = run_ingestion_pipeline(
                        source_input.strip(),
                        is_github=is_github
                    )

                    if "error" in stats:
                        st.error(stats["error"])
                    else:
                        # Build agent with fresh index
                        st.session_state.agent = build_agent()
                        st.session_state.codebase_stats = stats
                        st.session_state.codebase_loaded = True
                        st.session_state.chat_history = []
                        st.success("✅ Codebase loaded!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.divider()

    # ── Codebase Stats ────────────────────────────────────
    if st.session_state.codebase_stats:
        stats = st.session_state.codebase_stats
        st.subheader("📊 Codebase Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", stats["total_files"])
        with col2:
            st.metric("Chunks", stats["total_chunks"])

        st.markdown("**Languages:**")
        for lang, count in stats["languages"].items():
            st.markdown(f"- `{lang}`: {count} files")

        st.markdown("**Chunk types:**")
        for ctype, count in stats["chunk_types"].items():
            st.markdown(f"- `{ctype}`: {count}")

    st.divider()

    # ── Quick Actions ─────────────────────────────────────
    st.subheader("⚡ Quick Actions")
    st.markdown("Click to auto-fill a query:")

    quick_queries = [
        "Explain the overall architecture",
        "Find bugs and potential issues",
        "What are the main functions?",
        "Suggest refactoring improvements",
        "Where is error handling missing?",
    ]

    for q in quick_queries:
        if st.button(q, use_container_width=True, disabled=not st.session_state.codebase_loaded):
            st.session_state.prefill_query = q


# ── Main Chat Area ────────────────────────────────────────
if not st.session_state.codebase_loaded:

    # Welcome screen
    st.markdown("### 👈 Start by loading a codebase from the sidebar")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**💡 Explain**\n\nAsk what any function, class, or module does in plain English")

    with col2:
        st.warning("**🐛 Find Bugs**\n\nDetect potential bugs, bad patterns, and missing error handling")

    with col3:
        st.success("**🔧 Refactor**\n\nGet concrete suggestions to improve code quality")

    st.markdown("---")
    st.markdown("### 💬 Example Questions You Can Ask")

    examples = [
        "🔍 `explain how the authentication system works`",
        "🐛 `find bugs in the database connection code`",
        "🔧 `how can I refactor the API handler?`",
        "📍 `where is the User class used?`",
        "⚠️  `what error handling is missing?`",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")

else:

    # ── Chat Interface ─────────────────────────────────────
    st.subheader("💬 Chat with your codebase")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(message["content"])

    # Check for prefilled query from quick actions
    prefill = st.session_state.pop("prefill_query", "")

    # Chat input
    user_query = st.chat_input(
        "Ask anything about your codebase...",
        key="chat_input"
    ) or prefill

    if user_query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        # Get agent response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🧠 Thinking..."):
                try:
                    response = run_agent(
                        st.session_state.agent,
                        user_query
                    )
                    st.markdown(response)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:
                    err_msg = f"❌ Error: {str(e)}"
                    st.error(err_msg)

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()