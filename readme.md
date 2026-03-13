# 🧠 RAG-Based Codebase Explainer & Bug Finder

An AI agent that analyzes any codebase using 
Retrieval-Augmented Generation (RAG) + LangGraph.

## Features
- 💡 **Explain** any function or class in plain English
- 🐛 **Find Bugs** and bad patterns with severity ratings
- 🔧 **Refactor** suggestions with improved code examples
- 🔍 **Find Usages** of any function across the codebase
- 🌐 Supports GitHub repos and local folders
- 🐍 Smart AST-based chunking (not naive token splitting)

## Tech Stack
| Component     | Technology                        |
|---------------|-----------------------------------|
| LLM           | GPT-4o-mini (OpenAI)              |
| Embeddings    | sentence-transformers/all-MiniLM  |
| Vector Store  | FAISS                             |
| Agent         | LangGraph (ReAct loop)            |
| RAG Framework | LangChain                         |
| UI            | Streamlit                         |

## Architecture
```
GitHub / Local Repo
      ↓
AST-based Code Chunker   ← splits by function/class
      ↓
Sentence Transformer     ← embeds each chunk
      ↓
FAISS Vector Store       ← stores + indexes vectors
      ↓
LangGraph ReAct Agent    ← thinks → retrieves → answers
      ↓
GPT-4o-mini              ← generates final response
      ↓
Streamlit UI             ← chat interface
```

## Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/codebase-agent
cd codebase-agent
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Add API Key
```bash
# Create .env file
echo OPENAI_API_KEY=sk-your-key > .env
```

### 3. Run
```bash
python run.py
# Opens at http://localhost:8501
```

## Usage
1. Enter a **GitHub URL** or **local folder path** in the sidebar
2. Click **Analyze Codebase** and wait for indexing
3. Ask anything in the chat:
   - `explain how the auth middleware works`
   - `find bugs in the database module`
   - `where is the User class used?`
   - `suggest refactors for the API handler`

## Project Structure
```
codebase-agent/
├── ingestion/
│   ├── loader.py       # Load files from GitHub/local
│   ├── chunker.py      # AST-based code chunking
│   ├── embedder.py     # Embed + store in FAISS
│   └── pipeline.py     # Master ingestion pipeline
├── retrieval/
│   └── retriever.py    # Semantic search over code
├── agent/
│   ├── tools.py        # explain/bugs/refactor/usage tools
│   └── graph.py        # LangGraph ReAct agent
├── llm/
│   ├── client.py       # LLM wrapper
│   ├── prompts.py      # All prompt templates
│   └── analyzer.py     # High-level analysis interface
├── ui/
│   └── app.py          # Streamlit chat UI
└── run.py              # Entry point
```