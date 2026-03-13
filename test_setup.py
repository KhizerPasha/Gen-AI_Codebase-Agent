# test_setup.py
import sys
print(f"✅ Python version: {sys.version}")

print("Checking imports...")

try:
    from langchain_openai import ChatOpenAI
    print("✅ langchain_openai — OK")
except ImportError as e:
    print(f"❌ langchain_openai — FAILED: {e}")

try:
    import faiss
    print("✅ faiss — OK")
except ImportError as e:
    print(f"❌ faiss — FAILED: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence_transformers — OK")
except ImportError as e:
    print(f"❌ sentence_transformers — FAILED: {e}")

try:
    import chromadb
    print("✅ chromadb — OK")
except ImportError as e:
    print(f"❌ chromadb — FAILED: {e}")

try:
    import langgraph
    print("✅ langgraph — OK")
except ImportError as e:
    print(f"❌ langgraph — FAILED: {e}")

try:
    import streamlit
    print("✅ streamlit — OK")
except ImportError as e:
    print(f"❌ streamlit — FAILED: {e}")

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if key and key.startswith("sk-"):
        print("✅ .env + API key — OK")
    else:
        print("⚠️  .env loaded but API key missing or invalid")
except Exception as e:
    print(f"❌ dotenv — FAILED: {e}")

print("\n🎉 Phase 0 complete! You're ready for Phase 1.")