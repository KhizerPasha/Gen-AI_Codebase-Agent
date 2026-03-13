# test_phase1.py
from ingestion.loader import load_from_directory, summarize_loaded_files
from ingestion.chunker import chunk_all_files

# We'll test on OUR OWN project folder!
print("=" * 50)
print("📂 PHASE 1 TEST — Loading & Chunking")
print("=" * 50)

# Step 1: Load files
files = load_from_directory(".")
summarize_loaded_files(files)

# Step 2: Chunk them
chunks = chunk_all_files(files)

# Step 3: Preview first 3 chunks
print("\n🔍 Preview of first 3 chunks:\n")
for chunk in chunks[:3]:
    print(f"  📌 Name      : {chunk['name']}")
    print(f"     File      : {chunk['file_path']}")
    print(f"     Type      : {chunk['chunk_type']}")
    print(f"     Lines     : {chunk['start_line']} → {chunk['end_line']}")
    print(f"     Chars     : {chunk['char_count']}")
    print(f"     Code preview: {chunk['code'][:80].strip()}...")
    print()

print(f"✅ Phase 1 complete! {len(chunks)} chunks ready for embedding.")