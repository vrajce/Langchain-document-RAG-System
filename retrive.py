import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Initialize the SAME Embedding Model used for Ingestion
# (If you change this, the "language" won't match and search will fail)
print("🔌 Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load the "Frozen" Index from Disk
DB_NAME = "faiss_index_react"  # Make sure this matches your folder name

print(f"📂 Loading FAISS Index from {DB_NAME}...")
try:
    vectorstore = FAISS.load_local(
        DB_NAME, 
        embeddings, 
        allow_dangerous_deserialization=True # Required for loading local pickle files
    )
    print("✅ Index Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading index: {e}")
    exit()

# 3. The Search Function
def search(query):
    print(f"\n🔎 Searching for: '{query}'")
    
    # "k=3" means "Find the top 3 most relevant chunks"
    results = vectorstore.similarity_search(query, k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        # Print Metadata (Source & Ecosystem)
        print(f"📄 Source: {res.metadata.get('source', 'Unknown')}")
        print(f"🏷️  Tag: {res.metadata.get('ecosystem', 'General')}")
        # Print a snippet of the content
        print(f"📝 Content: {res.page_content[:300]}...") # First 300 chars

# 4. Run a Test!
# Try a specific query to see if it finds the right "Ecosystem"
search("How do I create a graph in LangGraph?")