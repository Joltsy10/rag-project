from src.document_loader import load_and_chunk
from src.embeddings import load_embedding_model
from src.vector_store import add_documents, similarity_search

# Load and chunk a document
sources = [
    {"type": "pdf", "path": "data/learning.pdf"},  # replace with your filename
]
chunks = load_and_chunk(sources)

# Load embedding model
print("Loading embedding model...")
embedding_model = load_embedding_model()

# Add to vector store
print("Adding to vector store...")
add_documents(chunks, embedding_model)

# Test similarity search
query = "What does machine learning mean according to Mitchell?"
print(f"\nQuery: {query}")
results = similarity_search(query, k=4, embedding_model=embedding_model)

print(f"\nTop {len(results)} results:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('file_name') or doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:200]}")
    print()

# Test duplicate prevention
print("Testing duplicate prevention - adding same document again...")
add_documents(chunks, embedding_model)