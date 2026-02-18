from src.document_loader import load_pdf, load_url, chunk_documents, load_and_chunk

# Test 1: Load a PDF and inspect a single Document object
print("=" * 50)
print("TEST 1: Single PDF document object")
print("=" * 50)
docs = load_pdf("data/syl.pdf")  # replace with your actual filename
print(f"Number of pages loaded: {len(docs)}")
print(f"\nFirst document page_content (first 200 chars):\n{docs[0].page_content[:200]}")
print(f"\nFirst document metadata:\n{docs[0].metadata}")

# Test 2: Chunk those documents and inspect a chunk
print("\n" + "=" * 50)
print("TEST 2: Chunked documents")
print("=" * 50)
chunks = chunk_documents(docs)
print(f"Number of chunks: {len(chunks)}")
print(f"\nFirst chunk content:\n{chunks[0].page_content}")
print(f"\nFirst chunk metadata:\n{chunks[0].metadata}")
print(f"\nLast chunk content:\n{chunks[-1].page_content}")

# Test 3: Load a URL
print("\n" + "=" * 50)
print("TEST 3: URL loading")
print("=" * 50)
url_docs = load_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
print(f"Documents loaded from URL: {len(url_docs)}")
print(f"\nContent preview:\n{url_docs[0].page_content[:300]}")
print(f"\nMetadata:\n{url_docs[0].metadata}")

# Test 4: Full multi-source pipeline
print("\n" + "=" * 50)
print("TEST 4: Multi-source load_and_chunk")
print("=" * 50)
sources = [
    {"type": "pdf", "path": "data/syl.pdf"},  # replace with your actual filename
    {"type": "url", "path": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}
]
all_chunks = load_and_chunk(sources)
print(f"\nTotal chunks from all sources: {len(all_chunks)}")
print(f"\nSample chunk from position 5:\n{all_chunks[6].page_content}")
print(f"\nMetadata of that chunk:\n{all_chunks[5].metadata}")

for i in range(5):
    chunk_a = all_chunks[i].page_content
    chunk_b = all_chunks[i+1].page_content
    
    # Take last 100 chars of chunk A and first 100 chars of chunk B
    end_of_a = chunk_a[-100:]
    start_of_b = chunk_b[:100:]
    
    print(f"\n--- Boundary between chunk {i} and chunk {i+1} ---")
    print(f"End of chunk {i}:   ...{end_of_a}")
    print(f"Start of chunk {i+1}: {start_of_b}...")
    print()