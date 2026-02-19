from langchain_community.vectorstores import Chroma
from src.embeddings import load_embedding_model
import os

CHROMA_PATH = ".chroma"

def get_vector_store(embedding_model = None):
    if embedding_model is None:
        embedding_model = load_embedding_model()

    vector_store = Chroma(
        collection_name= "rag_collection",
        embedding_function= embedding_model,
        persist_directory= CHROMA_PATH
    )
    return vector_store

def add_documents(chunks, embedding_model = None):
    if embedding_model is None:
        embedding_model = load_embedding_model()

    vector_store = get_vector_store(embedding_model)

    existing = vector_store.get()
    existing_sources = set()
    if existing and existing["metadatas"]:
        for meta in existing["metadatas"]:
            if "file_name" in meta:
                existing_sources.add(meta["file_name"])
            elif "source" in meta:
                existing_sources.add(meta["source"])

    new_chunks = []
    for chunk in chunks:
        source = chunk.metadata.get("file_name") or chunk.metadata.get("source")
        if source not in existing_sources:
            new_chunks.append(chunk)

    if not new_chunks:
        print("No new documents to add")
        return vector_store
    
    print(f"Adding {len(chunks)} new chunks to the vector store")
    vector_store.add_documents(new_chunks)

    print("Done")
    return vector_store

def similarity_search(query, k = 4, embedding_model = None):
    if embedding_model is None:
        embedding_model = load_embedding_model()

    vector_store = get_vector_store(embedding_model)
    results = vector_store.similarity_search(query, k=k)

    return results

def get_retriever(k = 4, embedding_model = None):
    if embedding_model is None:
        embedding_model = load_embedding_model

    vector_store = get_vector_store(embedding_model)
    return vector_store.as_retriever(search_kwargs = {"k":k})