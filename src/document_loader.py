from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from datetime import datetime
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["file_name"] = os.path.basename(file_path)
        doc.metadata["date_loaded"] = datetime.now().isoformat()

    return documents

def load_txt(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "txt"
        doc.metadata["file_name"] = os.path.basename(file_path)
        doc.metadata["date_loaded"] = datetime.now().isoformat()

    return documents

def load_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["source_type"] = "url"
        doc.metadata["source"] = url
        doc.metadata["date_loaded"] = datetime.now().isoformat()
    
    return documents

def chunk_documents(documents, chunk_size = 500, chunk_overlap = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )

    chunks = splitter.split_documents(documents)
    return chunks

def load_and_chunk(sources: list, chunk_size = 500, chunk_overlap = 50):
    all_documents = []

    for source in sources:
        if source["type"] == "pdf":
            docs = load_pdf(source["path"])
        elif source["type"] == "txt":
            docs = load_txt(source["path"])
        elif source["type"] == "url":
            docs = load_url(source["path"])
        else:
            print(f"Unknown source type: {source['type']}, skipping")
            continue

        all_documents.extend(docs)

    chunks = chunk_documents(all_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Total number of chunks after splitting: {len(chunks)}")

    return chunks

    