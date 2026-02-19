from src.document_loader import load_url
from src.embeddings import load_embedding_model
from src.vector_store import add_documents
from src.rag_chain import ask

embedding_model = load_embedding_model()
url_docs = load_url("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)")

from src.document_loader import chunk_documents
chunks = chunk_documents(url_docs)
add_documents(chunks, embedding_model)

response = ask("what is a transformer in deep learning?", embedding_model=embedding_model)
print(response)