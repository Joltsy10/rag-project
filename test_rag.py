from src.rag_chain import ask
from src.embeddings import load_embedding_model

embedding_model = load_embedding_model()

questions = [
    "what is machine learning",
    "What are the main components required for a system to learn from experience?",
]

for q in questions:
    print(f"\nQuestion: {q}")
    print(f"Answer: {ask(q, embedding_model=embedding_model)}")
    print("=" * 50)