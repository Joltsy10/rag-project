from src.embeddings import load_embedding_model

print("Loading embedding model (first run downloads ~80MB)...")
embeddings = load_embedding_model()

sentence = "Energy consumption peaks during winter months"
vector = embeddings.embed_query(sentence)

print(f"\nSentence: {sentence}")
print(f"Vector dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")

import numpy as np

sentences = [
    "Energy consumption peaks during winter months",  # original
    "Power usage increases in cold weather",           # semantically similar
    "The stock market crashed in 2008",                # completely unrelated
]

vectors = embeddings.embed_documents(sentences)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v0, v1, v2 = vectors[0], vectors[1], vectors[2]

print(f"\nSimilarity: original vs similar sentence: {cosine_similarity(v0, v1):.4f}")
print(f"Similarity: original vs unrelated sentence: {cosine_similarity(v0, v2):.4f}")