from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embedding_model