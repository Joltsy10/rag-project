import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.vector_store import get_retriever,get_hybrid_retriever
from langchain.schema.runnable import RunnableLambda

load_dotenv()

def load_llm():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name = "llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024
    )
    return llm

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        formatted.append(f"Source: {source}\nContent: {doc.page_content}")

    return "\n\n---\n\n".join(formatted)

def create_rag_chain(embedding_model = None, k = 4):
    retriever = get_retriever(k=k, embedding_model=embedding_model)
    llm = load_llm()

    prompt_template = """You are a helpful assistant that answers questions based strictly on the provided context.

    If the answer is not found in the context, say "I don't have enough information in the provided documents to answer this question." Do not make up answers.

    Always mention which source document your answer comes from.

    Context:{context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask(question, embedding_model = None, k = 4):
    chain = create_rag_chain(embedding_model=embedding_model, k=k)
    response = chain.invoke(question)
    return response

def rewrite_query(question, llm = None):
    if llm is None:
        llm = load_llm()

    prompt = f"""You are an expert at reformulating questions into better search queries.
    Rewrite the following question into a search query that would retrieve the most relevant information from a document database.
    Make it more specific and include likely related terms. Return ONLY the rewritten query, nothing else.

    Question: {question}
    Rewritten query:"""

    response = llm.invoke(prompt)
    return response.content.strip()

def create_rag_chain_with_rewrite(embedding_model = None, k = 4):
    llm = load_llm()

    prompt_template = """You are a helpful assistant that answers questions based strictly on the provided context.

    If the answer is not found in the context, say "I don't have enough information in the provided documents to answer this question." Do not make up answers.

    Always mention which source document your answer comes from.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    def retrieve_with_rewrite(question):
        rewritten = rewrite_query(question, llm)
        retriever = get_retriever(k=k, embedding_model=embedding_model)
        docs = retriever.invoke(rewritten)
        return format_docs(docs)
    
    chain = (
        {"context": RunnableLambda(retrieve_with_rewrite), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask_with_rewrite(question, embedding_model = None, k = 4):
    chain = create_rag_chain_with_rewrite(embedding_model=embedding_model, k = k)
    response = chain.invoke(question)
    return response

def create_rag_chain_V2(embedding_model = None, k = 4):
    retriever = get_retriever(k=k, embedding_model=embedding_model)
    llm = load_llm()

    prompt_template = """You are an expert assistant that answers questions using ONLY the provided context.

    Rules:
    1. Base your answer strictly on the context provided. Do not use outside knowledge.
    2. If the context contains partial information, use what's available and note what's missing.
    3. Always cite the source document for every claim you make.
    4. If the context contains no relevant information, say "The provided documents do not contain information about this topic."
    5. Be concise and specific — avoid vague answers.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask_V2(question, embedding_model = None, k = 4):
    chain = create_rag_chain_V2(embedding_model=embedding_model, k=k)
    response = chain.invoke(question)
    return response

def create_rag_chain_V3(embedding_model = None, k = 4):
    retriever = get_retriever(k=k, embedding_model=embedding_model)
    llm = load_llm()

    prompt_template = """You are an expert assistant. Follow these steps to answer the question:

    Step 1: Read the provided context carefully.
    Step 2: Identify which parts of the context are relevant to the question.
    Step 3: Formulate a precise answer using ONLY the relevant context.
    Step 4: Cite which source document your answer comes from.

    If no relevant information exists in the context, explicitly state: "The provided documents do not contain information about this topic."

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask_V3(question, embedding_model = None, k = 4):
    chain = create_rag_chain_V2(embedding_model=embedding_model, k=k)
    response = chain.invoke(question)
    return response

def create_rag_chain_hybrid(chunks, embedding_model = None, k = 4):
    
    retriever = get_hybrid_retriever(chunks, k=k, embedding_model=embedding_model)
    llm = load_llm()

    prompt_template = """You are an expert assistant that answers questions using ONLY the provided context.

    Rules:
    1. Base your answer strictly on the context provided. Do not use outside knowledge.
    2. If the context contains partial information, use what's available and note what's missing.
    3. Always cite the source document for every claim you make.
    4. If the context contains no relevant information, say "The provided documents do not contain information about this topic."
    5. Be concise and specific — avoid vague answers.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask_hybrid(question, chunks, embedding_model = None, k = 4):
    chain = create_rag_chain_hybrid(chunks, embedding_model, k=k)
    response = chain.invoke(question)
    return response

