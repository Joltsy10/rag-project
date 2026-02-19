import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.vector_store import get_retriever
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
