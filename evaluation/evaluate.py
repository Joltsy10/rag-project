import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vector_store import similarity_search
from src.rag_chain import ask
from src.embeddings import load_embedding_model
from src.rag_chain import ask_with_rewrite
from src.document_loader import load_and_chunk
from src.vector_store import add_documents, clear_vector_store


load_dotenv()

def llm_judge(question, answer, context, ground_truth, llm):
    prompt = f"""You are evaluating a RAG system. Score the following on a scale of 0 to 1.

    Question: {question}
    Retrieved Context: {context}
    Generated Answer: {answer}
    Ground Truth: {ground_truth}

    Score these three things, responding ONLY with valid JSON, nothing else:
    {{
        "faithfulness": <0-1, is the answer supported by the context?>,
        "answer_relevancy": <0-1, does the answer address the question?>,
        "context_recall": <0-1, does the context contain enough to answer the question?>,
        "completeness": <0-1, did the answer actually answer the question with substance? Saying 'I dont know' or 'I dont have enough information' should score 0>
    }}"""

    response = llm.invoke(prompt)
    try:
        scores = json.loads(response.content)
    except:
        scores = {"faithfulness": 0, "answer_relevancy": 0, "context_recall": 0, "completeness": 0}
    return scores

def run_evaluation(test_set_path = "evaluation/test_set.json", chunk_size = 500, chunk_overlap = 50,  k = 4):
    with open(test_set_path) as f:
        test_set = json.load(f)

    embedding_model = load_embedding_model()

    clear_vector_store()
    sources = [
        {"type": "pdf", "path": "data/learning.pdf"},
        {"type": "txt", "path": "data/transformers.txt"},
    ]
    chunks = load_and_chunk(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    add_documents(chunks, embedding_model)

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name = "llama-3.3-70b-versatile",
        temperature=0
    )

    all_scores = []

    print(f"Running evaluation with k={k} and chunk size = {chunk_size}\n")
    for item in test_set:
        question = item["question"]
        ground_truth = item["ground_truth"]

        answer = ask(question, embedding_model= embedding_model, k=k)
        retrieved_docs = similarity_search(question,k=k, embedding_model=embedding_model)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        scores = llm_judge(question, answer, context, ground_truth, llm)
        all_scores.append(scores)

        print(f"Q: {question[:70]}")
        print(f"A: {answer[:100]}")
        print(f"Scores: {scores}\n")

    answered = [s for s in all_scores if s["completeness"] > 0.3]
    unanswered = [s for s in all_scores if s["completeness"] <= 0.3]
    
    print(f"\nQuestions answered: {len(answered)}/{len(all_scores)}")
    print(f"Questions unanswered: {len(unanswered)}/{len(all_scores)}")

    avg_faithfulness = sum(s["faithfulness"] for s in answered) / len(answered) if answered else 0
    avg_relevancy = sum(s["answer_relevancy"] for s in answered) / len(answered) if answered else 0
    avg_recall = sum(s["context_recall"] for s in all_scores) / len(all_scores)
    avg_completeness = sum(s["completeness"] for s in all_scores) / len(all_scores)

    print("===== AVERAGE SCORES =====")
    print(f"Faithfulness:     {avg_faithfulness:.4f}")
    print(f"Answer Relevancy: {avg_relevancy:.4f}")
    print(f"Context Recall:   {avg_recall:.4f}")
    print(f"Completeness:     {avg_completeness:.4f}")

    results = {
        "config": {"k": k, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "scores": {
            "faithfulness": avg_faithfulness,
            "answer_relevancy": avg_relevancy,
            "context_recall": avg_recall,
            "completeness": avg_completeness
        }
    }

    output_path = f"evaluation/results_k{k}_chunk{chunk_size}_70b.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_evaluation(k=4, chunk_overlap=30, chunk_size=300)