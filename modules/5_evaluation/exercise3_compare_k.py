
import os
import json
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

with open("../../data/synthetic_tickets.json") as f:
    tickets = json.load(f)

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

docs = [Document(
    page_content=f"Ticket {t['ticket_id']}: {t['title']}. {t['description']}. Resolution: {t.get('resolution','N/A')}",
    metadata={"ticket_id": t["ticket_id"]}) for t in tickets]

vector_store = Chroma.from_documents(docs, embeddings)

def calculate_metrics(retrieved_ids, relevant_ids, k=3):
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    tp = len(retrieved_set & relevant_set)
    precision = tp / k if k > 0 else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

eval_queries = [
    {"question": "authentication login issues", "relevant_ticket_ids": ["TICK-001", "TICK-011", "TICK-016"]},
    {"question": "payment processing failures", "relevant_ticket_ids": ["TICK-003", "TICK-018"]},
    {"question": "mobile app crashes", "relevant_ticket_ids": ["TICK-004", "TICK-014"]},
    {"question": "slow dashboard loading", "relevant_ticket_ids": ["TICK-010"]},
    {"question": "database connection errors", "relevant_ticket_ids": ["TICK-002", "TICK-007"]}
]

print("=" * 60)
print("EXERCISE 3: Compare Different k Values")
print("=" * 60)
print(f"\n{'k':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 42)

for k in [1, 3, 5, 10]:
    metrics = []
    for query in eval_queries:
        docs_retrieved = vector_store.similarity_search(query["question"], k=k)
        retrieved = [doc.metadata["ticket_id"] for doc in docs_retrieved]
        m = calculate_metrics(retrieved, query["relevant_ticket_ids"], k=k)
        metrics.append(m)
    
    avg_p = np.mean([m["precision"] for m in metrics])
    avg_r = np.mean([m["recall"] for m in metrics])
    avg_f = np.mean([m["f1"] for m in metrics])
    print(f"{k:<6} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f:<12.4f}")

print("\nOBSERVATIONS:")
print("- As k increases, Recall goes UP (find more relevant docs)")
print("- As k increases, Precision goes DOWN (more noise included)")
print("- k=3 usually gives the best F1 balance")
