
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
print("EXERCISE 2: Evaluate All Queries")
print("=" * 60)

all_metrics = []
for query in eval_queries:
    docs_retrieved = vector_store.similarity_search(query["question"], k=3)
    retrieved = [doc.metadata["ticket_id"] for doc in docs_retrieved]
    metrics = calculate_metrics(retrieved, query["relevant_ticket_ids"])
    all_metrics.append(metrics)
    print(f"\nQuery: {query['question']}")
    print(f"  Retrieved:  {retrieved}")
    print(f"  Expected:   {query['relevant_ticket_ids']}")
    print(f"  Precision: {metrics['precision']}  Recall: {metrics['recall']}  F1: {metrics['f1']}")

avg_precision = np.mean([m["precision"] for m in all_metrics])
avg_recall = np.mean([m["recall"] for m in all_metrics])
avg_f1 = np.mean([m["f1"] for m in all_metrics])

print("\n" + "=" * 60)
print("AVERAGE SCORES ACROSS ALL QUERIES:")
print(f"  Precision@3: {avg_precision:.4f}")
print(f"  Recall@3:    {avg_recall:.4f}")
print(f"  F1@3:        {avg_f1:.4f}")
print("=" * 60)
