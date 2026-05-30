
import numpy as np

def calculate_metrics(retrieved_ids, relevant_ids, k=3):
    """
    Precision@k = how many retrieved docs were actually relevant
    Recall@k = how many relevant docs were actually retrieved
    F1@k = balance between precision and recall
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    tp = len(retrieved_set & relevant_set)
    
    precision = tp / k if k > 0 else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

print("=" * 60)
print("EXERCISE 1: Precision, Recall, F1")
print("=" * 60)

# Test 1
retrieved = ["TICK-001", "TICK-002", "TICK-003"]
relevant = ["TICK-001", "TICK-003"]
metrics = calculate_metrics(retrieved, relevant)
print(f"\nTest 1:")
print(f"  Retrieved: {retrieved}")
print(f"  Relevant:  {relevant}")
print(f"  Precision: {metrics['precision']} (expected 0.6667)")
print(f"  Recall:    {metrics['recall']} (expected 1.0)")
print(f"  F1:        {metrics['f1']} (expected 0.8)")

# Test 2
retrieved2 = ["TICK-002", "TICK-004", "TICK-005"]
relevant2 = ["TICK-001", "TICK-003"]
metrics2 = calculate_metrics(retrieved2, relevant2)
print(f"\nTest 2 (no overlap):")
print(f"  Retrieved: {retrieved2}")
print(f"  Relevant:  {relevant2}")
print(f"  Precision: {metrics2['precision']} (expected 0.0)")
print(f"  Recall:    {metrics2['recall']} (expected 0.0)")
print(f"  F1:        {metrics2['f1']} (expected 0.0)")

# Test 3
retrieved3 = ["TICK-001", "TICK-002", "TICK-003"]
relevant3 = ["TICK-001", "TICK-002", "TICK-003"]
metrics3 = calculate_metrics(retrieved3, relevant3)
print(f"\nTest 3 (perfect retrieval):")
print(f"  Retrieved: {retrieved3}")
print(f"  Relevant:  {relevant3}")
print(f"  Precision: {metrics3['precision']} (expected 1.0)")
print(f"  Recall:    {metrics3['recall']} (expected 1.0)")
print(f"  F1:        {metrics3['f1']} (expected 1.0)")
