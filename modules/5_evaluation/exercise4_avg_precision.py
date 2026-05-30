
import numpy as np

def average_precision(retrieved_ids, relevant_ids):
    """
    AP rewards systems that rank relevant docs HIGHER in results.
    Same retrieved docs but in different order = different score.
    """
    relevant_set = set(relevant_ids)
    precisions = []
    relevant_count = 0

    for k, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precisions.append(precision_at_k)

    return round(np.mean(precisions), 4) if precisions else 0.0

print("=" * 60)
print("EXERCISE 4: Average Precision")
print("=" * 60)

# Test 1: Relevant docs appear EARLY
retrieved1 = ["TICK-001", "TICK-002", "TICK-003"]
relevant1 = ["TICK-001", "TICK-003"]
ap1 = average_precision(retrieved1, relevant1)
print(f"\nTest 1 - Relevant docs early:")
print(f"  Retrieved: {retrieved1}")
print(f"  Relevant:  {relevant1}")
print(f"  AP: {ap1} (expected ~0.8333)")

# Test 2: Relevant docs appear LATE
retrieved2 = ["TICK-002", "TICK-004", "TICK-001"]
relevant2 = ["TICK-001", "TICK-003"]
ap2 = average_precision(retrieved2, relevant2)
print(f"\nTest 2 - Relevant docs late:")
print(f"  Retrieved: {retrieved2}")
print(f"  Relevant:  {relevant2}")
print(f"  AP: {ap2} (expected ~0.3333)")

print(f"\nKEY INSIGHT:")
print(f"  Same relevant doc (TICK-001) retrieved in both cases")
print(f"  But AP is higher when it appears earlier ({ap1} vs {ap2})")
print(f"  AP rewards systems that rank relevant docs first!")
