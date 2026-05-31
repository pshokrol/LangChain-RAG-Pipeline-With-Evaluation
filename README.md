# LangChain RAG Pipeline With Evaluation

A production-style RAG (Retrieval-Augmented Generation) system built with LangChain and OpenAI, covering pipeline design, evaluation metrics, and agentic reasoning.

## Modules

### Module 4 - RAG Pipeline
- **Part 1:** Compared 3 prompt templates (concise, step-by-step, bullet points)
- **Part 2:** Tested retrieval with different k values (k=1,3,5,10)
- **Part 3:** Implemented inline citation formatting [TICK-XXX]
- **Part 4:** Built a confidence-based fallback system to prevent hallucination

### Module 5 - Evaluation
- **Part 1:** Implemented Precision, Recall, F1 metrics from scratch
- **Part 2:** Evaluated all queries and computed average scores
- **Part 3:** Compared how metrics change across different k values
- **Part 4:** Implemented Average Precision to reward better ranking

### Module 6 - Agentic RAG
- **Part 1:** Built a LangChain agent with 3 custom tools
- **Part 2:** Tested tool selection accuracy across different query types
- **Part 3:** Improved tool descriptions for better selection accuracy
- **Part 4:** Added a new Priority Search Tool (Critical/High/Medium/Low)

## Tech Stack
- LangChain
- OpenAI GPT-4o-mini
- ChromaDB
- FAISS
- Python 3.12

## Key Concepts
- Retrieval-Augmented Generation (RAG)
- Semantic search with embeddings
- Precision, Recall, F1 evaluation
- Agentic tool selection with LLMs
- Anti-hallucination techniques
