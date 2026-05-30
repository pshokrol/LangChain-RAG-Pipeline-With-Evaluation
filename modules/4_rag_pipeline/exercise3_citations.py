
import os
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

with open("../../data/synthetic_tickets.json") as f:
    tickets = json.load(f)

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)

docs = [Document(
    page_content=f"Ticket {t['ticket_id']}: {t['title']}. {t['description']}. Resolution: {t.get('resolution','N/A')}",
    metadata={"ticket_id": t["ticket_id"]}) for t in tickets]

vector_store = Chroma.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

Example format:
"Database connection timeouts occur when the pool is undersized [TICK-002]. Increase max_connections [TICK-002]."

Context:
{context}

Question: {question}

Answer with inline citations:"""

prompt = ChatPromptTemplate.from_template(citation_prompt)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

queries = [
    "How do I fix authentication issues?",
    "What causes payment failures?",
    "Why is the dashboard slow?"
]

print("=" * 60)
print("EXERCISE 3: Citation Formatting")
print("=" * 60)

for query in queries:
    print(f"\nQUERY: {query}")
    print("-" * 40)
    result = chain.invoke(query)
    print(result)
