
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

template = """Answer the question using only the ticket context below. Cite ticket IDs.
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def smart_rag(query):
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    if not docs_with_scores:
        return "No relevant tickets found."
    
    best_distance = docs_with_scores[0][1]
    print(f"  [Confidence check] Best distance score: {best_distance:.4f}")
    
    if best_distance < 0.5:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        return chain.invoke(query)
    
    elif best_distance < 1.0:
        ticket_id = docs_with_scores[0][0].metadata["ticket_id"]
        return f"Found possibly relevant ticket ({ticket_id}), but confidence is moderate. Please verify manually."
    
    else:
        return "I dont have relevant ticket history for this question."

test_queries = [
    ("High confidence", "authentication problems"),
    ("Medium confidence", "system performance"),
    ("Low confidence", "how to bake cookies")
]

print("=" * 60)
print("EXERCISE 4: Fallback System")
print("=" * 60)

for label, query in test_queries:
    print(f"\n[{label}] QUERY: {query}")
    print("-" * 40)
    result = smart_rag(query)
    print(result)
