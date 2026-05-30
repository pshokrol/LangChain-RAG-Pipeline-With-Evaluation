
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

template_A = """Answer the question using only the ticket context below. Cite ticket IDs.
Context: {context}
Question: {question}
Answer:"""

template_B = """You are a support assistant. Answer using ONLY the context below.
Context: {context}
Question: {question}
Think step by step:
1. What tickets are relevant?
2. What information do they contain?
3. How does this answer the question?
Answer:"""

template_C = """Answer using only the context. Format as bullet points with ticket citations.
Context: {context}
Question: {question}
Answer (bullet points with sources):"""

query = "How do I fix authentication issues?"

for name, template in [("A - Concise", template_A),
                        ("B - Step by Step", template_B),
                        ("C - Bullet Points", template_C)]:
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    result = chain.invoke(query)
    print(f"\n--- Template {name} ---")
    print(result)
