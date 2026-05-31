
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import Tool

with open("../../data/synthetic_tickets.json") as f:
    tickets = json.load(f)

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)

docs = [Document(
    page_content=f"Ticket {t['ticket_id']}: {t['title']}. {t['description']}. Resolution: {t.get('resolution','N/A')}",
    metadata={"ticket_id": t["ticket_id"]}) for t in tickets]

vector_store = Chroma.from_documents(docs, embeddings)

def search_similar_tickets(query):
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

def get_ticket_by_id(ticket_id):
    ticket_id = ticket_id.strip().upper()
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            return f"Ticket {t['ticket_id']}: {t['title']}\n{t['description']}\nResolution: {t.get('resolution','N/A')}"
    return f"Ticket {ticket_id} not found."

def get_statistics(query=""):
    categories = {}
    for t in tickets:
        cat = t.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    stats = f"Total tickets: {len(tickets)}\n"
    for cat, count in categories.items():
        stats += f"  {cat}: {count}\n"
    return stats

tools = [
    Tool(name="SearchSimilarTickets",
         func=search_similar_tickets,
         description="Search for tickets similar to a query. Use for troubleshooting questions."),
    Tool(name="GetTicketByID",
         func=get_ticket_by_id,
         description="Get a specific ticket by ID like TICK-001. Use when user mentions a ticket ID."),
    Tool(name="GetTicketStatistics",
         func=get_statistics,
         description="Get ticket counts by category. Use for statistics or count questions.")
]

llm_with_tools = llm.bind_tools(tools)
tools_map = {t.name: t for t in tools}

def run_agent(query):
    print(f"\nQUERY: {query}")
    print("-" * 40)
    messages = [
        SystemMessage(content="You are a support assistant. Use tools to answer questions about support tickets."),
        HumanMessage(content=query)
    ]
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            print(f"  Tool selected: {tool_name}")
            print(f"  Input: {tool_input}")
            
            tool_result = tools_map[tool_name].func(list(tool_input.values())[0])
            messages.append(response)
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
            
            final_response = llm_with_tools.invoke(messages)
            print(f"  Answer: {final_response.content}")
    else:
        print(f"  Answer: {response.content}")

print("=" * 60)
print("MODULE 6 PART 1: Observe Agent Behavior")
print("=" * 60)

queries = [
    "How do I fix authentication problems?",
    "Show me ticket TICK-005",
    "How many tickets are in each category?"
]

for query in queries:
    run_agent(query)
