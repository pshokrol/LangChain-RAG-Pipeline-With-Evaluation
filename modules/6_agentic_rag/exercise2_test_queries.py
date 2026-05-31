
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

with open("../../data/synthetic_tickets.json") as f:
    tickets = json.load(f)

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)

docs = [Document(
    page_content=f"Ticket {t['ticket_id']}: {t['title']}. {t['description']}. Resolution: {t.get('resolution','N/A')}",
    metadata={"ticket_id": t["ticket_id"]}) for t in tickets]

vector_store = Chroma.from_documents(docs, embeddings)

@tool
def SearchSimilarTickets(query: str) -> str:
    """Search for tickets similar to a query. Use for troubleshooting questions like how to fix or why is X happening."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def GetTicketByID(ticket_id: str) -> str:
    """Get a specific ticket by ID like TICK-001. Use when user mentions a ticket ID."""
    ticket_id = ticket_id.strip().upper()
    for t in tickets:
        if t["ticket_id"] == ticket_id:
            return f"Ticket {t['ticket_id']}: {t['title']}\n{t['description']}\nResolution: {t.get('resolution','N/A')}"
    return f"Ticket {ticket_id} not found."

@tool
def GetTicketStatistics(query: str) -> str:
    """Get ticket counts by category. Use ONLY for statistics or count questions."""
    categories = {}
    for t in tickets:
        cat = t.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    stats = f"Total tickets: {len(tickets)}\n"
    for cat, count in categories.items():
        stats += f"  {cat}: {count}\n"
    return stats

tools = [SearchSimilarTickets, GetTicketByID, GetTicketStatistics]
llm_with_tools = llm.bind_tools(tools)
tools_map = {t.name: t for t in tools}

def run_agent(query):
    messages = [
        SystemMessage(content="You are a support assistant. Use tools to answer questions about support tickets."),
        HumanMessage(content=query)
    ]
    try:
        response = llm_with_tools.invoke(messages)
        tool_used = "None"
        answer = response.content

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_used = tool_call["name"]
            tool_input = list(tool_call["args"].values())[0]
            tool_result = str(tools_map[tool_used].invoke(tool_input))[:500]
            messages = [
                SystemMessage(content="You are a support assistant."),
                HumanMessage(content=query),
                response,
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            ]
            final_response = llm_with_tools.invoke(messages)
            answer = final_response.content

        return tool_used, answer
    except Exception as e:
        return "Error", str(e)[:100]

test_queries = [
    ("What issues have we seen with payments?", "SearchSimilarTickets"),
    ("Get ticket TICK-003",                     "GetTicketByID"),
    ("How many tickets are in each category?",  "GetTicketStatistics"),
    ("How to resolve mobile app crashes",       "SearchSimilarTickets"),
]

print("=" * 60)
print("MODULE 6 PART 2: Test Different Queries")
print("=" * 60)

for query, expected_tool in test_queries:
    actual_tool, answer = run_agent(query)
    match = "YES" if actual_tool == expected_tool else "NO"
    print(f"\nQuery: {query}")
    print(f"  Expected Tool: {expected_tool}")
    print(f"  Actual Tool:   {actual_tool}")
    print(f"  Match:         {match}")
    print(f"  Answer: {answer[:150]}")
