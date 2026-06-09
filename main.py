from dotenv import load_dotenv
import os
from urllib.parse import urlparse
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

from rag_db import collection

load_dotenv()

# -------------------------------
# CLIENT + APP
# -------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hub.mfgsolicitors.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# MEMORY STORE
# -------------------------------
chat_history = {}

# -------------------------------
# REQUEST MODEL
# -------------------------------
class Question(BaseModel):
    session_id: str
    question: str


# -------------------------------
# GEMINI CALL
# -------------------------------
def safe_generate_content(prompt, sources=None):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": sources or []
        }

    except Exception as e:
        print("🔥 GEMINI ERROR:", repr(e))
        return {
            "answer": "AI temporarily unavailable. Try again shortly.",
            "sources": sources or []
        }


# -------------------------------
# CLEAN SOURCE TITLE
# -------------------------------
def clean_source(url: str):
    try:
        path = urlparse(url).path
        slug = path.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()
    except:
        return url


# -------------------------------
# PICK BEST DOC
# -------------------------------
def pick_best_doc(docs, query):
    query_words = set(query.lower().split())

    def score(doc):
        doc_lower = doc.lower()
        return sum(1 for w in query_words if w in doc_lower)

    return max(docs, key=score)


# -------------------------------
# RAG SEARCH
# -------------------------------
def search_humhub(query):
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "", [], ""

    top_doc = pick_best_doc(docs, query)

    context_parts = []
    sources = []

    for doc, meta in zip(docs[:3], metas[:3]):
        wiki_page = meta.get("wiki_page", "unknown")

        sources.append({
            "title": clean_source(wiki_page),
            "url": wiki_page
        })

        context_parts.append(
            f"SOURCE: {wiki_page}\nCONTENT:\n{doc[:1200]}"
        )

    context = "\n\n---\n\n".join(context_parts)[:4000]

    return context, sources, top_doc


# -------------------------------
# SMART ROUTER (IMPORTANT FIX)
# -------------------------------
def needs_gemini(question: str) -> bool:
    q = question.lower()

    # If user is asking for explanation/summary → Gemini
    complex_keywords = [
        "summarise", "summary", "explain", "compare",
        "difference", "why", "how does", "meaning"
    ]

    # policy lookup style → NO GEMINI
    simple_keywords = [
        "what is", "who is", "where is", "when is",
        "how long", "how many", "maternity", "harassment",
        "leave", "policy", "holiday"
    ]

    if any(k in q for k in complex_keywords):
        return True

    if any(k in q for k in simple_keywords):
        return False

    # default safe mode = use Gemini
    return True


# -------------------------------
# MAIN ENDPOINT
# -------------------------------
@app.post("/ask")
def ask(data: Question):
    try:
        session_id = data.session_id
        question = data.question

        history = chat_history.get(session_id, [])

        # -------------------------------
        # RAG
        # -------------------------------
        context, sources, top_doc = search_humhub(question)

        # -------------------------------
        # 🚀 ROUTING DECISION
        # -------------------------------
        if not needs_gemini(question):
            # PURE CHROMA MODE (NO GEMINI)
            return {
                "answer": top_doc[:1200] if top_doc else context[:1200],
                "sources": sources
            }

        # -------------------------------
        # GEMINI MODE
        # -------------------------------
        history_text = "\n".join(history)

        prompt = f"""
You are a strict internal company assistant.

RULES:
- Only use context
- If not in context say: "I could not find relevant information in the company policies."

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

QUESTION:
{question}

Answer clearly and concisely.
"""

        ai_response = safe_generate_content(prompt, sources)

        # memory update
        history.append(f"User: {question}")
        history.append(f"AI: {ai_response['answer']}")
        chat_history[session_id] = history[-10:]

        return {
            "answer": ai_response["answer"],
            "sources": sources
        }

    except Exception as e:
        print("🔥 ERROR:", repr(e))
        traceback.print_exc()
        return {
            "answer": "Internal server error",
            "sources": []
        }
