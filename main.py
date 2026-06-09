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
    allow_origins=[
        "https://hub.mfgsolicitors.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# MEMORY STORE (simple in-memory)
# -------------------------------
chat_history = {}

# -------------------------------
# REQUEST MODEL
# -------------------------------
class Question(BaseModel):
    session_id: str
    question: str

# -------------------------------
# SAFE GEMINI CALL
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
        print("🔥 AI ERROR:", repr(e))

        return {
            "answer": "AI temporarily unavailable. Try again shortly.",
            "sources": sources or []
        }

# -------------------------------
# CLEAN URL TITLE
# -------------------------------
def clean_source(url: str):
    try:
        path = urlparse(url).path
        slug = path.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()
    except:
        return url

# -------------------------------
# HELPERS (put these near top)
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
        return "No relevant context found.", [], ""

    # FIXED: use query instead of data.question
    top_doc = pick_best_doc(docs, query)

    context_parts = []
    sources = []

    MAX_DOCS = 3
    MAX_CHARS_PER_DOC = 1200

    for doc, meta in zip(docs[:MAX_DOCS], metas[:MAX_DOCS]):
        wiki_page = meta.get("wiki_page", "unknown")

        sources.append({
            "title": clean_source(wiki_page),
            "url": wiki_page
        })

        doc = doc[:MAX_CHARS_PER_DOC]

        context_parts.append(
            f"SOURCE: {wiki_page}\nCONTENT:\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)
    context = context[:4000]

    return context, sources, top_doc


# -------------------------------
# SIMPLE QUESTION DETECTOR
# -------------------------------
def is_simple_question(question):
    q = question.lower()

    # ONLY factual lookup triggers (safe)
    keywords = [
        "what is",
        "who is",
        "where is",
        "when is",
        "how long",
        "how many"
    ]

    # hard block overly broad keywords
    skip_words = [
        "summarise",
        "summary",
        "explain",
        "compare",
        "difference",
        "policy"
    ]

    if any(w in q for w in skip_words):
        return False

    return any(k in q for k in keywords)

# -------------------------------
# MAIN ENDPOINT
# -------------------------------
import traceback

@app.post("/ask")
def ask(data: Question):
    try:
        print("API KEY LOADED:", os.getenv("GEMINI_API_KEY") is not None)

        session_id = data.session_id
        history = chat_history.get(session_id, [])

        # -------------------------------
        # RAG
        # -------------------------------
        context, sources = search_humhub(data.question)

        # -------------------------------
        # SIMPLE MODE (NO GEMINI)
        # -------------------------------
        if is_simple_question(data.question):
            return {
                "answer": context[:1500],
                "sources": sources
            }

        history_text = "\n".join(history)

        # -------------------------------
        # PROMPT
        # -------------------------------
        prompt = f"""
You are a strict internal company assistant for HumHub. 

You MUST follow these rules:

1. ONLY use the provided context below. 
2. If the context does not contain the answer, say: "I could not find relevant information in the company policies." 
3. NEVER guess or use outside knowledge. 
4. NEVER change the topic. 
5. NEVER mention unrelated topics.

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

USER QUESTION:
{data.question}

INSTRUCTIONS: 
- Answer ONLY using CONTEXT above 
- Be concise and professional 
- If context is unrelated, say you cannot find relevant information 
- Do NOT mention missing topics like "mileage" unless asked 
- Do NOT include a Sources section 
- Do NOT list URLs
"""

        # -------------------------------
        # GEMINI CALL
        # -------------------------------
        ai_response = safe_generate_content(prompt, sources)

        # -------------------------------
        # MEMORY
        # -------------------------------
        history.append(f"User: {data.question}")
        history.append(f"AI: {ai_response['answer']}")
        chat_history[session_id] = history[-10:]

        # -------------------------------
        # CLEAN SOURCES
        # -------------------------------
        clean_sources = [
            {
                "title": s["title"],
                "url": s["url"]
            }
            for s in sources
        ]

        return {
            "answer": ai_response["answer"],
            "sources": clean_sources
        }

    except Exception as e:
        print("🔥 ERROR IN /ask:", repr(e))
        traceback.print_exc()

        return {
            "answer": "Internal server error",
            "sources": []
        }
