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
    q_words = set(query.lower().split())

    def score(doc):
        doc_lower = doc.lower()
        return sum(1 for w in q_words if w in doc_lower)

    return max(docs, key=score)

# -------------------------------
# RETRIEVAL SCORE (FIXED)
# -------------------------------
def retrieval_score(docs, query):
    if not docs:
        return 0

    q_words = set(query.lower().split())
    best_doc = max(docs, key=len).lower()

    return sum(1 for w in q_words if w in best_doc)

#-----------------------------
# RETRIEVAL CONFIDENCE
# ----------------------------

def retrieval_confidence(docs, query):
    """
    Simple confidence score:
    how many query words appear in retrieved chunks
    """
    q_words = set(query.lower().split())

    combined = " ".join(docs).lower()

    if not combined.strip():
        return 0

    return sum(1 for w in q_words if w in combined)

# -------------------------------
# DIRECT ANSWER (NO GEMINI)
# -------------------------------
def build_direct_answer(question, top_doc):
    sentences = top_doc.split(".")

    q_words = set(question.lower().split())

    for s in sentences:
        if any(w in s.lower() for w in q_words):
            return s.strip()

    return sentences[0].strip() if sentences else top_doc

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
        return "", [], "", 0

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

    score = retrieval_confidence(docs, query)

    return context, sources, top_doc, score

# -------------------------------
# MAIN ENDPOINT
# -------------------------------
@app.post("/ask")
def ask(data: Question):
    try:
        session_id = data.session_id
        question = data.question
        history = chat_history.get(session_id, [])

        context, sources, top_doc, score = search_humhub(question)

        # -------------------------------
        # 🚀 LEVEL 1: NO GEMINI (HIGH CONFIDENCE)
        # -------------------------------
        if score >= 3:
            return {
                "answer": top_doc[:1200],
                "sources": sources
            }

        # -------------------------------
        # 🚀 LEVEL 2: STILL NO GEMINI (DIRECT EXTRACTION)
        # -------------------------------
        if score >= 1 and len(top_doc) > 50:
            return {
                "answer": build_direct_answer(question, top_doc),
                "sources": sources
            }

        # -------------------------------
        # 🚀 LEVEL 3: ONLY NOW CALL GEMINI
        # -------------------------------
        history_text = "\n".join(history)

        prompt = f"""
You are a strict internal assistant.

ONLY use context.

CONTEXT:
{context}

QUESTION:
{question}

If answer not in context say:
"I could not find relevant information in the company policies."
"""

        ai_response = safe_generate_content(prompt, sources)

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
