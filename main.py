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
# APP + CLIENT
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
# MEMORY
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
# HELPERS
# -------------------------------
def clean_source(url: str):
    try:
        path = urlparse(url).path
        slug = path.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()
    except:
        return url


def pick_best_doc(docs, metas, query):

    q_words = set(query.lower().split())

    policy_scores = {}

    for doc, meta in zip(docs, metas):

        title = meta.get("policy_title", "")

        score = 0

        for word in q_words:
            if word in title.lower():
                score += 50

        for word in q_words:
            if word in doc.lower():
                score += 1

        policy_scores.setdefault(title, 0)
        policy_scores[title] += score

    best_policy = max(policy_scores, key=policy_scores.get)

    best_doc = None
    best_doc_score = -1

    for doc, meta in zip(docs, metas):

        if meta.get("policy_title") != best_policy:
            continue

        score = sum(
            1 for w in q_words
            if w in doc.lower()
        )

        if score > best_doc_score:
            best_doc_score = score
            best_doc = doc

    return best_doc

    def score(doc):
        doc_lower = doc.lower()
        return sum(1 for w in q_words if w in doc_lower)

    return max(docs, key=score)


def retrieval_confidence(docs, query):
    q_words = set(query.lower().split())
    combined = " ".join(docs).lower()

    if not combined.strip():
        return 0

    return sum(1 for w in q_words if w in combined)

# -------------------------------
# DIRECT ANSWER (NO GEMINI)
# -------------------------------
def build_direct_answer(question, doc):
    sentences = doc.split(".")
    q_words = set(question.lower().split())

    for s in sentences:
        if any(w in s.lower() for w in q_words) and len(s.strip()) > 20:
            return s.strip() + "."

    return sentences[0].strip() if sentences else doc

# -------------------------------
# STRONG ANSWER BUILDER
# -------------------------------
def build_answer_from_chunks(question, doc):
    sentences = doc.split(".")
    q_words = set(question.lower().split())

    best = max(
        sentences,
        key=lambda s: sum(w in s.lower() for w in q_words),
        default=doc
    )

    if len(best.strip()) > 20:
        return best.strip() + "."

    return doc[:800]

# -------------------------------
# RAG SEARCH
# -------------------------------
def search_humhub(query):
    results = collection.query(
        query_texts=[query],
        n_results=30,
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
        question = data.question
        session_id = data.session_id

        history = chat_history.get(session_id, [])

        # -------------------------------
        # RAG
        # -------------------------------
        context, sources, top_doc, score = search_humhub(question)

        # -------------------------------
        # LEVEL 1: STRONG MATCH (NO GEMINI)
        # -------------------------------
        if score >= 5:
            return {
                "answer": build_answer_from_chunks(question, top_doc),
                "sources": sources
            }

        # -------------------------------
        # LEVEL 2: MEDIUM MATCH (NO GEMINI)
        # -------------------------------
        if score >= 2 and len(top_doc) > 50:
            return {
                "answer": build_direct_answer(question, top_doc),
                "sources": sources
            }

        # -------------------------------
        # LEVEL 3: GEMINI
        # -------------------------------
        prompt = f"""
You are a strict internal assistant.

ONLY use the context below.

If answer is not in context say:
"I could not find relevant information in the company policies."

CONTEXT:
{context}

QUESTION:
{question}
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
