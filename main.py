from dotenv import load_dotenv
import os
from urllib.parse import urlparse

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
            model="gemini-1.5-flash-002",
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
# RAG SEARCH
# -------------------------------
def search_humhub(query):
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "metadatas"]
    )

    print("RESULTS:", results)

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    if not docs or not docs[0]:
        return "No relevant context found.", []

    docs = docs[0]
    metas = metas[0] if metas else []

    context_parts = []
    sources = []

    for doc, meta in zip(docs, metas):
        wiki_page = meta.get("wiki_page", "unknown")
        sources.append(wiki_page)

        context_parts.append(f"""
SOURCE: {wiki_page}

CONTENT:
{doc}
""")

    context = "\n\n".join(context_parts)

    return context, list(set(sources))

# -------------------------------
# MAIN ENDPOINT
# -------------------------------
@app.post("/ask")
def ask(data: Question):
    try:
        print("API KEY LOADED:", os.getenv("GEMINI_API_KEY") is not None)
        session_id = data.session_id

        # get history
        history = chat_history.get(session_id, [])

        # RAG
        context, sources = search_humhub(data.question)

        history_text = "\n".join(history)

        # clean source list for prompt
        source_block = "\n".join(sources)

        # prompt
        prompt = f"""
You are an internal company assistant for HumHub.

Use ONLY the context below:

{context}

Conversation history:
{history_text}

Answer clearly and concisely.

At the end, include a Sources section.

IMPORTANT RULES:
- Only use the sources provided
- Do NOT invent sources
- Do NOT guess

Sources:
{source_block}
"""

        # generate
        ai_response = safe_generate_content(prompt, sources)

        # update memory
        history.append(f"User: {data.question}")
        history.append(f"AI: {ai_response['answer']}")
        chat_history[session_id] = history[-10:]

        # format sources
        unique_sources = list(dict.fromkeys(sources))

        clean_sources = [
            {
                "title": clean_source(url),
                "url": url
            }
            for url in unique_sources
        ]

        return {
            "answer": ai_response["answer"],
            "sources": clean_sources
        }

    except Exception as e:
        print("🔥 ERROR IN /ask:", repr(e))
        return {
            "answer": "Internal server error",
            "sources": []
        }
