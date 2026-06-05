from dotenv import load_dotenv
import os
from urllib.parse import urlparse
import time

load_dotenv()

chat_history = {}

from rag_db import collection

def safe_generate_content(prompt, retries=3):
    last_error = None

    for i in range(retries):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        except Exception as e:
            last_error = e
            print(f"⚠️ Gemini retry {i+1} failed:", repr(e))
            time.sleep(1.5)

    return None

def clean_source(url: str):
    try:
        path = urlparse(url).path
        slug = path.rstrip("/").split("/")[-1]
        title = slug.replace("-", " ").title()
        return title
    except:
        return url

def search_humhub(query):
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No relevant context found.", []

    context_parts = []
    sources = []

    for doc, meta in zip(docs, metas):
        wiki_page = meta.get("wiki_page", "unknown")
        pdf_url = meta.get("pdf_url", "unknown")

        sources.append(wiki_page)

        context_parts.append(
            f"""
SOURCE: {wiki_page}
PDF: {pdf_url}

CONTENT:
{doc}
"""
        )

    context = "\n\n".join(context_parts)

    return context, sources

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
chat_histories = {}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()

class Question(BaseModel):
    session_id: str
    question: str

@app.post("/ask")
def ask(data: Question):
    try:
        session_id = data.session_id

        # get previous chat history
        history = chat_history.get(session_id, [])

        context, sources = search_humhub(data.question)

        history_text = "\n".join(history)

        prompt = f"""
You are an internal company assistant for HumHub.

Use ONLY the context below:

{context}

Conversation history:
{history_text}

Answer clearly and concisely.

At the end of your answer, output a "Sources" section.

CRITICAL RULES:
- Use ONLY the provided sources list below
- Do NOT add any new sources
- Do NOT guess sources

Sources:
{sources}
"""

        response = safe_generate_content(prompt)

        if not response:
            return {
                "answer": "Sorry, the AI service is temporarily busy. Please try again in a moment.",
                "sources": sources
            }

        # store memory
        history.append(f"User: {data.question}")
        history.append(f"AI: {response.text}")
        chat_history[session_id] = history[-10:]  # keep last 10 lines

        unique_sources = list(dict.fromkeys(sources))  # preserves order, removes duplicates
        clean_sources = [
            {
                "title": clean_source(url),
                "url": url
            }
            for url in unique_sources
        ]

        return {
            "answer": response.text,
            "sources": clean_sources
        }

    except Exception as e:
        print("🔥 ERROR IN /ask:", repr(e))
        return {"error": str(e)}