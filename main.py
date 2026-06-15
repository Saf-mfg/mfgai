from dotenv import load_dotenv
from urllib.parse import urlparse
from collections import Counter
import traceback
import re
import os
import zipfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from rag_db import collection

# -------------------------------
# INIT
# -------------------------------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()

chat_history = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hub.mfgsolicitors.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# REQUEST
# -------------------------------
class Question(BaseModel):
    session_id: str
    question: str


# -------------------------------
# UTIL
# -------------------------------
def clean_source(url: str):
    try:
        path = urlparse(url).path
        slug = path.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()
    except:
        return url


STOP_WORDS = {
    "what","is","are","the","a","an","how","do","does","can","i",
    "we","you","of","for","to","please","tell","about"
}

def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return {
        w for w in words
        if len(w) > 2 and w not in STOP_WORDS
    }


def is_definition_question(q: str):
    q = q.lower()
    return any(x in q for x in ["what is", "define", "meaning of"])


def clean_sentences(text):
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def create_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )

    return response.embeddings[0].values


# -------------------------------
# GEMINI
# -------------------------------
def safe_generate_content(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print("GEMINI ERROR:", repr(e))
        return "AI temporarily unavailable."


# -------------------------------
# RAG CORE (RETRIEVAL + RERANK)
# -------------------------------
def retrieve_context(query: str):
    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=30,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return [], []

    # -------------------------------
    # SEMANTIC RERANK (IMPORTANT PART)
    # -------------------------------
    query_vec = model.encode([query])
    doc_vecs = model.encode(docs)

    scores = cosine_similarity(query_vec, doc_vecs)[0]
    top_indices = scores.argsort()[-6:][::-1]

    top_docs = [docs[i] for i in top_indices]
    top_metas = [metas[i] for i in top_indices]

    return top_docs, top_metas


# -------------------------------
# DIRECT ANSWER BUILDER
# -------------------------------
def build_direct_answer(question, context):
    sentences = clean_sentences(context)
    keywords = extract_keywords(question)

    scored = []

    for s in sentences:
        score = 0
        lower = s.lower()

        # strong definition signals
        if any(x in lower for x in [
            "is defined as",
            "means",
            "refers to",
            "definition"
        ]):
            score += 50

        for k in keywords:
            if k in lower:
                score += 5

        scored.append((score, s))

    scored.sort(reverse=True, key=lambda x: x[0])

    return " ".join([s for _, s in scored[:3]])


# -------------------------------
# MAIN ENDPOINT
# -------------------------------
@app.post("/ask")
def ask(data: Question):
    try:
        question = data.question

        # -------------------------------
        # RETRIEVE
        # -------------------------------
        docs, metas = retrieve_context(question)

        context = "\n\n".join(docs[:4])

        # -------------------------------
        # DIRECT MODE (FAST PATH)
        # -------------------------------
        if len(question.split()) <= 6 or is_definition_question(question):
            if context:
                return {
                    "answer": build_direct_answer(question, context),
                    "sources": [
                        {
                            "title": clean_source(m.get("wiki_page", "")),
                            "url": m.get("wiki_page", "")
                        }
                        for m in metas[:3]
                    ]
                }

        # -------------------------------
        # GEMINI FALLBACK
        # -------------------------------
        prompt = f"""
You are a strict internal policy assistant.

ONLY use the context below.

If the answer is not in context, say:
"I could not find relevant information in the company policies."

CONTEXT:
{context}

QUESTION:
{question}
"""

        answer = safe_generate_content(prompt)

        return {
            "answer": answer,
            "sources": [
                {
                    "title": clean_source(m.get("wiki_page", "")),
                    "url": m.get("wiki_page", "")
                }
                for m in metas[:3]
            ]
        }

    except Exception as e:
        print("ERROR:", repr(e))
        traceback.print_exc()

        return {
            "answer": "Internal server error",
            "sources": []
        }
