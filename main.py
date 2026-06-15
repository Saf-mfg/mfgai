from dotenv import load_dotenv
from urllib.parse import urlparse
from collections import Counter
import traceback
import re

import os
import zipfile

if not os.path.exists("./humhub_db"):
    print("📦 Extracting humhub_db.zip...")

    with zipfile.ZipFile("humhub_db.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    print("✅ Database extracted")

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


LOOKUP_WORDS = {
    "policy",
    "procedure",
    "guideline"
}

STOP_WORDS = {
    "what","is","are","the","a","an",
    "how","do","does","can","i",
    "we","you","of","for","to",
    "please","tell","about"
}

def extract_keywords(text):
    words = re.findall(r"\w+", text.lower())
    return {
        w for w in words
        if len(w) > 2 and w not in STOP_WORDS
    }

def retrieval_confidence(docs, query):

    keywords = extract_keywords(query)

    combined = " ".join(docs).lower()

    score = 0

    for word in keywords:
        if word in combined:
            score += 1

    # boost definition signals
    if "definition" in query.lower() or "define" in query.lower():
        if any(x in combined for x in ["means", "defined", "is when", "refers to"]):
            score += 3

    return score


# -------------------------------
# DIRECT ANSWER (NO GEMINI)
# -------------------------------
def build_direct_answer(question, combined_doc):

    keywords = extract_keywords(question)

    # normalise PDF extraction
    text = re.sub(r'\s+', ' ', combined_doc)

    sentences = re.split(
        r'(?<=[.!?])\s+(?=\d+\.?\d*|\w)',
        text
    )

    scored = []

    for sentence in sentences:

        lower = sentence.lower()
        score = 0

        # definition priority
        if (
            "harassment is" in lower
            or "means" in lower
            or "defined as" in lower
            or "refers to" in lower
        ):
            score += 100

        # continuation boost
        if "single incident can amount" in lower:
            score += 50

        if "disciplinary procedure" in lower:
            score += 30

        if "dismissal" in lower:
            score += 20


        for keyword in keywords:
            if keyword in lower:
                score += 5

        # remove policy metadata/noise
        if "policy title" in lower:
            score -= 50

        if "version" in lower:
            score -= 50

        if "review due" in lower:
            score -= 50

        if len(sentence) > 40:
            scored.append((score, sentence))


    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    answer_sentences = []

    for score, sentence in scored:
        if score >= 5:
            answer_sentences.append(sentence)

        if len(answer_sentences) >= 2:
            break


    return " ".join(answer_sentences)

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
    top_chunks = docs[:5]

    policy_scores = Counter()

    for meta in metas[:10]:
        title = (meta.get("policy_title") or "").lower()

        score = 0

        # keyword overlap with query
        for word in extract_keywords(query):
            if word in title:
                score += 3

        # strong boosts for exact matches
        if "maternity" in title and "maternity" in query.lower():
            score += 10

        if "harassment" in title and "harassment" in query.lower():
            score += 10

        if "pregnancy" in title and "maternity" in query.lower():
            score -= 5  # important: stop wrong selection

        policy_scores[meta.get("policy_title", "")] += score

    best_policy = None

    if policy_scores:
        best_policy, best_score = policy_scores.most_common(1)[0]

        if best_score <= 0:
            best_policy = None

    for i, doc in enumerate(docs[:5]):
        print(f"\nTOP CHUNK {i+1}")
        print(doc[:500])

    print(f"Retrieved {len(docs)} chunks")

    if not docs:
        return "", [], "", 0

    context_parts = []
    sources = []
    seen_sources = set()

    policy_chunks = []

    for doc, meta in zip(docs, metas):
        if best_policy and meta.get("policy_title") == best_policy:
            policy_chunks.append(doc)

    # fallback if filtering fails
    if not policy_chunks:
        return "", [], "", 0, None

    combined_doc = "\n".join(policy_chunks)
    
    if best_policy:
        filtered = [
            (doc, meta)
            for doc, meta in zip(docs, metas)
            if meta.get("policy_title") == best_policy
        ]
    else:
        filtered = [
            (doc, meta)
            for doc, meta in zip(docs, metas)
            if meta.get("policy_title")
        ]

    if not filtered:
        filtered = list(zip(policy_chunks, metas[:len(policy_chunks)]))

    for doc, meta in filtered:

        wiki_page = meta.get(
            "wiki_page",
            "unknown"
        )

        if wiki_page in seen_sources:
            continue

        seen_sources.add(wiki_page)

        sources.append({
            "title": clean_source(wiki_page),
            "url": wiki_page
        })

        context_parts.append(
            f"SOURCE: {wiki_page}\nCONTENT:\n{doc[:1200]}"
        )

        if len(sources) >= 3:
            break

    context = "\n\n---\n\n".join(context_parts)[:4000]
    score = retrieval_confidence(
        policy_chunks[:5],
        query
    )
    return context, sources, combined_doc, score, best_policy

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
        context, sources, combined_doc, score, best_policy = search_humhub(question)
        print("\n===================")
        print("QUESTION:", question)
        print("KEYWORDS:", extract_keywords(question))
        print("SCORE:", score)
        print("TOP TEXT:")
        print(combined_doc[:500])
        print("===================\n")

        # -------------------------------
        # LEVEL 1: STRONG MATCH (NO GEMINI)
        # -------------------------------
        keywords = extract_keywords(
            question
        )

        is_policy_lookup = any(
            word in question.lower()
            for word in LOOKUP_WORDS
        )
        
        if (
            is_policy_lookup
            and best_policy
            and score >= 2
        ):
            return {
                "answer": build_direct_answer(
                    question,
                    combined_doc
                ),
                "sources": sources
            }
        
        simple_patterns = [
            "what is",
            "define",
            "who is",
            "where is"
        ]

        simple_question = any(
            question.lower().startswith(p)
            for p in simple_patterns
        )
        
        if (
            simple_question
            and score >= 1
        ):
            print("DIRECT ANSWER ROUTE")
            
            return {
                "answer": build_direct_answer(
                    question,
                    combined_doc
                ),
                "sources": sources
            }

        if (
            len(question.split()) <= 4
            and score >= 1
        ):
            print("SHORT QUERY ROUTE")

            return {
                "answer": build_direct_answer(
                    question,
                    combined_doc
                ),
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

        print("GEMINI ROUTE")
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
