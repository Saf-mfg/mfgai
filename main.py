from dotenv import load_dotenv
from urllib.parse import urlparse
import traceback
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from rag_db import collection


# -------------------------------
# INIT
# -------------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = FastAPI()


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
# REQUEST MODEL
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


# -------------------------------
# GEMINI
# -------------------------------

def ask_gemini(question, context):

    prompt = f"""
You are an internal company policy assistant.

Answer the user's question using ONLY the policy context provided.

Rules:
- Do not invent policies.
- Do not use outside knowledge.
- If the context does not contain the answer, say:
  "I could not find relevant information in the company policies."
- Give a clear, helpful explanation.
- Mention relevant policy names when possible.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# -------------------------------
# RAG RETRIEVAL
# -------------------------------

def retrieve_context(query):

    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=[
            "documents",
            "metadatas"
        ]
    )

    docs = results.get(
        "documents",
        [[]]
    )[0]

    metas = results.get(
        "metadatas",
        [[]]
    )[0]

    return docs, metas


# -------------------------------
# MAIN ENDPOINT
# -------------------------------

@app.post("/ask")
def ask(data: Question):

    try:

        question = data.question


        # -------------------------------
        # GET RELEVANT POLICY CHUNKS
        # -------------------------------

        docs, metas = retrieve_context(question)


        if not docs:

            return {
                "answer": "I could not find relevant information in the company policies.",
                "sources": []
            }


        context = "\n\n".join(
            docs
        )


        # -------------------------------
        # GEMINI REASONING
        # -------------------------------

        try:

            answer = ask_gemini(
                question,
                context
            )

        except Exception as e:

            print(
                "GEMINI ERROR:",
                repr(e)
            )

            # fallback: still return retrieved policies
            return {
                "answer": (
                    "I found relevant policy information, "
                    "but the AI response service is temporarily unavailable. "
                    "Please review the sources below."
                ),
                "sources": [
                    {
                        "title": clean_source(
                            m.get("wiki_page", "")
                        ),
                        "url": m.get("wiki_page", "")
                    }
                    for m in metas[:5]
                ]
            }


        # -------------------------------
        # RESPONSE
        # -------------------------------

        return {
            "answer": answer,
            "sources": [
                {
                    "title": clean_source(
                        m.get("wiki_page", "")
                    ),
                    "url": m.get("wiki_page", "")
                }
                for m in metas[:5]
            ]
        }


    except Exception as e:

        print(
            "SERVER ERROR:",
            repr(e)
        )

        traceback.print_exc()

        return {
            "answer": "Something went wrong processing your request.",
            "sources": []
        }
