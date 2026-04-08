"""
app/api.py
FastAPI wrapper around the IIT chatbot orchestrator.

Stateless design — all conversational state is passed in with each request
and returned in the response. The client (Streamlit comparison UI) owns state.

Run from the repo root:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

API contract (shared with teammate so they can build a compatible endpoint):

    POST /chat
    Request:
        {
            "query":          "string",
            "history":        [{"role": "user"|"assistant", "content": "string"}, ...],
            "tuition_state":  {}   // pass {} on first turn
            "calendar_state": {}   // pass {} on first turn
        }
    Response:
        {
            "reply":           "string",
            "is_clarification": bool,
            "domains":         ["DOCUMENTS", ...],
            "tuition_state":   {...},
            "calendar_state":  {...}
        }

    GET /health
    Response: {"status": "ok", "model": "group7"}
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.orchestrator import handle_turn

app = FastAPI(title="IIT Chatbot API", version="1.0")

# Allow Streamlit (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:          str
    history:        list       = []
    tuition_state:  dict       = {}
    calendar_state: dict       = {}


class ChatResponse(BaseModel):
    reply:           str
    is_clarification: bool
    domains:         list
    tuition_state:   dict
    calendar_state:  dict


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "group7"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Single-turn chat endpoint. Stateless — caller owns all state.
    """
    reply, domains, tuition_state, calendar_state, is_clar = handle_turn(
        query=req.query,
        history=req.history,
        tuition_state=req.tuition_state,
        calendar_state=req.calendar_state,
        stream=False,   # no streaming over HTTP
    )

    return ChatResponse(
        reply=str(reply),
        is_clarification=is_clar,
        domains=domains,
        tuition_state=tuition_state,
        calendar_state=calendar_state,
    )
