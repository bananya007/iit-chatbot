"""
app/comparison_app.py
Side-by-side IIT Chatbot comparison UI.

Each panel is fully independent — its own input, history, and state.
Calls separate backend APIs so zero code is shared between the two models.

Run from the repo root:
    streamlit run app/comparison_app.py

Configure backend URLs in the sidebar or via environment variables:
    CHATBOT_A_URL=http://localhost:8000   (your model)
    CHATBOT_B_URL=http://teammate-url     (teammate's model)
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_A_URL = os.getenv("CHATBOT_A_URL", "http://localhost:8000")
DEFAULT_B_URL = os.getenv("CHATBOT_B_URL", "https://bessie-pinnate-noumenally.ngrok-free.dev")

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IIT Chatbot Comparison",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align:center; color:#CC0000;'>🎓 IIT Chatbot Comparison</h1>
    <hr style='border: 1px solid #CC0000;'>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — backend URLs ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Backend URLs")
    url_a = st.text_input("Chatbot A URL", value=DEFAULT_A_URL)
    url_b = st.text_input("Chatbot B URL", value=DEFAULT_B_URL)
    st.markdown("---")
    if st.button("🗑️ Clear both conversations"):
        for key in ["history_a", "history_b", "tuition_state_a",
                    "calendar_state_a", "input_a", "input_b"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Session state init ─────────────────────────────────────────────────────────

def _init():
    defaults = {
        "history_a":        [],
        "history_b":        [],
        "tuition_state_a":  {},
        "calendar_state_a": {},
        # Teammate manages their own state server-side or statelessly —
        # we only track history for display on their panel.
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── API call helpers ────────────────────────────────────────────────────────────

def call_chatbot_a(query: str) -> dict:
    """Call your stateless FastAPI backend. Passes full state each turn."""
    try:
        resp = requests.post(
            f"{url_a}/chat",
            json={
                "query":          query,
                "history":        st.session_state.history_a,
                "tuition_state":  st.session_state.tuition_state_a,
                "calendar_state": st.session_state.calendar_state_a,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"reply": f"⚠️ Cannot reach Chatbot A at {url_a}. Is the server running?",
                "is_clarification": False, "domains": [],
                "tuition_state": st.session_state.tuition_state_a,
                "calendar_state": st.session_state.calendar_state_a}
    except Exception as e:
        return {"reply": f"⚠️ Error: {e}", "is_clarification": False, "domains": [],
                "tuition_state": st.session_state.tuition_state_a,
                "calendar_state": st.session_state.calendar_state_a}


def call_chatbot_b(query: str) -> dict:
    """Call teammate's backend at /ask. Their API uses 'prompt' and returns 'response'."""
    try:
        resp = requests.post(
            f"{url_b}/ask",
            json={"prompt": query},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Normalise their response shape to match our internal convention
        return {
            "reply":            data.get("response", ""),
            "is_clarification": data.get("is_clarification", False),
        }
    except requests.exceptions.ConnectionError:
        return {"reply": f"⚠️ Cannot reach Chatbot B at {url_b}. Is the server running?",
                "is_clarification": False}
    except Exception as e:
        return {"reply": f"⚠️ Error: {e}", "is_clarification": False}


# ── Chat panel renderer ─────────────────────────────────────────────────────────

def _safe_md(text: str) -> str:
    """Escape dollar signs so Streamlit doesn't render them as LaTeX."""
    return text.replace("$", r"\$")


def render_history(history: list):
    """Render conversation history inside a panel."""
    for msg in history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(_safe_md(msg["content"]))
        else:
            with st.chat_message("assistant"):
                st.markdown(_safe_md(msg["content"]))
                if msg.get("is_clarification"):
                    st.caption("clarification")


# ── Two-column layout ──────────────────────────────────────────────────────────

st.markdown(
    """<style>
    [data-testid="column"]:first-child {
        border-right: 2px solid #CC0000;
        padding-right: 2rem;
    }
    [data-testid="column"]:last-child {
        padding-left: 2rem;
    }
    </style>""",
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)

# ── Panel A ────────────────────────────────────────────────────────────────────

with col_a:
    st.markdown("### Chatbot A &nbsp; `Ananya`", unsafe_allow_html=True)
    st.markdown(
        "<div style='border:1px solid #ddd; border-radius:8px; padding:12px; "
        "min-height:400px; max-height:500px; overflow-y:auto;'>",
        unsafe_allow_html=True,
    )
    render_history(st.session_state.history_a)
    st.markdown("</div>", unsafe_allow_html=True)

    query_a = st.text_input(
        "Ask me anything related to IIT...",
        key="input_a",
        label_visibility="collapsed",
        placeholder="Ask me anything related to IIT...",
    )
    send_a = st.button("Send ➤", key="send_a", use_container_width=False)

    if send_a and query_a.strip():
        with st.spinner("Chatbot A is thinking…"):
            result = call_chatbot_a(query_a.strip())

        reply        = result.get("reply", "")
        is_clar      = result.get("is_clarification", False)

        # Update state
        st.session_state.tuition_state_a  = result.get("tuition_state",  {})
        st.session_state.calendar_state_a = result.get("calendar_state", {})
        st.session_state.history_a.append({"role": "user",      "content": query_a.strip()})
        st.session_state.history_a.append({"role": "assistant",  "content": reply,
                                            "is_clarification": is_clar})
        st.rerun()

# ── Panel B ────────────────────────────────────────────────────────────────────

with col_b:
    st.markdown("### Chatbot B &nbsp; `Karthik`", unsafe_allow_html=True)
    st.markdown(
        "<div style='border:1px solid #ddd; border-radius:8px; padding:12px; "
        "min-height:400px; max-height:500px; overflow-y:auto;'>",
        unsafe_allow_html=True,
    )
    render_history(st.session_state.history_b)
    st.markdown("</div>", unsafe_allow_html=True)

    query_b = st.text_input(
        "Ask me anything related to IIT...",
        key="input_b",
        label_visibility="collapsed",
        placeholder="Ask me anything related to IIT...",
    )
    send_b = st.button("Send ➤", key="send_b", use_container_width=False)

    if send_b and query_b.strip():
        with st.spinner("Chatbot B is thinking…"):
            result = call_chatbot_b(query_b.strip())

        reply   = result.get("reply", "")
        is_clar = result.get("is_clarification", False)

        st.session_state.history_b.append({"role": "user",      "content": query_b.strip()})
        st.session_state.history_b.append({"role": "assistant",  "content": reply,
                                            "is_clarification": is_clar})
        st.rerun()
