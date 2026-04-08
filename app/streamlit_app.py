"""
app/streamlit_app.py
IIT Student Assistant — Streamlit web interface.

Run from the ElasticSearch/ root:
    streamlit run app/streamlit_app.py
"""

import os
import types

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.core.orchestrator import handle_turn

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IIT Student Assistant",
    page_icon="🎓",
    layout="centered",
)
st.title("IIT Student Assistant")
st.caption("Academic policies · Tuition & fees · Academic calendar · Department contacts")

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: list = []
if "tuition_state" not in st.session_state:
    st.session_state.tuition_state: dict = {}
if "calendar_state" not in st.session_state:
    st.session_state.calendar_state: dict = {}

# ── Render existing conversation ──────────────────────────────────────────────

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Handle new input ──────────────────────────────────────────────────────────

if query := st.chat_input("Ask about policies, tuition, calendar, or contacts…"):

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        reply, domains, st.session_state.tuition_state, st.session_state.calendar_state, is_clar = handle_turn(
            query,
            st.session_state.history,
            st.session_state.tuition_state,
            st.session_state.calendar_state,
            stream=True,
        )

        if isinstance(reply, types.GeneratorType):
            reply = st.write_stream(reply)
        else:
            st.write(reply)

        if os.getenv("STREAMLIT_DEBUG") and domains:
            with st.expander("Debug", expanded=False):
                st.write(f"**Domains:** {domains}")
                st.write(f"**Reply preview:** {str(reply)[:200]}…")

    st.session_state.history.append({"role": "user",      "content": query})
    st.session_state.history.append({"role": "assistant", "content": reply})
