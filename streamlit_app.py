"""
Streamlit UI for the Nour persona chatbot — same two-layer pipeline as
chat.py (FAISS scope gate + Ollama generation), just with a browser chat
window instead of a terminal loop.

Requires an Ollama model already created (see generate_modelfile.py):
    ollama create nour -f Modelfile

Usage:
    streamlit run streamlit_app.py
"""

import ollama
import streamlit as st

from chat import FALLBACK, SCOPE_THRESHOLD, ScopeGate
from data.persona_data import EM_PROMPT

st.set_page_config(page_title="Chat with Nour", page_icon="💬")


@st.cache_resource(show_spinner="Loading scope gate (embedding model + FAISS index)...")
def load_gate() -> ScopeGate:
    return ScopeGate()


st.title("Chat with Nour")

with st.sidebar:
    model_name = st.text_input("Ollama model", value="nour")
    threshold = st.slider("Scope gate threshold", 0.0, 1.0, SCOPE_THRESHOLD, 0.01)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    if st.button("Clear conversation"):
        st.session_state.messages = [{"role": "system", "content": EM_PROMPT}]
        st.rerun()

gate = load_gate()
gate.threshold = threshold

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": EM_PROMPT}]

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_msg = st.chat_input("Ask Nour anything...")
if user_msg:
    with st.chat_message("user"):
        st.markdown(user_msg)

    last_assistant_msg = next(
        (
            m["content"]
            for m in reversed(st.session_state.messages)
            if m["role"] == "assistant" and m["content"] != FALLBACK
        ),
        None,
    )
    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.chat_message("assistant"):
        if not gate.is_in_scope(user_msg, last_assistant_msg):
            reply = FALLBACK
            st.markdown(reply)
        else:
            with st.spinner("Thinking..."):
                response = ollama.chat(
                    model=model_name,
                    messages=st.session_state.messages,
                    options={"temperature": temperature},
                )
            reply = response["message"]["content"]
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
