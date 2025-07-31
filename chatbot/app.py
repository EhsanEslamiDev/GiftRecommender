"""
Tiny Streamlit UI that drives the LangGraph pipeline.
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph import ChatState, chat_pipeline 

# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Supermarket Assistant", page_icon="🛒")
st.title(" Supermarket Assistant")

# Initialise persistent chat state once
if "state" not in st.session_state:
    st.session_state.state = ChatState()

state = st.session_state.state
chat_box = st.container()

# Render existing conversation
for m in state.messages:
    role = "assistant" if isinstance(m, AIMessage) else "user"
    with chat_box.chat_message(role):
        st.markdown(m.content)

# If this is the very first run, get the initial “What product …?” prompt
if not state.messages:
    reply = chat_pipeline("", state)        # empty input, just to start the loop
    with chat_box.chat_message("assistant"):
        st.markdown(reply)

# User input
user_prompt = st.chat_input("Describe the product you need…")

# When the user submits, run the pipeline again
if user_prompt:
    with chat_box.chat_message("user"):
        st.markdown(user_prompt)

    reply = chat_pipeline(user_prompt, state)
    with chat_box.chat_message("assistant"):
        st.markdown(reply)
