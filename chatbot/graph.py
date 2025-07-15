"""
Two‑phase LangGraph workflow (ask ↔ generate) with a separate routing node
that decides which phase to execute. The graph always ends after one phase,
preventing infinite recursion.
"""
from typing import List, Union, Optional, Literal
from pydantic import BaseModel

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from utils import generate_response

# ───────────────────────────────────────────────────────────────────────────────
# State model
# ───────────────────────────────────────────────────────────────────────────────
class ChatState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = []
    awaiting_description: bool = False     # True → we’re waiting for user text
    response: Optional[str] = None         # assistant’s most recent utterance

    def __getitem__(self, k): return getattr(self, k)
    def __setitem__(self, k, v): setattr(self, k, v)
    def keys(self): return self.__dict__.keys()

def last_user_message(state: ChatState) -> str:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""

# ───────────────────────────────────────────────────────────────────────────────
# Nodes
# ───────────────────────────────────────────────────────────────────────────────
def ask_node(state: ChatState) -> ChatState:
    """Ask the customer to describe the product."""
    msg = AIMessage(content="What product are you looking for?")
    state.messages.append(msg)
    state.awaiting_description = True
    state.response = msg.content
    return state

def generate_node(state: ChatState) -> ChatState:
    """Recommend a product based on the customer’s description."""
    description = last_user_message(state)
    answer      = generate_response(description)
    msg         = AIMessage(content=answer)

    state.messages.append(msg)
    state.response = msg.content
    state.awaiting_description = False      # we’re done—back to asking
    return state

def route(state: ChatState) -> Literal["ask", "generate"]:
    """
    Decide which node to execute:
    • If we are still waiting for a description, but now we have one → generate
    • Otherwise → ask
    """
    return "generate" if state.awaiting_description else "ask"

# ───────────────────────────────────────────────────────────────────────────────
# Build the workflow
# ───────────────────────────────────────────────────────────────────────────────
workflow = StateGraph(ChatState)

# A dummy pass‑through node so we can attach conditional edges
workflow.add_node("router", lambda s: s)
workflow.add_node("ask", ask_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route)

app = workflow.compile(checkpointer=InMemorySaver())

# ───────────────────────────────────────────────────────────────────────────────
# Helper used by Streamlit
# ───────────────────────────────────────────────────────────────────────────────
def chat_pipeline(user_input: str, state: ChatState) -> str:
    """
    Append a new user message (if any), run ONE graph step, copy the values that
    came back (as a plain dict) into our long‑lived ChatState, and return the
    assistant’s text reply.
    """
    if user_input.strip():
        state.messages.append(HumanMessage(content=user_input))

    # `app.invoke` returns a dict → pull its data back into the Pydantic object
    new_state = app.invoke(state, config={"configurable": {"thread_id": "demo"}})

    state.messages             = new_state["messages"]
    state.awaiting_description = new_state["awaiting_description"]
    state.response             = new_state["response"]
    return state.response      # string we’ll show in the UI

