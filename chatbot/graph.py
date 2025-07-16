"""
Four‑phase LangGraph workflow:
    router → (ask | clarify_question | refine_description | generate)

The router decides what to do next, so we never repeat the “What product …?”
prompt and we never recurse indefinitely.
"""
from typing import List, Union, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from utils import (
    generate_response,
    generate_clarifying_question,
    generate_refined_description,
)

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def last_user_message(state: "ChatState") -> Optional[str]:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage):
            return m.content
    return None


# ───────────────────────────────────────────────────────────────────────────────
# State model
# ───────────────────────────────────────────────────────────────────────────────
class ChatState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = []
    response: Optional[str] = None      # assistant’s latest utterance

    # workflow flags / scratch‑pad
    description: Optional[str] = None          # customer’s first answer
    refined_description: Optional[str] = None  # after clarification
    generated: bool = False                    # recommendation already sent?

    # allow dict‑style access (LangGraph convenience)
    def __getitem__(self, k):  # noqa: D401 (property‑like)
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def keys(self):
        return self.__dict__.keys()


# ───────────────────────────────────────────────────────────────────────────────
# Nodes
# ───────────────────────────────────────────────────────────────────────────────
def ask_node(state: ChatState) -> ChatState:
    """Initial greeting: trigger only once, right at the start."""
    msg = AIMessage(content="What product are you looking for?")
    state.messages.append(msg)
    state.response = msg.content
    return state


def clarify_question_node(state: ChatState) -> ChatState:
    """Ask ONE follow‑up question about the customer’s description."""
    description = last_user_message(state)
    state.description = description                      # persist
    question = generate_clarifying_question(description)
    msg = AIMessage(content=question)
    state.messages.append(msg)
    state.response = msg.content
    return state


def refine_description_node(state: ChatState) -> ChatState:
    """Fuse the first description + clarification into a single search string."""
    clarification = last_user_message(state)
    refined = generate_refined_description(
        state.description, clarification
    )
    state.refined_description = refined
    # no need to talk to the user here – we’ll jump straight into “generate”
    return state


def generate_node(state: ChatState) -> ChatState:
    """Recommend a product."""
    query = state.refined_description or state.description
    answer = generate_response(query)
    msg = AIMessage(content=answer)
    state.messages.append(msg)
    state.response = msg.content
    state.generated = True
    return state


# ───────────────────────────────────────────────────────────────────────────────
# Router
# ───────────────────────────────────────────────────────────────────────────────
def route_node(state: ChatState) -> ChatState:
    """A do‑nothing pass‑through node."""
    return state

def decide_next(state: ChatState):
    
    """Return the *label* of the next node or END."""
    if state.generated:
        return END

    if not state.messages:
        return "ask"

    last_msg_is_user = isinstance(state.messages[-1], HumanMessage)

    if state.description is None:
        return "clarify_question" if last_msg_is_user else END

    if state.refined_description is None:
        return "refine_description" if last_msg_is_user else END

    return "generate"



# ───────────────────────────────────────────────────────────────────────────────
# Build the workflow
# ───────────────────────────────────────────────────────────────────────────────
workflow = StateGraph(ChatState)

# real work nodes
workflow.add_node("ask", ask_node)
workflow.add_node("clarify_question", clarify_question_node)
workflow.add_node("refine_description", refine_description_node)
workflow.add_node("generate", generate_node)

# router hub
workflow.add_node("route", route_node)                 # <‑‑ returns ChatState
workflow.add_conditional_edges("route", decide_next)   # <‑‑ returns str | END
workflow.set_entry_point("route")

# after every “thinking” step, jump back to the router
workflow.add_edge("ask", "route")
workflow.add_edge("clarify_question", "route")
workflow.add_edge("refine_description", "route")
workflow.add_edge("generate", "route")   # will immediately hit END inside router


app = workflow.compile(checkpointer=InMemorySaver())


# ───────────────────────────────────────────────────────────────────────────────
# Helper used by Streamlit
# ───────────────────────────────────────────────────────────────────────────────
def chat_pipeline(user_input: str, state: ChatState) -> str:
    # ─── 1. Handle new user input ────────────────────────────────────────────
    if user_input.strip():
        # If a recommendation was already produced, start a fresh round
        if state.generated:
            state.generated = False
            state.description = None
            state.refined_description = None

        # Record the user’s new message
        state.messages.append(HumanMessage(content=user_input))

    # ─── 2. Run ONE LangGraph step ───────────────────────────────────────────
    new_state = app.invoke(
        state,
        config={"configurable": {"thread_id": "demo"}}
    )

    # ─── 3. Merge partial update back into our long‑lived ChatState ──────────
    for k, v in new_state.items():
        setattr(state, k, v)

    # ─── 4. Return the assistant’s latest utterance ──────────────────────────
    return state.response


