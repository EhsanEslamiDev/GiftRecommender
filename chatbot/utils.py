"""
Shared utilities: prompt templates, retriever factory, and the single‑shot
`generate_response` function that combines the prompt with the language model.
"""
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.prompts import PromptTemplate, format_document
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# ────────────────────────────────────────────────────────────────────────────────
# Environment & model
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGUAGE_MODEL  = os.getenv("LANGUAGE_MODEL", "gpt-3.5-turbo")

model = ChatOpenAI(model_name=LANGUAGE_MODEL, temperature=0)

# ────────────────────────────────────────────────────────────────────────────────
# Prompt: system + user
# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """
You are a helpful supermarket clerk.  
Customer need (free‑text description):  
{description}

Below is information on several products we stock.  
Choose **the ONE** that best matches the customer’s request and present all
relevant details about that product. Do **not** invent new products—only pick
from those shown.

Products:
{context}
""".strip()

system_msg  = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
human_msg   = HumanMessagePromptTemplate.from_template(
    input_variables=["description", "context"],
    template="{description}",
)
CHAT_PROMPT = ChatPromptTemplate.from_messages([system_msg, human_msg])

# Template for rendering each product document (metadata + content)
DOC_TEMPLATE = PromptTemplate.from_template("""
Product: {product}
Category: {category} → {sub_category}
Brand: {brand}
Sale Price: ₹{sale_price}, Market Price: ₹{market_price}
Type: {type}, Rating: {rating}
Description:
{page_content}
""".strip())

def format_docs(docs) -> str:
    """Render retrieved documents for insertion into the system prompt."""
    return "\n–––\n".join(format_document(d, DOC_TEMPLATE) for d in docs)

# ────────────────────────────────────────────────────────────────────────────────
# Retriever factory
# ────────────────────────────────────────────────────────────────────────────────
def get_retriever() -> QdrantVectorStore.as_retriever:
    """Return a Qdrant retriever (k=3) for the product catalogue."""
    qdrant_url      = os.getenv("QDRANT_URL")
    qdrant_api_key  = os.getenv("QDRANT_API_KEY")
    collection_name = "product_catalog"

    embeddings  = OpenAIEmbeddings()
    vecstore    = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=False,
        timeout=60,
    )
    return vecstore.as_retriever(search_kwargs={"k": 3})

# Instantiate once and keep for the whole session
RETRIEVER = get_retriever()

# ────────────────────────────────────────────────────────────────────────────────
# End‑to‑end response generator
# ────────────────────────────────────────────────────────────────────────────────
def generate_response(description: str) -> str:
    """
    (1) Retrieve top‑k product pages → (2) format them → (3) build prompt
    → (4) run LLM → return “assistant” text.
    """
    docs     = RETRIEVER.invoke(description)
    context  = format_docs(docs)

    chain = (
        {"description": RunnablePassthrough(),
         "context":     RunnablePassthrough()}
        | CHAT_PROMPT
        | model
        | StrOutputParser()
    )
    return chain.invoke({"description": description, "context": context})
