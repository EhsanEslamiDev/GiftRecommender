# Super Market Assistant

A multi-agent chatbot that suggests products from a specific supermarket’s dataset based on a natural-language description of user needs. It clarifies requests, formulates a query, retrieves relevant products with RAG, and generates tailored suggestions. Built with LangGraph for agent orchestration, Streamlit for the chat interface, and a Qdrant vector database for retrieval.

## Features

- Multi-agent workflow: clarify intent, build query, retrieve via RAG, and generate personalized suggestions.  
- Agent orchestration implemented with LangGraph.  
- Streamlit-based chat interface.  
- Vector search backed by a Qdrant server.

## How it works

1. Clarifier agent asks a follow-up question to refine the initial request.  
2. Query builder agent converts the initial and refined descriptions into a structured query.  
3. Retrieval agent uses the query to fetch the most relevant products from the database using RAG.  
4. Suggestion agent produces a final product recommendation grounded in retrieved context.

## Tech stack

- Orchestration: LangGraph  
- Interface: Streamlit  
- Retrieval: RAG over a Qdrant vector database  
- LLM provider: OpenAI (via API)

## Getting started

1. Create and activate a Python virtual environment in the terminal.  
2. Install dependencies from requirements.txt.  
3. Open example.env, add the OpenAI API key and the LangChain API key, then save and rename it to .env.  
4. From the repository root, start the app:
   ```
   streamlit run chatbot/app.py
   ```
5. Follow the link shown in the terminal to open the chatbot and begin using it.
