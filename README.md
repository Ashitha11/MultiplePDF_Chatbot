# Multi-PDF Chatbot

A powerful chatbot that allows users to upload multiple PDFs and query them for relevant answers. This project leverages **LangChain, LangGraph, vector databases, and multimodal Retrieval-Augmented Generation (RAG)** to enhance chatbot capabilities.

## Features

- **Multi-PDF Support:** Upload multiple PDFs and get responses based on the most relevant document.
- **LangChain Implementation:** Uses LangChain to process, store, and retrieve data.
- **LangGraph Integration (Branch: ****************************`langgraph`****************************):** Implements LangGraph for advanced workflow control.
- **Multimodal RAG (Branch: ****************************`multimodal-rag`****************************):** Supports text and images for richer responses.
- **Fast & Accurate:** Optimized retrieval for precise document-based responses.

## Installation & Setup

### Prerequisites

Ensure you have:

- Python 3.8+
- pip
- Virtual environment (recommended)
- OpenAI API Key (if using OpenAI models)
- Vector database (like Pinecone or ChromaDB)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashitha11/MultiplePDF_Chatbot.git
   cd MultiplePDF_Chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```
4. Run the chatbot:
   ```bash
   python app.py
   ```

## Usage

- Upload PDFs via the interface.
- Ask questions, and the chatbot retrieves answers based on the latest and most relevant document.
- Supports multiple PDFs for context-aware querying.

## Branch Details

### `main`

- Standard multi-PDF chatbot implementation with LangChain and vector databases.

### `langgraph`

- Introduces LangGraph for advanced conversational flows and improved agent-based processing.

### `multimodal-rag`

- Adds multimodal capabilities (text + images) using RAG for better context understanding.

## Example Queries

- "Summarize the key points from the latest uploaded PDF."
- "What does the document say about XYZ?"
- "Can you generate an image-based response using the multimodal approach?"

## Future Improvements

- External API integration.
- Real-time streaming answers.
- More database integrations.
- Improved multimodal capabilities.
- multi-AI agent workflow 

## Technologies Used

- Python
- LangChain
- LangGraph
- OpenAI API
- Vector Databases (Pinecone/ChromaDB)
- Streamlit (if UI is included)

