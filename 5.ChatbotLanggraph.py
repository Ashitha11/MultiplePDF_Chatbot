import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime
import os

# Load environment variables
load_dotenv()

# Define the state structure for LangGraph
class ChatbotState(TypedDict):
    pdf_files: List[Any]  # List of uploaded PDF file objects (Streamlit UploadedFile)
    text_chunks: List[str]  # Extracted text chunks from PDFs
    metadata_list: List[Dict[str, Any]]  # Metadata for each chunk
    vector_store: Any  # FAISS vector store
    convo_chain: Any  # ConversationalRetrievalChain instance
    user_query: str  # Current user question
    chat_history: List[Any]  # Chat history messages
    response: str  # Generated response

# Node 1: Extract text from PDFs
def extract_text_node(state: ChatbotState) -> ChatbotState:
    if not state.get("pdf_files"):
        return state
    
    all_text_chunks = []
    metadata_list = []
    sorted_pdfs = sorted(((pdf, datetime.now()) for pdf in state["pdf_files"]), key=lambda x: x[1], reverse=True)
    
    for pdf, timestamp in sorted_pdfs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text.encode("utf-8", "ignore").decode("utf-8")
        
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        all_text_chunks.extend(chunks)
        metadata_list.extend([{"filename": pdf.name, "timestamp": timestamp, "score": 1.0}] * len(chunks))
    
    state["text_chunks"] = all_text_chunks
    state["metadata_list"] = metadata_list
    return state

# Node 2: Build the vector store
def build_vector_store_node(state: ChatbotState) -> ChatbotState:
    if not state.get("text_chunks") or not state.get("metadata_list"):
        return state
    
    processed_chunks = [t.encode("utf-8", "ignore").decode("utf-8") for t in state["text_chunks"]]
    vector_store = FAISS.from_texts(processed_chunks, OpenAIEmbeddings(), metadatas=state["metadata_list"])
    state["vector_store"] = vector_store
    return state

# Node 3: Initialize the chatbot
def initialize_chatbot_node(state: ChatbotState) -> ChatbotState:
    if not state.get("vector_store"):
        return state
    
    system_prompt = "You are a helpful assistant. Prioritize answers from the most recently uploaded PDF first."
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    memory.save_context({"content": system_prompt}, {"content": "System prompt initialized."})
    
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=state["vector_store"].as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    state["convo_chain"] = convo_chain
    state["chat_history"] = memory.buffer_as_messages
    return state

# Node 4: Handle user query and generate response
def handle_query_node(state: ChatbotState) -> ChatbotState:
    if not state.get("convo_chain") or not state.get("user_query"):
        return state
    
    response = state["convo_chain"]({"question": state["user_query"]})
    state["chat_history"] = response["chat_history"]
    state["response"] = response["answer"]
    return state

# Build the LangGraph workflow
def build_graph():
    workflow = StateGraph(ChatbotState)
    
    # Add nodes
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("build_vector_store", build_vector_store_node)
    workflow.add_node("initialize_chatbot", initialize_chatbot_node)
    workflow.add_node("handle_query", handle_query_node)
    
    # Define edges
    workflow.set_entry_point("extract_text")
    workflow.add_edge("extract_text", "build_vector_store")
    workflow.add_edge("build_vector_store", "initialize_chatbot")
    workflow.add_edge("initialize_chatbot", "handle_query")
    workflow.add_edge("handle_query", END)
    
    return workflow.compile()

# Streamlit UI and integration with LangGraph
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    # Assuming css, bot_template, user_template are defined elsewhere
    # st.write(css, unsafe_allow_html=True)
    
    # Initialize session state
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "chatbot_state" not in st.session_state:
        st.session_state.chatbot_state = {
            "pdf_files": [],
            "text_chunks": [],
            "metadata_list": [],
            "vector_store": None,
            "convo_chain": None,
            "user_query": "",
            "chat_history": [],
            "response": ""
        }
    
    st.header("Chat with multiple PDFs 📚")
    st.write("Upload PDFs and interact with them!")
    
    # User query input
    user_question = st.text_input("Ask a question")
    
    # Sidebar for PDF uploads
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Upload") and pdf_files:
            with st.spinner("Processing PDFs..."):
                # Update state with uploaded PDFs
                initial_state = st.session_state.chatbot_state.copy()
                initial_state["pdf_files"] = pdf_files
                
                # Run the graph up to chatbot initialization
                result = st.session_state.graph.invoke(initial_state)
                st.session_state.chatbot_state = result
                
                # Display uploaded PDFs
                sorted_pdfs = sorted(((pdf, datetime.now()) for pdf in pdf_files), key=lambda x: x[1], reverse=True)
                st.write("Uploaded PDFs (Most Recent First):")
                for pdf, timestamp in sorted_pdfs:
                    st.write(f"📄 {pdf.name} (Uploaded at: {timestamp})")
                most_recent_pdf = sorted_pdfs[-1][0].name
                st.write(f"🆕 Most Recently Uploaded PDF: **{most_recent_pdf}**")
                
                st.success("PDFs processed successfully!")
    
    # Handle user query
    if user_question:
        with st.spinner("Generating response..."):
            query_state = st.session_state.chatbot_state.copy()
            query_state["user_query"] = user_question
            query_state["response"] = ""  # Reset response for new query
            
            result = st.session_state.graph.invoke(query_state)
            st.session_state.chatbot_state = result
            
            # Display chat history (simplified, adapt with your templates)
            for i, message in enumerate(result["chat_history"]):
                if i % 2 == 0:
                    st.write(f"User: {message.content}")
                else:
                    st.write(f"Bot: {message.content}")

if __name__ == "__main__":
    main()
