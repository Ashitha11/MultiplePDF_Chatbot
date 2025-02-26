import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime
from htmltemplates import css, bot_template, user_template
import os
import pandas as pd
from collections import defaultdict

# Load environment variables
load_dotenv()

# Define the state structure for LangGraph
class ChatbotState(TypedDict):
    pdf_files: List[Any]
    text_chunks: List[str]
    metadata_list: List[Dict[str, Any]]
    vector_store: Any
    convo_chain: Any
    user_query: str
    chat_history: List[Any]
    response: str
    comparison_table: str

# Node 1: Extract text from PDFs
def extract_text_node(state: ChatbotState) -> ChatbotState:
    if not state.get("pdf_files"):
        return state
    
    all_text_chunks = []
    metadata_list = []
    sorted_pdfs = sorted(((pdf, datetime.now()) for pdf in state["pdf_files"]), key=lambda x: x[1], reverse=True)
    
    for pdf, timestamp in sorted_pdfs:
        text = ""
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        
        print(f"Extracted text from {pdf.name}: {text[:500]}...")  # Debug
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
    
    system_prompt = """
You are a helpful and intelligent AI assistant designed to answer questions based on uploaded PDFs. 
Maintain the context of the conversation throughout and provide accurate, relevant answers by 
prioritizing the most recent PDFs. After responding, suggest insightful follow-up prompts to guide 
deeper exploration of the topic. Ensure clarity, coherence, and a natural conversational flow.
"""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    memory.save_context({"content": system_prompt}, {"content": "System prompt initialized."})
    
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.6),
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
    
    if "compare" in state["user_query"].lower() and "pdf" in state["user_query"].lower():
        state["response"] = "Generating comparison table..."
        return state
    else:
        response = state["convo_chain"]({"question": state["user_query"]})
        state["chat_history"] = response["chat_history"]
        state["response"] = response["answer"]
        return state

# Node 5: Compare PDFs and generate a verbal table
def compare_pdfs_node(state: ChatbotState) -> ChatbotState:
    if not state.get("text_chunks") or not state.get("metadata_list"):
        state["response"] = "No PDFs available to compare."
        return state
    
    # Group chunks by PDF
    pdf_data = defaultdict(list)
    for chunk, metadata in zip(state["text_chunks"], state["metadata_list"]):
        pdf_data[metadata["filename"]].append(chunk)
    
    # Prepare content for LLM comparison
    pdf_summaries = {filename: "\n".join(chunks)[:2000] for filename, chunks in pdf_data.items()}  # Limit to 2000 chars per PDF
    llm = ChatOpenAI(temperature=0.6)
    
    # Define comparison aspects (customize these based on your PDFs)
    aspects = ["main topic", "key findings", "tone or style"]
    comparison = {"Aspect": [], **{filename: [] for filename in pdf_data.keys()}}
    
    for aspect in aspects:
        prompt = f"""
        Compare the following PDFs based on their {aspect}. Provide a concise, human-like verbal description of the differences or similarities for each PDF. Return the response in the format:
        - [PDF1 Name]: Description
        - [PDF2 Name]: Description
        PDFs:
        {', '.join(pdf_summaries.keys())}
        Content:
        {chr(10).join([f'{name}: {content}' for name, content in pdf_summaries.items()])}
        """
        response = llm.predict(prompt)
        comparison["Aspect"].append(aspect)
        for filename in pdf_data.keys():
            # Extract the description for this PDF from the response
            desc_start = response.find(f"- [{filename}]:") + len(f"- [{filename}]:")
            desc_end = response.find("- [", desc_start) if "- [" in response[desc_start:] else len(response)
            description = response[desc_start:desc_end].strip()
            comparison[filename].append(description)
    
    # Create a DataFrame and convert to HTML
    df = pd.DataFrame(comparison)
    state["comparison_table"] = df.to_html(index=False, classes="table table-striped", escape=False)
    state["response"] = "Comparison table generated below."
    return state

# Build the LangGraph workflow
def build_graph():
    workflow = StateGraph(ChatbotState)
    
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("build_vector_store", build_vector_store_node)
    workflow.add_node("initialize_chatbot", initialize_chatbot_node)
    workflow.add_node("handle_query", handle_query_node)
    workflow.add_node("compare_pdfs", compare_pdfs_node)
    
    workflow.set_entry_point("extract_text")
    workflow.add_edge("extract_text", "build_vector_store")
    workflow.add_edge("build_vector_store", "initialize_chatbot")
    workflow.add_edge("initialize_chatbot", "handle_query")
    workflow.add_conditional_edges(
        "handle_query",
        lambda state: "compare_pdfs" if "compare" in state["user_query"].lower() and "pdf" in state["user_query"].lower() else END,
        {"compare_pdfs": "compare_pdfs", END: END}
    )
    workflow.add_edge("compare_pdfs", END)
    
    return workflow.compile()

# Streamlit UI and integration with LangGraph
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    
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
            "response": "",
            "comparison_table": ""
        }
    
    st.header("Chat with multiple PDFs 📚")
    st.write("Upload PDFs and interact with them! Type 'compare the PDFs' to see a verbal comparison table.")
    
    st.subheader("Chat History")
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chatbot_state["chat_history"]):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    if st.session_state.chatbot_state["comparison_table"]:
        st.subheader("PDF Comparison Table")
        st.write(st.session_state.chatbot_state["comparison_table"], unsafe_allow_html=True)
    
    user_question = st.text_input("Ask a question", key="user_input")
    
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Upload") and pdf_files:
            with st.spinner("Processing PDFs..."):
                initial_state = st.session_state.chatbot_state.copy()
                initial_state["pdf_files"] = pdf_files
                
                result = st.session_state.graph.invoke(initial_state)
                st.session_state.chatbot_state = result
                
                sorted_pdfs = sorted(((pdf, datetime.now()) for pdf in pdf_files), key=lambda x: x[1], reverse=True)
                st.write("Uploaded PDFs (Most Recent First):")
                for pdf, timestamp in sorted_pdfs:
                    st.write(f"📄 {pdf.name} (Uploaded at: {timestamp})")
                most_recent_pdf = sorted_pdfs[-1][0].name
                st.write(f"🆕 Most Recently Uploaded PDF: **{most_recent_pdf}**")
                st.success("PDFs processed successfully!")
    
    if user_question:
        with st.spinner("Generating response..."):
            query_state = st.session_state.chatbot_state.copy()
            query_state["user_query"] = user_question
            query_state["response"] = ""
            
            result = st.session_state.graph.invoke(query_state)
            st.session_state.chatbot_state = result

if __name__ == "__main__":
    main()
