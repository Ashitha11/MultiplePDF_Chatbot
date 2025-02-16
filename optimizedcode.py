import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
import os
from datetime import datetime

def extract_text_from_pdf(pdf):  
    """Extract text from a PDF file."""
    return "".join(page.extract_text() for page in PdfReader(pdf).pages if page.extract_text())

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    return CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)

def create_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks."""
    return FAISS.from_texts(text_chunks, OpenAIEmbeddings())

def initialize_chatbot(vector_store, system_prompt):
    """Initialize the chatbot with memory and retrieval capabilities."""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    memory.save_context({"content": system_prompt}, {"content": "System prompt initialized."})
    return ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vector_store.as_retriever(), memory=memory)

def handle_user_input(user_question):
    """Process user input and display chat responses."""
    response = st.session_state.convo({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session state variables
    for key in ["convo", "chat_history", "sorted_pdfs"]:
        if key not in st.session_state:
            st.session_state[key] = None if key == "convo" else []
    
    st.header("Chat with multiple PDFs 📚")
    st.write("Upload PDFs and interact with them!")
    
    user_question = st.text_input("Ask a question")
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Upload") and pdf_files:
            with st.spinner("Processing PDFs..."):
                # Assign timestamps and sort PDFs by upload time (most recent first)
                sorted_pdfs = sorted(((pdf, datetime.now()) for pdf in pdf_files), key=lambda x: x[1], reverse=True)
                st.session_state.sorted_pdfs = sorted_pdfs
                
                st.write("Uploaded PDFs (Most Recent First):")
                for pdf, timestamp in sorted_pdfs:
                    st.write(f"📄 {pdf.name} (Uploaded at: {timestamp})")
                
                most_recent_pdf = sorted_pdfs[-1][0].name
                st.write(f"🆕 Most Recently Uploaded PDF: **{most_recent_pdf}**")
                
                # Extract and process text from PDFs
                all_text_chunks = [chunk for pdf, _ in sorted_pdfs for chunk in split_text_into_chunks(extract_text_from_pdf(pdf))]
                
                # Create vector store and initialize chatbot
                st.session_state.convo = initialize_chatbot(create_vector_store(all_text_chunks), system_prompt="""
                You are a helpful assistant. Prioritize answers from the most recently uploaded PDF first.
                """)
                
                st.success("PDFs processed successfully!")

if __name__ == '__main__':
    main()
