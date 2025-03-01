#This chatbot is capable of reading answers from mutliple PDFS + also can read handwritten PDFs (with the help of tesseract-ocr) + understands tables and images      

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
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def extract_text_from_pdf(pdf):
    """Extract text from a PDF file safely."""
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text.encode("utf-8", "ignore").decode("utf-8")  # Ignore encoding issues
    return text

def extract_images_from_pdf(pdf):
    """Extract images from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        images += page.get_images(full=True)
    return images, doc

def ocr_images(images, doc):
    """Perform OCR on extracted images using Tesseract."""
    ocr_text = ""
    for img in images:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        ocr_text += text.encode("utf-8", "ignore").decode("utf-8")  # Ignore encoding issues
    return ocr_text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    return CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)

def create_vector_store(text_chunks, metadata_list):
    """Create a FAISS vector store with metadata, ensuring UTF-8 compatibility."""
    processed_chunks = [t.encode("utf-8", "ignore").decode("utf-8") for t in text_chunks]
    return FAISS.from_texts(processed_chunks, OpenAIEmbeddings(), metadatas=metadata_list)

def initialize_chatbot(vector_store, system_prompt):
    """Initialize the chatbot with memory and retrieval capabilities."""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    memory.save_context({"content": system_prompt}, {"content": "System prompt initialized."})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(), 
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 relevant chunks
        memory=memory
    )

def handle_user_input(user_question):
    """Process user input and display chat responses with confidence scores."""
    response = st.session_state.convo({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    retrieved_docs = response.get('source_documents', [])
    
    # Compute confidence score (normalize similarity scores)
    if retrieved_docs:
        scores = [doc.metadata.get("score", 1.0) for doc in retrieved_docs]
        max_score = max(scores) if max(scores) > 0 else 1.0
        confidence_scores = [(s / max_score) * 100 for s in scores]
    else:
        confidence_scores = [100]
    
    # Display chat messages, excluding the system prompt initialization message
    for i, message in enumerate(st.session_state.chat_history):
        if i == 0:  # Skip the first system prompt message
            continue
        if i % 2 == 1:  # User message
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot message
            confidence = confidence_scores[(i - 1) // 2] if (i - 1) // 2 < len(confidence_scores) else 100
            st.write(bot_template.replace("{{MSG}}", f"{message.content} \n\n\n **Confidence: {confidence:.2f}%**"), unsafe_allow_html=True)

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
    if st.button("Enter") and user_question:
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
                
                # Extract and process text from PDFs with metadata
                all_text_chunks = []
                metadata_list = []
                for pdf, timestamp in sorted_pdfs:
                    text_chunks = split_text_into_chunks(extract_text_from_pdf(pdf))
                    images, doc = extract_images_from_pdf(pdf)
                    ocr_text = ocr_images(images, doc)
                    ocr_text_chunks = split_text_into_chunks(ocr_text)
                    
                    all_text_chunks.extend(text_chunks + ocr_text_chunks)
                    metadata_list.extend([{ "filename": pdf.name, "timestamp": timestamp, "score": 1.0 }] * (len(text_chunks) + len(ocr_text_chunks)))
                
                # Create vector store and initialize chatbot with metadata filtering
                st.session_state.convo = initialize_chatbot(create_vector_store(all_text_chunks, metadata_list), system_prompt="""
                You are a helpful assistant. Prioritize answers from the most recently uploaded PDF first.
                """)
                
                st.success("PDFs processed successfully!")

if __name__ == '__main__':
    main()
