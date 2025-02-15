#MULTI-PDFS BASED CHATBOT + PDFS SORTED ACC TO UPLOAD TIME 
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
import os
from datetime import datetime

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks 

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_convo(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return convo_chain

def handle_user_input(user_question):
    response = st.session_state.convo({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    # API keys are stored in a .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "convo" not in st.session_state:
        st.session_state.convo = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
    
    if "sorted_pdfs" not in st.session_state:
        st.session_state.sorted_pdfs = []

    st.header("Chat with multiple PDFs :books:")
    st.write("Upload multiple PDFs and chat with them!")
    user_question = st.text_input("Ask a question")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing PDFs..."):
                uploaded_pdfs = {}

                if pdf_docs:
                    for pdf in pdf_docs:
                        # Assign timestamp to each uploaded file
                        timestamp = datetime.now()  # Store timestamp in ISO format
        
                        # Save file and metadata in dictionary
                        uploaded_pdfs[pdf.name] = {
                            "file": pdf,
                            "timestamp": timestamp
                        }

                    # Sort uploaded PDFs by timestamp (most recent first)
                    original_pdfs = sorted(uploaded_pdfs.items(), key=lambda x: x[1]["timestamp"])
                    sorted_pdfs = list(reversed(original_pdfs))  # Convert list_reverseiterator to list
    
                    # Add sorted_pdfs to session state
                    st.session_state.sorted_pdfs = sorted_pdfs
    
                    # Display uploaded files
                    st.write("Uploaded PDFs (Most Recent First):")
                    for pdf_name, data in sorted_pdfs:
                        st.write(f"📄 {pdf_name} (Uploaded at: {data['timestamp']})")
    
                    # Display the most recent PDF
                    most_recent_pdf = sorted_pdfs[0][0]  # First entry in sorted list (most recent)
                    st.write(f"🆕 Most Recently Uploaded PDF: **{most_recent_pdf}**")

                    # Process and create vector store for all PDFs
                    all_text_chunks = []
                    for pdf_name, data in st.session_state.sorted_pdfs:
                        raw_text = get_pdf_text(data["file"])
                        text_chunks = get_text_chunks(raw_text)
                        all_text_chunks.extend(text_chunks)

                    combined_vector_store = get_vectorstore(all_text_chunks)

                    st.success("PDFs processed successfully!")
                    st.session_state.convo = get_convo(combined_vector_store)

if __name__ == '__main__':
    main()
