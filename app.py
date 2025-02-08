import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfFileReader, PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:                                            #going thro 1pdf at a time
        pdf_reader = PdfReader(pdf)                                 #one pdf object for each pdf 
        for page in pdf_reader.pages:                               #getting the number of pages in the pdf
            text += page.extract_text()                             #extracting the text from the pdf
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
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
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return convo_chain

def handle_user_input(user_question):
    response = st.session_state.convo({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

def main():
    #API keys are stored in a .env file
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "convo" not in st.session_state:
        st.session_state.convo = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    st.write("Upload multiple PDFs and chat with them!")
    user_question = st.text_input("Ask a question")
    if user_question:
        handle_user_input(user_question)

    #a sidebar where user can upload multiple PDFs
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
        if st.button("Upload"): 
            with st.spinner("Processing PDFs..."):
                #get pdf text 
                raw_text = get_pdf_text(pdf_docs) 

                #get the text chunks 
                text_chunks = get_text_chunks(raw_text)

                #create the vector store 
                vector_store = get_vectorstore(text_chunks)

                st.success("PDFs processed successfully!")

                #chatbot code
                st.session_state.convo = get_convo(vector_store)                #this way, it doesn't re-initialise the variable 


if __name__ == '__main__':
    main()