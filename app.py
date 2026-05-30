import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Environment Variables from .env file
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract raw text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Divide text into small chunks with overlap to retain context."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Convert chunks to embeddings and store in a local FAISS DB."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

def get_conversation_chain(vectorstore):
    """Create the LangChain pipeline linking LLM, Vector Store, and Memory."""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY is missing! Check your .env file setup.")
        st.stop()

    # FIX: Explicitly add 'https://' to make httpx happy
    client_options = {
        "api_endpoint": "https://generativelanguage.googleapis.com"
    }

    # Pass the corrected client options dictionary
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=api_key,
        client_options=client_options
    )
    
    # Configure legacy memory correctly for ConversationalRetrievalChain
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

def handle_userinput(user_question):
    """Process user question, fetch answer, and update chat UI."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    # Display the conversation
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"🧑 **User:** {message.content}")
        else:
            st.write(f"🤖 **Bot:** {message.content}")
            
            # Citation Highlighting from legacy source docs
            if 'source_documents' in response and i == len(st.session_state.chat_history) - 1:
                with st.expander("📚 View Document Sources Used"):
                    for doc in response['source_documents']:
                        st.markdown(f"*{doc.page_content[:200]}...*")

def main():
    st.set_page_config(page_title="Multi-PDF Chatbot", page_icon="📚")
    st.header("Chat with Multiple PDFs 🤖")

    # Initialize session state variables so data persists across re-renders
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process your PDFs first via the sidebar!")

    # Sidebar layout for file uploads
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True, 
            type=["pdf"]
        )
        if st.button("Process"):
            with st.spinner("Processing documents..."):
                # 1. Get PDF Text
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. Get Text Chunks
                text_chunks = get_text_chunks(raw_text)
                
                # 3. Create Vector Store
                vectorstore = get_vectorstore(text_chunks)
                
                # 4. Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                st.success("Done! Your chatbot is ready to answer questions.")

if __name__ == '__main__':
    main()