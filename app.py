import streamlit as st
import torch
from MultiPdfChatApp import MultiPDFChatApp
import os

# Ensure torch path is clear
torch.classes.__path__ = []

# Load secrets from Streamlit Cloud
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
GROQ_MODEL = st.secrets.get("GROQ_MODEL", None)
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", None)
DEEPSEEK_MODEL = st.secrets.get("DEEPSEEK_MODEL", None)

# Optionally set as environment variables for compatibility
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if GROQ_MODEL:
    os.environ["GROQ_MODEL"] = GROQ_MODEL
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
if DEEPSEEK_MODEL:
    os.environ["DEEPSEEK_MODEL"] = DEEPSEEK_MODEL

# Initialize session state for chat history if not already exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for chat_app
if 'chat_app' not in st.session_state:
    st.session_state.chat_app = None

# App header
st.header('Multi-PDF Chat App')

# Sidebar for file upload
with st.sidebar:
    st.write("Upload PDFs")
    uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type="pdf", key="pdf")
    
    if uploaded_files:
        st.info("File Uploaded Successfully")
    else:
        st.warning("No files uploaded.")

# Project name input
project_name = None
if uploaded_files:
    st.write("Your Project Name")
    project_name = st.text_input("Project Name")

# Initialize chat app when button is clicked
if project_name and uploaded_files:
    if st.button("Start Chat"):
        with st.spinner("Initializing App..."):
            try:
                # Create and store chat app in session state
                st.session_state.chat_app = MultiPDFChatApp(project_name, uploaded_files)
                status = st.session_state.chat_app.run_chat()
                
                if status is True:
                    st.success("Now you can chat with the AI!")
                else:
                    st.error("Chat initialization failed. Please check the uploaded files.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display Chat History
for chat in st.session_state.chat_history:
    role, message = chat
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)

# User Input Field
if st.session_state.chat_app is not None:
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Append user message to chat history
        st.session_state.chat_history.append(("user", user_input))
        
        # Get AI response
        try:
            response = st.session_state.chat_app.get_conversation_chain(user_input)
            
            # Append AI response to chat history
            st.session_state.chat_history.append(("assistant", response))
            
            # Rerun the app to refresh the interface
            st.rerun()
        except Exception as e:
            st.error(f"Error in getting response: {str(e)}")
