import streamlit as st
import torch
from MultiPdfChatApp import MultiPDFChatApp
import os

# Clear TorchScript path
torch.classes.__path__ = []

# Check and display API key configuration
try:
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your-groq-api-key":
        st.error("⚠️ Please set your actual GROQ_API_KEY in Streamlit Cloud secrets!")
        st.info("Go to Streamlit Cloud → Settings → Secrets to add your API key.")
        st.stop()
    st.sidebar.success("✅ API Key configured")
    st.sidebar.info(f"Model: {st.secrets.get('GROQ_MODEL', 'Not set')}")
except Exception as e:
    st.error("❌ API keys not configured properly.")
    st.error(f"Error: {str(e)}")
    st.stop()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_app' not in st.session_state:
    st.session_state.chat_app = MultiPDFChatApp(project_name="default")

if 'chat_initialized' not in st.session_state:
    st.session_state.chat_initialized = False

# Header
st.header('📚 Multi-PDF Chat App')

# Sidebar PDF uploader (dynamic during chat)
with st.sidebar:
    st.write("📁 Upload or add PDF files")
    new_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type="pdf")

    if new_files:
        if st.session_state.chat_initialized:
            st.session_state.chat_app.add_new_pdfs(new_files)
            st.success("✅ New PDFs added to the conversation!")
        else:
            st.session_state.chat_app = MultiPDFChatApp("session", new_files)
            with st.spinner("Initializing chat..."):
                success = st.session_state.chat_app.run_chat()
                if success:
                    st.session_state.chat_initialized = True
                    st.success("✅ Chat initialized with uploaded PDFs!")
                else:
                    st.error("❌ Initialization failed.")

# Chat interface
if st.session_state.chat_initialized:
    st.success("Chat is active. Ask questions about your PDFs!")
else:
    st.info("📢 No PDFs uploaded. You can still chat generally!")

# Display history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# User input
user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = st.session_state.chat_app.get_conversation_chain(user_input)
        st.session_state.chat_history.append(("assistant", response))
    st.rerun()

# Reset
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.chat_app = MultiPDFChatApp(project_name="default")
    st.session_state.chat_initialized = False
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 Tip: You can upload more PDFs any time during the chat.")
