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

# Sidebar PDF uploader (dynamic)
with st.sidebar:
    st.write("📁 Upload or add PDF files")
    new_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type="pdf")

    if new_files:
        with st.spinner("Processing new PDFs..."):
            for i, pdf in enumerate(new_files):
                st.session_state.chat_app.add_new_pdfs([pdf])
                st.progress((i + 1) / len(new_files))
            st.success("✅ PDFs added!")

# Show full chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Display user message immediately
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # Assistant responds
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_app.get_conversation_chain(user_input)
            st.write(response)
    st.session_state.chat_history.append(("assistant", response))

# Reset option
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.chat_app = MultiPDFChatApp(project_name="default")
    st.session_state.chat_initialized = False
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 Tip: You can upload more PDFs anytime during the chat.")
