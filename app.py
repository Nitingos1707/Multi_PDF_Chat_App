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
    st.session_state.chat_app = None

if 'chat_initialized' not in st.session_state:
    st.session_state.chat_initialized = False

# Header
st.header('📚 Multi-PDF Chat App')

# Sidebar PDF uploader
with st.sidebar:
    st.write("📁 Upload your PDF files")
    uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size/1024:.1f} KB)")
    else:
        st.warning("No files uploaded.")

# Project name input
project_name = st.text_input("Project Name", placeholder="Enter project name...") if uploaded_files else None

# Start chat button
if uploaded_files and project_name and not st.session_state.chat_initialized:
    if st.button("🚀 Start Chat"):
        with st.spinner("Initializing chat application..."):
            try:
                st.session_state.chat_app = MultiPDFChatApp(project_name, uploaded_files)

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔍 Processing PDFs...")
                progress_bar.progress(25)

                success = st.session_state.chat_app.run_chat()

                if success:
                    progress_bar.progress(100)
                    status_text.text("✅ Chat initialized successfully!")
                    st.session_state.chat_initialized = True
                    st.rerun()
                else:
                    st.error("❌ Chat initialization failed.")
                    with st.expander("Debug Information"):
                        st.write("Check for image-based PDFs or corrupted files.")
            except Exception as e:
                st.error("💥 Error during initialization")
                with st.expander("Error Details"):
                    st.code(str(e))

# Active chat
if st.session_state.chat_initialized:
    st.success("Chat is active. Ask questions about your PDFs!")

# Display history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_app.get_conversation_chain(user_input)
            st.write(response)

    st.session_state.chat_history.append(("assistant", response))

# Reset chat
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.chat_app = None
    st.session_state.chat_initialized = False
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 Tip: Upload multiple PDFs and click 'Start Chat' to analyze them together.")
