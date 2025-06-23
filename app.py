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

if "upload_queue" not in st.session_state:
    st.session_state.upload_queue = []

# Header
st.header('📚 Multi-PDF Chat App')

# Sidebar uploader (just queues files)
with st.sidebar:
    st.write("📁 Upload PDFs (chat will remain active)")
    new_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type="pdf")
    if new_files:
        for f in new_files:
            st.session_state.upload_queue.append(f)
        st.success(f"📥 Queued {len(new_files)} file(s) for processing.")

# Background PDF processing (1 per run)
# Process all queued files at once
if st.session_state.upload_queue:
    with st.spinner("🔄 Processing uploaded PDFs..."):
        while st.session_state.upload_queue:
            pdf = st.session_state.upload_queue.pop(0)
            st.session_state.chat_app.add_new_pdfs([pdf])
            st.toast(f"✅ {pdf.name} processed!")


# Display chat history
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

# Reset button
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.chat_app = MultiPDFChatApp(project_name="default")
    st.session_state.chat_initialized = False
    st.session_state.upload_queue = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 Tip: You can keep chatting while PDFs are being processed in the background.")
