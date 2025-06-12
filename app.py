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

# Sidebar PDF uploader
with st.sidebar:
    st.write("📁 Upload your PDF files")
    uploaded_files = st.file_uploader("Choose PDFs", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")

        if not st.session_state.chat_initialized:
            with st.spinner("Initializing with uploaded PDFs..."):
                try:
                    st.session_state.chat_app = MultiPDFChatApp("default", uploaded_files)
                    if st.session_state.chat_app.run_chat():
                        st.session_state.chat_initialized = True
                        st.toast("✅ PDFs processed!")
                    else:
                        st.error("❌ Failed to process PDFs.")
                except Exception as e:
                    st.error("💥 Initialization failed")
                    with st.expander("Error Details"):
                        st.code(str(e))

# Show chat status
if uploaded_files:
    st.success("📄 You can now ask questions about your uploaded PDFs!")
else:
    st.info("💬 No PDFs uploaded — you're chatting with a general AI assistant.")

# Display chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Chat input (works with or without PDFs)
user_input = st.chat_input("Ask a question...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_app.get_conversation_chain(user_input)
            except Exception as e:
                response = f"💥 Error during response: {e}"
        st.write(response)

    st.session_state.chat_history.append(("assistant", response))

# Reset chat
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.chat_app = MultiPDFChatApp("default")
    st.session_state.chat_initialized = False
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 Tip: You can chat with or without uploading PDFs.")
