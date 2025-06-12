import streamlit as st
import torch
from MultiPdfChatApp import MultiPDFChatApp
import os

# Ensure torch path is clear
torch.classes.__path__ = []

# Check API key configuration
try:
    # Check if API keys are properly set
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your-groq-api-key":
        st.error("⚠️ Please set your actual GROQ_API_KEY in Streamlit Cloud secrets!")
        st.info("Go to your Streamlit Cloud app settings → Secrets and add your API keys")
        st.stop()
    
    # Debug info in sidebar
    st.sidebar.success("✅ API Key configured")
    st.sidebar.info(f"Model: {st.secrets.get('GROQ_MODEL', 'Not set')}")
    
except Exception as e:
    st.error("❌ API keys not configured properly in Streamlit Cloud secrets")
    st.error(f"Error: {str(e)}")
    st.stop()

# Initialize session state for chat history if not already exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for chat_app
if 'chat_app' not in st.session_state:
    st.session_state.chat_app = None

# Initialize session state for chat initialized flag
if 'chat_initialized' not in st.session_state:
    st.session_state.chat_initialized = False

# App header
st.header('Multi-PDF Chat App')

# Sidebar for file upload
with st.sidebar:
    st.write("Upload PDFs")
    uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type="pdf", key="pdf")
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.size/1024:.1f} KB)")
    else:
        st.warning("No files uploaded.")

# Project name input
project_name = None
if uploaded_files:
    st.write("Your Project Name")
    project_name = st.text_input("Project Name", placeholder="Enter project name...")

# Initialize chat app when button is clicked
if project_name and uploaded_files and not st.session_state.chat_initialized:
    if st.button("Start Chat"):
        with st.spinner("Initializing App..."):
            try:
                # Create and store chat app in session state
                st.session_state.chat_app = MultiPDFChatApp(project_name, uploaded_files)
                status = st.session_state.chat_app.run_chat()
                
                if status is True:
                    st.session_state.chat_initialized = True
                    st.success("🎉 Chat initialized successfully! You can now ask questions about your PDFs.")
                    st.rerun()
                else:
                    st.error("❌ Chat initialization failed. Please check the uploaded files.")
            except Exception as e:
                st.error(f"An error occurred during initialization: {str(e)}")

# Show initialization status
if st.session_state.chat_initialized:
    st.success("Chat is active - Ask questions about your PDFs below!")

# Add some quick action buttons if chat is initialized
if st.session_state.chat_initialized:
    st.write("Quick Actions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Summarize"):
            # Auto-add summarize question
            if st.session_state.chat_app:
                user_input = "Please provide a comprehensive summary of all the uploaded documents."
                st.session_state.chat_history.append(("user", user_input))
                
                try:
                    with st.spinner("Generating summary..."):
                        response = st.session_state.chat_app.get_conversation_chain(user_input)
                        if response and not response.startswith("Error:"):
                            st.session_state.chat_history.append(("assistant", response))
                        else:
                            st.error(f"Failed to generate summary: {response}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    with col2:
        if st.button("🔍 Key Points"):
            if st.session_state.chat_app:
                user_input = "What are the main key points and important findings from these documents?"
                st.session_state.chat_history.append(("user", user_input))
                
                try:
                    with st.spinner("Extracting key points..."):
                        response = st.session_state.chat_app.get_conversation_chain(user_input)
                        if response and not response.startswith("Error:"):
                            st.session_state.chat_history.append(("assistant", response))
                        else:
                            st.error(f"Failed to extract key points: {response}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error extracting key points: {str(e)}")
    
    with col3:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.chat_app = None
            st.session_state.chat_initialized = False
            st.rerun()

# Display Chat History
for chat in st.session_state.chat_history:
    role, message = chat
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)

# User Input Field
if st.session_state.chat_app is not None and st.session_state.chat_initialized:
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Append user message to chat history
        st.session_state.chat_history.append(("user", user_input))
        
        # Get AI response
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.chat_app.get_conversation_chain(user_input)
                
                if response and not response.startswith("Error:"):
                    # Append AI response to chat history
                    st.session_state.chat_history.append(("assistant", response))
                else:
                    st.error(f"Failed to get response: {response or 'Unknown error'}")
                    # Still add the error to chat history for context
                    st.session_state.chat_history.append(("assistant", f"Sorry, I encountered an error: {response or 'Unknown error'}"))
                
                # Rerun the app to refresh the interface
                st.rerun()
        except Exception as e:
            error_msg = f"Error in getting response: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append(("assistant", f"Sorry, I encountered an error: {str(e)}"))
            st.rerun()

elif project_name and uploaded_files:
    st.info("👆 Click 'Start Chat' above to begin chatting with your PDFs")
else:
    st.info("👈 Please upload PDF files and enter a project name to get started")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Upload multiple PDFs to ask questions across all documents!")
