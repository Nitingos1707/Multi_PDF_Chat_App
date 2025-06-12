# Multi_PDF_Chat_App

Multi_PDF_Chat_App is a Streamlit-based web application that allows users to upload multiple PDF documents and interactively chat with an AI about their contents. The app leverages LangChain, HuggingFace embeddings, ChromaDB, and Groq LLMs for document processing and conversational retrieval.

## Features
- Upload and process multiple PDF files
- Extract and chunk text from PDFs
- Store and retrieve document embeddings using ChromaDB
- Conversational AI interface powered by Groq LLM
- Persistent chat history within the session

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone https://github.com/Nitingos1707/Multi_PDF_Chat_App.git
cd Multi_PDF_Chat_App
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with your API keys and tokens:
```
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```
**Note:** Never commit your `.env` file to version control.

### 5. Run the Application
```powershell
streamlit run app.py
```

## Usage
1. Open the app in your browser (Streamlit will provide a local URL).
2. Upload one or more PDF files using the sidebar.
3. Enter a project name and click "Start Chat".
4. Ask questions about the uploaded PDFs in the chat interface.

## Project Structure
- `app.py` - Streamlit UI and main app logic
- `MultiPdfChatApp.py` - Core class for PDF processing and chat
- `requirements.txt` - Python dependencies
- `Storage/` - Directory for ChromaDB vector storage

## License
This project is for educational and research purposes only.