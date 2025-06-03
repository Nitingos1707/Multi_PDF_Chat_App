# MultiPDFChatApp

MultiPDFChatApp is a Streamlit-based web application that allows users to upload and chat with multiple PDF documents using AI. The app leverages PyTorch and custom logic to enable interactive conversations with the content of uploaded PDFs.

## Features
- Upload and manage multiple PDF files
- Assign a project name to your PDF collection
- Chat with the content of your PDFs using an AI assistant
- Maintains chat history for each session
- User-friendly Streamlit interface

## How It Works
1. **Upload PDFs:** Use the sidebar to upload one or more PDF files.
2. **Project Name:** Enter a project name to organize your session.
3. **Start Chat:** Click the "Start Chat" button to initialize the AI chat with your PDFs.
4. **Chat:** Type your questions in the chat input. The AI will respond based on the content of your uploaded PDFs.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Nitingos1707/MultiPDFChatApp.git
   cd MultiPDFChatApp
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the Streamlit app:
```sh
streamlit run app.py
```

Open the provided local URL in your browser to use the app.

## Requirements
- Python 3.8+
- streamlit
- torch
- (Other dependencies listed in `requirements.txt`)

## Project Structure
- `app.py` - Main Streamlit application
- `MultiPdfChatApp.py` - Core logic for PDF chat functionality
- `requirements.txt` - Python dependencies
- `Storage/` - Directory for storing PDF embeddings and session data

## License
This project is licensed under the MIT License.

## Acknowledgements
- Built with [Streamlit](https://streamlit.io/)
- Powered by [PyTorch](https://pytorch.org/)