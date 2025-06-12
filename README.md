
# 📚 Multi PDF Chat App

**Multi PDF Chat App** is a powerful Streamlit-based web app that lets you upload multiple PDFs and **chat with an AI** about their content in real-time. It combines the strength of **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq's blazing-fast LLaMA models** for fast, smart, document-aware conversations.

<p align="center">
  <a href="https://multipdfchatapp.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Open-App-green?style=for-the-badge&logo=streamlit" alt="Launch App">
  </a>
</p>

---

## 🚀 Features

- ✅ Upload and process **multiple PDFs**
- 🧠 Ask questions about the contents across all documents
- 🗃️ Uses **FAISS** for fast vector-based search
- 🔗 Powered by **Groq LLaMA 3.3 70B** via LangChain
- ♻️ Seamless conversational memory
- 💬 Keep chatting even without uploading PDFs

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Nitingos1707/Multi_PDF_Chat_App.git
cd Multi_PDF_Chat_App
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Secrets
Create a `.env` file with your API keys:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

> 💡 **Note:** Never commit `.env` or sensitive keys to GitHub.

---

## ▶️ Run the App Locally
```bash
streamlit run app.py
```

Then visit `http://localhost:8501` in your browser.

---

## 💡 How to Use

1. Visit the app: [multipdfchatapp.streamlit.app](https://multipdfchatapp.streamlit.app/)
2. Upload one or more PDFs using the sidebar.
3. Start chatting right away — no project name needed!
4. You can also chat without uploading PDFs — it's always ready.

---

## 🧱 Project Structure

```
Multi_PDF_Chat_App/
├── app.py                   # Streamlit interface
├── MultiPdfChatApp.py       # PDF processing and chat logic
├── requirements.txt         # Dependencies
└── Storage/                 # Local FAISS vectorstore
```

---

## 📄 License

This project is open for **educational and research purposes** only. Not intended for commercial use.

---

> Built with ❤️ by [@Nitingos1707](https://github.com/Nitingos1707)
