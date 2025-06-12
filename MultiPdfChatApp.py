from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_groq.chat_models import ChatGroq
from PyPDF2 import PdfReader
import streamlit as st
import tiktoken
import torch
import os

class MultiPDFChatApp:
    def __init__(self, project_name: str, pdf_docs: list = []):
        torch.classes.__path__ = []
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Storage")
        os.makedirs(base_dir, exist_ok=True)

        # Removed chroma_persist_dir since we are not persisting on Streamlit Cloud
        self.pdf_docs = pdf_docs
        self.project_name = project_name
        self.raw_text = ""
        self.text_chunks = None
        self.vectorstore = None

    def get_pdf_text(self):
        self.raw_text = ""
        for pdf in self.pdf_docs:
            pdf.seek(0)
            reader = PdfReader(pdf)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    self.raw_text += f"\n--- {pdf.name} (Page {page_num + 1}) ---\n{text}\n"
        if not self.raw_text.strip():
            raise ValueError("No extractable text found in PDFs.")
        return self.raw_text

    def get_text_chunks(self):
        if not self.raw_text:
            raise ValueError("No text available for chunking.")
        encoding = tiktoken.get_encoding("cl100k_base")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda text: len(encoding.encode(text)),
            separators=["\n\n", "\n", " ", ""]
        )
        self.text_chunks = splitter.split_text(self.raw_text)
        if not self.text_chunks:
            raise ValueError("Text chunking failed.")
        return self.text_chunks

    def get_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Use in-memory vectorstore (no persistence for Streamlit Cloud)
        self.vectorstore = Chroma.from_texts(
            texts=self.text_chunks,
            embedding=embeddings
        )

        return self.vectorstore

    def get_conversation_chain(self, question: str):
        if not question.strip():
            return "Please ask a valid question."
        if not self.vectorstore:
            return "Error: Vectorstore not ready."
        try:
            llm = ChatGroq(
                api_key=st.secrets["GROQ_API_KEY"],
                model_name=st.secrets.get("GROQ_MODEL", "mixtral-8x7b-32768"),
                temperature=0.4,
                max_tokens=1000
            )
        except Exception as e:
            return f"Error loading model: {e}"

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=memory,
            return_source_documents=False
        )

        response = chain.invoke({'question': question})
        return response.get("answer", "Sorry, no answer could be generated.")

    def run_chat(self):
        try:
            self.get_pdf_text()
            self.get_text_chunks()
            self.get_vectorstore()
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False
