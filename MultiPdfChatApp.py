from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_groq.chat_models import ChatGroq
from PyPDF2 import PdfReader
import streamlit as st
import tiktoken
import torch
import hashlib

class MultiPDFChatApp:
    def __init__(self, project_name: str, pdf_docs: list = []):
        torch.classes.__path__ = []
        self.project_name = project_name
        self.pdf_docs = pdf_docs
        self.text_chunks = []
        self.vectorstore = None
        self.chunk_hashes = set()

        self.llm = ChatGroq(
            api_key=st.secrets["GROQ_API_KEY"],
            model_name=st.secrets.get("GROQ_MODEL", "mixtral-8x7b-32768"),
            temperature=0.4,
            max_tokens=1000
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def hash_text(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_pdf_text(self, pdfs):
        raw_texts = []
        for pdf in pdfs:
            pdf.seek(0)
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    raw_texts.append(text)
        return raw_texts

    def get_text_chunks(self, texts):
        encoding = tiktoken.get_encoding("cl100k_base")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda t: len(encoding.encode(t)),
            separators=["\n\n", "\n", " ", ""]
        )
        all_chunks = []
        for text in texts:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                chunk_hash = self.hash_text(chunk)
                if chunk_hash not in self.chunk_hashes:
                    self.chunk_hashes.add(chunk_hash)
                    all_chunks.append(chunk)
        return all_chunks

    def build_vectorstore(self, chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings)

    def run_chat(self):
        try:
            texts = self.get_pdf_text(self.pdf_docs)
            self.text_chunks = self.get_text_chunks(texts)
            if self.text_chunks:
                self.vectorstore = self.build_vectorstore(self.text_chunks)
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def get_conversation_chain(self, question: str):
        if not question.strip():
            return "Please ask a valid question."
        try:
            if self.vectorstore:
                chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
                    memory=self.memory,
                    return_source_documents=False
                )
                response = chain.invoke({'question': question})
                return response.get("answer", "Sorry, no answer could be generated.")
            else:
                return self.llm.invoke(question).content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                return "🚫 Groq API rate limit reached. Please wait a few minutes or upgrade your plan."
            return f"💥 Error during response: {e}"
