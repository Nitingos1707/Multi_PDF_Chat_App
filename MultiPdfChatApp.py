from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini model
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

        # ✅ Gemini LLM from Google AI
        self.llm = ChatGoogleGenerativeAI(
            model=st.secrets.get("GEMINI_MODEL", "models/gemini-pro"),
            google_api_key=st.secrets["GEMINI_API_KEY"],
            temperature=0.4,
            max_output_tokens=1000
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

    def add_new_pdfs(self, new_pdfs: list):
        try:
            self.pdf_docs += new_pdfs
            new_texts = self.get_pdf_text(new_pdfs)
            new_chunks = self.get_text_chunks(new_texts)

            if not new_chunks:
                print("⚠️ No new chunks found in uploaded PDFs.")
                return

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            if not self.vectorstore:
                self.vectorstore = FAISS.from_texts(new_chunks, embedding=embeddings)
            else:
                self.vectorstore.add_texts(new_chunks, embedding=embeddings)

            print(f"✅ Added {len(new_chunks)} new chunks.")
        except Exception as e:
            print(f"❌ Failed to add new PDFs: {e}")

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
                return self.llm.invoke(question)
        except Exception as e:
            return f"Error during response generation: {str(e)}"
