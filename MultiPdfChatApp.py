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
import os

class MultiPDFChatApp:
    def __init__(self, project_name: str, pdf_docs: list = []):
        torch.classes.__path__ = []
        self.project_name = project_name
        self.pdf_docs = pdf_docs
        self.raw_text = ""
        self.text_chunks = []
        self.vectorstore = None

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

    def get_pdf_text(self, pdfs):
        raw_text = ""
        for pdf in pdfs:
            pdf.seek(0)
            reader = PdfReader(pdf)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    raw_text += f"\n--- {pdf.name} (Page {page_num + 1}) ---\n{text}\n"
        if not raw_text.strip():
            raise ValueError("No extractable text found in PDFs.")
        return raw_text

    def get_text_chunks(self, text):
        encoding = tiktoken.get_encoding("cl100k_base")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda t: len(encoding.encode(t)),
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def build_vectorstore(self, chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings)

    def run_chat(self):
        try:
            print("🔹 Extracting text...")
            self.raw_text = self.get_pdf_text(self.pdf_docs)
            print("✅ Text length:", len(self.raw_text))

            print("🔹 Splitting into chunks...")
            self.text_chunks = self.get_text_chunks(self.raw_text)
            print("✅ Chunks:", len(self.text_chunks))

            print("🔹 Creating FAISS vectorstore...")
            self.vectorstore = self.build_vectorstore(self.text_chunks)
            print("✅ Vectorstore ready.")
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
                # No PDFs: fallback to general LLM
                return self.llm.invoke(question).content
        except Exception as e:
            return f"Error during response generation: {str(e)}"

    def add_new_pdfs(self, new_pdfs: list):
        try:
            self.pdf_docs += new_pdfs
            new_text = self.get_pdf_text(new_pdfs)
            new_chunks = self.get_text_chunks(new_text)
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
