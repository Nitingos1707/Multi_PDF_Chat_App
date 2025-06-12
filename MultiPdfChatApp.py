from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import tiktoken
import torch
import os

class MultiPDFChatApp:
    def __init__(self, project_name: str, pdf_docs: list = []):
        torch.classes.__path__=[]
        load_dotenv()
        
        BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Storage")        
        if os.path.exists(BASE_DIR):
            print("Project directory exists.")
        else:
            os.makedirs(BASE_DIR)
            print("Project directory created.")
        
        self.pdf_docs = pdf_docs
        self.raw_text = ""
        self.text_chunks = None
        self.vectorstore = None
        self.conversation_chain = None
        self.chat_history = None
        self.chroma_persist_dir = f"{BASE_DIR}/{project_name}_chroma_db"  # Directory to store ChromaDB
        print("MultiPDFChatApp initialized.")
    
    def get_pdf_text(self):
        self.raw_text = ""
        print("Extracting text from PDFs...")
        for pdf in self.pdf_docs:
            print(f"Extracting text from {pdf}")
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                self.raw_text += page.extract_text()
        print("Text extracted from PDFs.")
        return self.raw_text
    
    def get_text_chunks(self):
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust based on model context window
            chunk_overlap=200,
            length_function=lambda text: len(encoding.encode(text))  # Token-based splitting
        )
        self.text_chunks = text_splitter.split_text(self.raw_text)
        print("Text split into chunks.")
        return self.text_chunks
    
    def get_vectorstore(self):
        """Loads an existing ChromaDB vectorstore and updates it with new documents if needed."""
        try:
            embeddings = HuggingFaceEmbeddings()
            
            # Check if ChromaDB exists and is not empty
            if os.path.exists(self.chroma_persist_dir) and os.listdir(self.chroma_persist_dir):
                print("Loading existing ChromaDB vectorstore...")
                self.vectorstore = Chroma(
                    persist_directory=self.chroma_persist_dir,
                    embedding_function=embeddings
                )
                
                # Get existing document IDs
                existing_docs = self.vectorstore.get(include=["documents"])["documents"]
                
                # Identify new documents to add
                new_docs = [doc for doc in self.text_chunks if doc not in existing_docs]
                
                if new_docs:
                    print(f"Adding {len(new_docs)} new documents to the vectorstore...")
                    self.vectorstore.add_texts(new_docs)
                    print("New documents added successfully.")
                else:
                    print("No new documents to add.")
            
            else:
                print("Creating new ChromaDB vectorstore...")
                self.vectorstore = Chroma.from_texts(
                    texts=self.text_chunks, 
                    embedding=embeddings, 
                    persist_directory=self.chroma_persist_dir
                )            
            return self.vectorstore

        except Exception as e:
            print(f"Error handling vectorstore: {e}")
            return None
    
    def get_conversation_chain(self, question):
        try:            
            # Convert to LangChain compatible LLM
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL"),
                temperature=0.4
            )        
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(),
                memory=memory
            )
            response = conversation_chain.invoke({'question': question})
            return response['answer']
        except Exception as e:
            print(f"Error handling conversation chain: {e}")
            return None
    
    def run_chat(self):
        try:
            self.get_pdf_text()
            self.get_text_chunks()
            self.get_vectorstore()
            return True
        except Exception as e:
            print(f"Error running chat: {e}")
            return False