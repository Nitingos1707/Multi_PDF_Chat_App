from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_groq.chat_models import ChatGroq
from PyPDF2 import PdfReader
import streamlit as st
import tiktoken
import torch
import os

class MultiPDFChatApp:
    def __init__(self, project_name: str, pdf_docs: list = []):
        """Initialize the Multi-PDF Chat Application"""
        torch.classes.__path__ = []
        
        # Create storage directory
        BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Storage")        
        if os.path.exists(BASE_DIR):
            print("Project directory exists.")
        else:
            os.makedirs(BASE_DIR)
            print("Project directory created.")
        
        # Initialize instance variables
        self.pdf_docs = pdf_docs
        self.raw_text = ""
        self.text_chunks = None
        self.vectorstore = None
        self.conversation_chain = None
        self.chat_history = None
        self.chroma_persist_dir = f"{BASE_DIR}/{project_name}_chroma_db"  # Directory to store ChromaDB
        self.project_name = project_name
        
        print(f"MultiPDFChatApp initialized for project: {project_name}")
    
    def get_pdf_text(self):
        """Extract text from all uploaded PDF files"""
        self.raw_text = ""
        print("Extracting text from PDFs...")
        
        try:
            if not self.pdf_docs:
                raise ValueError("No PDF files provided")
            
            for i, pdf in enumerate(self.pdf_docs):
                print(f"Extracting text from PDF {i+1}: {pdf.name}")
                
                try:
                    # Reset file pointer to beginning
                    pdf.seek(0)
                    pdf_reader = PdfReader(pdf)
                    
                    print(f"PDF has {len(pdf_reader.pages)} pages")
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():  # Only add non-empty pages
                                self.raw_text += f"\n--- Document: {pdf.name} (Page {page_num + 1}) ---\n"
                                self.raw_text += page_text + "\n"
                                print(f"Extracted {len(page_text)} characters from page {page_num + 1}")
                        except Exception as e:
                            print(f"Error extracting text from page {page_num + 1} of {pdf.name}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error reading PDF {pdf.name}: {e}")
                    # Try alternative approach
                    try:
                        import PyPDF2
                        pdf.seek(0)
                        pdf_reader = PyPDF2.PdfReader(pdf)
                        for page_num, page in enumerate(pdf_reader.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    self.raw_text += f"\n--- Document: {pdf.name} (Page {page_num + 1}) ---\n"
                                    self.raw_text += page_text + "\n"
                            except:
                                continue
                    except Exception as e2:
                        print(f"Alternative PDF reading also failed for {pdf.name}: {e2}")
                        continue
            
            if not self.raw_text.strip():
                raise ValueError("No text could be extracted from any of the PDF files. The PDFs might be image-based or corrupted.")
                
            print(f"Text extraction completed. Total characters: {len(self.raw_text)}")
            return self.raw_text
            
        except Exception as e:
            print(f"Error in PDF text extraction: {e}")
            raise Exception(f"Failed to extract text from PDFs: {str(e)}")
    
    def get_text_chunks(self):
        """Split the extracted text into manageable chunks"""
        try:
            if not self.raw_text:
                raise ValueError("No text available to split into chunks")
                
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            print("Splitting text into chunks...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Adjust based on model context window
                chunk_overlap=200,
                length_function=lambda text: len(encoding.encode(text)),  # Token-based splitting
                separators=["\n\n", "\n", " ", ""]
            )
            
            self.text_chunks = text_splitter.split_text(self.raw_text)
            
            if not self.text_chunks:
                raise ValueError("No text chunks were created")
                
            print(f"Text split into {len(self.text_chunks)} chunks.")
            return self.text_chunks
            
        except Exception as e:
            print(f"Error in text chunking: {e}")
            raise Exception(f"Failed to split text into chunks: {str(e)}")
    
    def get_vectorstore(self):
        """Create or load ChromaDB vectorstore"""
        try:
            if not self.text_chunks:
                raise ValueError("No text chunks available for vectorstore creation")
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Check if ChromaDB exists and is not empty
            if os.path.exists(self.chroma_persist_dir) and os.listdir(self.chroma_persist_dir):
                print("Loading existing ChromaDB vectorstore...")
                try:
                    self.vectorstore = Chroma(
                        persist_directory=self.chroma_persist_dir,
                        embedding_function=embeddings
                    )
                    
                    # Test if vectorstore is working
                    test_results = self.vectorstore.similarity_search("test", k=1)
                    
                    # Get existing document IDs to avoid duplicates
                    existing_docs = self.vectorstore.get(include=["documents"])["documents"]
                    
                    # Identify new documents to add
                    new_docs = [doc for doc in self.text_chunks if doc not in existing_docs]
                    
                    if new_docs:
                        print(f"Adding {len(new_docs)} new documents to the vectorstore...")
                        self.vectorstore.add_texts(new_docs)
                        print("New documents added successfully.")
                    else:
                        print("No new documents to add.")
                        
                except Exception as e:
                    print(f"Error loading existing vectorstore: {e}")
                    print("Creating new vectorstore...")
                    # If loading fails, create new vectorstore
                    self.vectorstore = Chroma.from_texts(
                        texts=self.text_chunks, 
                        embedding=embeddings, 
                        persist_directory=self.chroma_persist_dir
                    )
            else:
                print("Creating new ChromaDB vectorstore...")
                self.vectorstore = Chroma.from_texts(
                    texts=self.text_chunks, 
                    embedding=embeddings, 
                    persist_directory=self.chroma_persist_dir
                )
                print("Vectorstore created successfully.")
            
            return self.vectorstore

        except Exception as e:
            print(f"Error handling vectorstore: {e}")
            raise Exception(f"Failed to create/load vectorstore: {str(e)}")
    
    def get_conversation_chain(self, question):
        """Get AI response for the given question"""
        try:
            if not question or not question.strip():
                return "Please provide a valid question."
            
            if not self.vectorstore:
                return "Error: Vectorstore not initialized. Please restart the chat."
            
            # Get API key from Streamlit secrets
            try:
                api_key = st.secrets["GROQ_API_KEY"]
                model = st.secrets.get("GROQ_MODEL", "mixtral-8x7b-32768")
            except Exception as e:
                print(f"Error accessing secrets: {e}")
                return f"Error: API configuration not found. Please check your Streamlit secrets."
            
            if not api_key or api_key == "your-groq-api-key":
                return "Error: Please set your actual GROQ_API_KEY in Streamlit Cloud secrets"
            
            print(f"Using model: {model}")
            
            # Initialize the language model
            llm = ChatGroq(
                api_key=api_key,
                model_name=model,
                temperature=0.4,
                max_tokens=1000,
                timeout=30
            )
            
            # Create memory for conversation history
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
            
            # Create the conversational retrieval chain
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Return top 4 most relevant chunks
                ),
                memory=memory,
                return_source_documents=False,
                verbose=True
            )
            
            # Get response
            print(f"Processing question: {question[:100]}...")
            response = conversation_chain.invoke({'question': question})
            
            answer = response.get('answer', 'No answer generated')
            
            if not answer or answer.strip() == "":
                return "I couldn't generate a response. Please try rephrasing your question."
            
            print("Response generated successfully.")
            return answer
            
        except Exception as e:
            error_msg = f"Error in conversation chain: {str(e)}"
            print(error_msg)
            
            # Return user-friendly error messages
            if "API" in str(e):
                return "Error: There was an issue with the AI service. Please check your API configuration."
            elif "timeout" in str(e).lower():
                return "Error: Request timed out. Please try again with a simpler question."
            elif "rate" in str(e).lower() or "limit" in str(e).lower():
                return "Error: API rate limit exceeded. Please wait a moment and try again."
            else:
                return f"Error: {str(e)}"
    
    def run_chat(self):
        """Initialize the chat application by processing PDFs and creating vectorstore"""
        try:
            print("Starting chat initialization...")
            
            # Debug: Check if PDFs are available
            if not self.pdf_docs:
                raise Exception("No PDF documents provided")
            
            print(f"Processing {len(self.pdf_docs)} PDF file(s)")
            for i, pdf in enumerate(self.pdf_docs):
                print(f"PDF {i+1}: {pdf.name} ({pdf.size} bytes)")
            
            # Step 1: Extract text from PDFs
            print("Step 1: Extracting text from PDFs...")
            self.get_pdf_text()
            if not self.raw_text:
                raise Exception("No text extracted from PDFs - files might be image-based or corrupted")
            print(f"✅ Text extraction successful: {len(self.raw_text)} characters")
            
            # Step 2: Split text into chunks
            print("Step 2: Splitting text into chunks...")
            self.get_text_chunks()
            if not self.text_chunks:
                raise Exception("No text chunks created - text might be too short")
            print(f"✅ Text chunking successful: {len(self.text_chunks)} chunks")
            
            # Step 3: Create vectorstore
            print("Step 3: Creating vectorstore...")
            self.get_vectorstore()
            if not self.vectorstore:
                raise Exception("Vectorstore creation failed - check embeddings")
            print("✅ Vectorstore creation successful")
            
            print("🎉 Chat initialization completed successfully.")
            return True
            
        except Exception as e:
            error_msg = f"❌ Error in chat initialization: {str(e)}"
            print(error_msg)
            
            # More specific error messages
            if "No text extracted" in str(e):
                print("💡 Suggestion: Your PDF might be image-based. Try using OCR tools to convert it to text-based PDF first.")
            elif "No PDF documents" in str(e):
                print("💡 Suggestion: Make sure PDF files are properly uploaded.")
            elif "Vectorstore creation failed" in str(e):
                print("💡 Suggestion: There might be an issue with the embedding model or storage.")
            
            return False
    
    def reset_chat(self):
        """Reset the chat application"""
        try:
            self.raw_text = ""
            self.text_chunks = None
            self.vectorstore = None
            self.conversation_chain = None
            self.chat_history = None
            print("Chat reset successfully.")
            return True
        except Exception as e:
            print(f"Error resetting chat: {e}")
            return False
