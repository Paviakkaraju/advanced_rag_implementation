import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub

# Load environment variables if needed
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_token:
    print("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your environment or .env file.")

# Configuration
embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "codework_chunks"
text_chunks_file = "chunks.json"  

# Initialize embedding model
embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model_name)

# Load pre-existing chunks from a JSON file
if not os.path.exists(text_chunks_file):
    print(f"Error: JSON file '{text_chunks_file}' not found.")
    exit()

with open(text_chunks_file, 'r', encoding='utf-8') as f:
    pre_existing_text_chunks = json.load(f)

if not pre_existing_text_chunks:
    print("Error: The 'pre_existing_text_chunks' list is empty. Please add your text chunks.")
    exit()

# Convert chunks to Document objects
print(f"Converting {len(pre_existing_text_chunks)} text chunks to Document objects...")
docs = []
for i, chunk in enumerate(pre_existing_text_chunks):
    for title, content in chunk.items():
        metadata = {"source": f"pre_loaded_chunk_{i+1}", "title": title}
        docs.append(Document(page_content=content, metadata=metadata))
print("Document objects created.")

# Create or load vector store
print(f"Creating or loading vector store in directory: '{persist_directory}'...")
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
print("Vector store created/loaded and persisted successfully.")
