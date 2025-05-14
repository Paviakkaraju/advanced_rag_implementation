import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# # Load environment variables if needed
load_dotenv()

huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_api_token:
    print("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your environment or .env file.")

# # Configuration
persist_directory = "codework_chunks"  # Directory where Chroma DB is stored
embedding_model_name = "all-MiniLM-L6-v2"

# # Initialize the embedding model (must be the same as used during indexing)
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

# # Load the existing vector store from disk
print("Loading persisted Chroma vector store...")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print("Vector store loaded successfully.")

# # Define a retrieval
def get_vectorstore_retriever(k=3, vectorstore=vectordb):
     """
     Returns a retriever from the loaded vector store.
    
     Args:
         k (int): Number of top similar results to retrieve.
    
     Returns:
         Retriever object.
     """
     vectorstore_retriever = vectordb.as_retriever(search_kwargs={"k": k})
     print(f"Retriever set up with top-{k} result(s).")
     return vectorstore_retriever

if __name__ == "__main__":
    retriever = get_vectorstore_retriever()
    # Optional test input
    query = "What services does Codework provide?"
    results = retriever.get_relevant_documents(query)
    print("\nRetrieved documents:")
    for doc in results:
        print(f"- {doc.metadata.get('title', 'No title')}: {doc.page_content}")



# from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_chroma import Chroma
# from langchain.schema.runnable import Runnable
# import os

# def get_vectorstore_retriever() -> Runnable:
#     # Set up Hugging Face embeddings
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     except Exception as e:
#         raise EnvironmentError("[ERROR] HuggingFace credentials not set or model loading failed.") from e

#     # Load Chroma DB if exists
#     persist_directory = "chroma_db"
#     # if not os.path.exists(persist_directory):
#         # raise FileNotFoundError("Chroma DB directory not found. Please build and persist the vectorstore first.")

#     db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#     return db.as_retriever()

