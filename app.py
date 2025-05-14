# Install the required packages if you haven't already:
# pip install langchain langchain-community sentence_transformers chromadb huggingface_hub python-dotenv unstructured

# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_core.documents import Document # To create Document objects from text chunks
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub

# --- Configuration ---
# Load environment variables (e.g., for HUGGINGFACEHUB_API_TOKEN)
# Create a .env file in your project root with HUGGINGFACEHUB_API_TOKEN="your_token"
load_dotenv()

# Retrieve the Hugging Face API token from environment variables
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_api_token:
    print("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your environment or .env file.")
    

# Define the embedding model
# SentenceTransformerEmbeddings is used to load embedding models to convert text to numerical representations.
embeddings_model_name = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model_name)

# Define the path for persisting the vector database
# This is where your vectorized chunks will be stored.
persist_directory = "my_rag_chroma_db_from_chunks"


pre_existing_text_chunks = [
    "The capital of France is Paris. It is known for the Eiffel Tower.",
    "Langchain is a powerful framework for developing applications powered by language models.",
    "Vector databases like Chroma are essential for storing and efficiently retrieving text embeddings in RAG systems.",
    "RAG, or Retrieval Augmented Generation, enhances LLM responses by grounding them in external knowledge.",
    "To initialize a HuggingFaceHub LLM in Langchain, you need an API token."
]

if not pre_existing_text_chunks:
    print("Error: The 'pre_existing_text_chunks' list is empty. Please add your text chunks.")
    exit()

# Convert your text chunks into Langchain Document objects.
# This is a necessary step for many Langchain vector store integrations.
# You can also add metadata to each document if you have it.
print(f"Converting {len(pre_existing_text_chunks)} text chunks to Document objects...")
docs = []
for i, chunk_text in enumerate(pre_existing_text_chunks):
    # You can add metadata relevant to each chunk, e.g., its original source, page number, etc.
    metadata = {"source": f"pre_loaded_chunk_{i+1}"}
    docs.append(Document(page_content=chunk_text, metadata=metadata))
print("Document objects created.")

# --- Vector Store Creation ---
# This code will convert the Document objects (your chunks) to vectors using the specified embeddings model
# and store these vectors in a Chroma vector database.
# The persist_directory allows the database to be saved to disk and reloaded later.
# If the directory already exists and contains a compatible database, Chroma might load it.
# For a fresh start, ensure the directory is empty or use a new one.
print(f"Creating or loading vector store in directory: '{persist_directory}'...")
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
print("Vector store created/loaded and persisted successfully.")

# --- Retriever Setup ---
# The retriever is responsible for fetching relevant documents from the vector database based on a query.
# 'search_kwargs={"k": 3}' means it will retrieve the top 3 most similar documents. Adjust 'k' as needed.
vectorstore_retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)
print("Retriever set up.")

# --- LLM Setup ---
# Here, HuggingFaceHub is used to access a Large Language Model (LLM).
# We are using 'google/flan-t5-large'. You can explore other models on Hugging Face Hub.
# 'temperature' controls the randomness of the output (0.0 is deterministic, higher is more random).
# 'max_length' sets the maximum number of tokens the model can generate for the answer.
print("Setting up LLM via HuggingFaceHub...")
if huggingface_api_token: # Proceed only if token is available
    turbo_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature":0.5, "max_length":512}, # Adjusted max_length from original 512 to 150 for concise answers
        huggingfacehub_api_token=huggingface_api_token
    )
    print("LLM set up successfully.")
else:
    print("LLM setup skipped due to missing Hugging Face API token.")
    turbo_llm = None # Ensure llm variable exists

# --- QA Chain Setup ---
# RetrievalQA chain combines the retriever and the LLM.
# 'chain_type="stuff"' means all retrieved documents will be "stuffed" into the context for the LLM.
# Other chain types like "map_reduce" or "refine" can be used for larger contexts.
# 'return_source_documents=True' allows you to see which documents were retrieved and used for the answer.
if turbo_llm:
    qa_chain = RetrievalQA.from_chain_type(
        llm=turbo_llm,
        chain_type="stuff", # "stuff" is good for a few documents. Consider other types for many/large docs.
        retriever=vectorstore_retriever,
        return_source_documents=True
    )
    print("QA Chain created.")
else:
    qa_chain = None
    print("QA Chain creation skipped as LLM is not available.")

# --- Querying the RAG System ---
if qa_chain:
    # Define your query.
    user_query = input("Please enter your query: ")

    # Optional: Add instructions or context to your query for the LLM.
    # This prompt engineering can significantly improve results.
    # warning_prompt = "Please refrain from speculating if you're unsure. Simply state that you don't know. Answers should be concise, within 100 words."
    # instruction_prompt = "You are a helpful AI Assistant. Your job is to generate an output based on the provided context documents for the query."
    # full_query = f"{warning_prompt} {instruction_prompt} Query: {user_query}"
    
    # Using the plain user query for this example
    full_query = user_query

    print(f"\nProcessing query: '{full_query}'")
    # Use .invoke() with a dictionary for the input. The key for RetrievalQA is "query".
    llm_response = qa_chain.invoke({"query": full_query})

    # --- Displaying the Response ---
    print("\nLLM Response:")
    print(llm_response.get("result", "No result generated."))

    print("\nSource Documents Used:")
    if llm_response.get("source_documents"):
        for i, source_doc in enumerate(llm_response["source_documents"]):
            print(f"\n--- Source Document {i+1} ---")
            print(f"Content: {source_doc.page_content}")
            if source_doc.metadata:
                print(f"Metadata: {source_doc.metadata}")
    else:
        print("No source documents were returned.")
else:
    print("Cannot query as the QA chain was not set up (likely due to missing LLM).")

print("\n--- RAG Process Complete ---")

# --- Optional: How to reload the persisted database later ---
# If you want to reuse the created vector database in another session or script without re-processing the chunks:
# Ensure the `embeddings` model is the same one used during creation.
# print("\nExample of reloading the persisted database:")
# persisted_db_path_to_load = persist_directory # Should be the same path used above
# if os.path.exists(persisted_db_path_to_load) and os.path.isdir(persisted_db_path_to_load):
#     print(f"Attempting to load persisted vector store from: '{persisted_db_path_to_load}'")
#     try:
#         loaded_vectordb = Chroma(
#             persist_directory=persisted_db_path_to_load,
#             embedding_function=embeddings # Must use the same embedding function
#         )
#         loaded_retriever = loaded_vectordb.as_retriever(search_kwargs={"k": 3})
#         print("Retriever reloaded successfully from persisted database.")
#         # You can then create a new qa_chain with this loaded_retriever and your LLM:
#         # if turbo_llm:
#         #   reloaded_qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=loaded_retriever, return_source_documents=True)
#         #   # Now you can use reloaded_qa_chain.invoke({"query": "Your new query"})
#     except Exception as e:
#         print(f"Error loading persisted vector store: {e}")
# else:
#     print(f"Persisted directory '{persisted_db_path_to_load}' not found. Cannot reload.")