# qachain_setup.py

from langchain.chains import RetrievalQA
from llm import GroqLLM
from retriever import get_vectorstore_retriever

# Assume vectorstore_retriever is already loaded
# Example: from your_vectorstore_setup import vectorstore_retriever

# --- QA Chain Setup ---
turbo_llm = GroqLLM()

if turbo_llm:
    qa_chain = RetrievalQA.from_chain_type(
        llm=turbo_llm,
        chain_type="stuff",
        retriever=get_vectorstore_retriever(),
        return_source_documents=True
    )
    print("✅ QA Chain created using Groq.")
else:
    qa_chain = None
    print("⚠️ QA Chain not created (LLM missing).")

# --- Querying the RAG System ---
if qa_chain:
    user_query = input("Please enter your query: ")
    full_query = user_query

    print(f"\n🔍 Processing query: '{full_query}'")
    llm_response = qa_chain.invoke({"query": full_query})

    print("\n🤖 LLM Response:")
    print(llm_response.get("result", "No result generated."))

    print("\n📚 Source Documents Used:")
    if llm_response.get("source_documents"):
        for i, source_doc in enumerate(llm_response["source_documents"]):
            print(f"\n--- Source Document {i+1} ---")
            print(f"Content: {source_doc.page_content}")
            if source_doc.metadata:
                print(f"Metadata: {source_doc.metadata}")
    else:
        print("No source documents were returned.")
else:
    print("❌ Cannot query as the QA chain was not set up.")

print("\n✅ --- RAG Process Complete ---")
