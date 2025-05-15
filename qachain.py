# qachain_tool.py
from langchain.chains import RetrievalQA
from llm import GroqLLM
from retriever import get_vectorstore_retriever

class QAChainTool:
    def __init__(self):
        self.llm = GroqLLM()
        if self.llm:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=get_vectorstore_retriever(),
                return_source_documents=True
            )
        else:
            self.qa_chain = None

    def query_relevant_chunks(self, user_query: str):
        if not self.qa_chain:
            return {
                "error": "QA chain not initialized, LLM missing."
            }
        
        response = self.qa_chain.invoke({"query": user_query})
        
        result_text = response.get("result", "No result generated.")
        source_docs = []
        if response.get("source_documents"):
            for doc in response["source_documents"]:
                source_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        return {
            "query": user_query,
            "result": result_text,
            "source_documents": source_docs
        }

# Example usage in MCP server route (pseudo):
# from qachain_tool import QAChainTool
# qa_tool = QAChainTool()
# response = qa_tool.query_relevant_chunks("What is AI?")
# return response  # send back as JSON in MCP API response
