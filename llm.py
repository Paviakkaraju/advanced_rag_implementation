# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# headers = {
#     "Authorization": f"Bearer {GROQ_API_KEY}",
#     "Content-Type": "application/json"
# }

# def chat_with_groq(user_message: str, context: str = "", 
#                    model: str = "llama3-8b-8192", temperature: float = 0.7, max_tokens: int = 100) -> str:
#     """
#     Send a RAG-style prompt to the Groq API and return the assistant's reply.

#     :param user_message: The user query.
#     :param context: Retrieved context or documents.
#     :param model: Groq-hosted model name.
#     :param temperature: Sampling temperature.
#     :param max_tokens: Max tokens for completion.
#     :return: Assistant's response.
#     """
#     system_prompt = (
#         "You are a helpful assistant for codework company answers queries from users using the provided context to help me understand the company better.\n"
#         "Respond as if you are a part of the organization and don't answer any question outside the scope of codework. \n"
#         "If a user query includes about the company but you cannot find the appropriate context, redirect them to visit the company website for more details:https://codework.ai/ \n"
#         "Ensure that your tone is professional and helpful!\n\n"
#         f"Context:\n{context}"
#     )

#     data = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ],
#         "temperature": temperature,
#         "max_tokens": max_tokens
#     }

#     response = requests.post(GROQ_API_URL, headers=headers, json=data)
#     result = response.json()

#     try:
#         return result['choices'][0]['message']['content']
#     except (KeyError, IndexError):
#         return f"❌ Unexpected response: {result}"

# groq_llm.py
import os
import requests
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation
from dotenv import load_dotenv

load_dotenv()

class GroqLLM(LLM):
    model: str = "llama3-8b-8192"
    temperature: float = 0.7
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = (
        "You are a helpful assistant for codework company answers queries from users using the provided context to help me understand the company better.\n"
        "Respond as if you are a part of the organization, include 'We' and 'our', and don't answer any question outside the scope of codework. \n"
        "If a user query includes about the company but you cannot find the appropriate context, redirect them to visit the company website for more details:https://codework.ai/ \n"
         "Ensure that your tone is professional and helpful!"
        )

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"]
