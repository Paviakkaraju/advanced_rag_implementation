
import os
import requests
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation
from dotenv import load_dotenv

load_dotenv()

class GroqLLM(LLM):
    model: str = "llama3-8b-8192"
    temperature: float = 0.5
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
    "You are a helpful and professional assistant representing Codework. Your role is to answer user queries using the provided company context to help users understand our organization better.\n"
    "Respond as if you are a part of the Codework team—use 'we' and 'our' in your responses.\n"
    "If a user asks about the company but relevant context is not available, kindly direct them to our official website for more information: https://codework.ai/\n"
    "Only respond to questions related to Codework. If a query falls outside this scope, politely inform the user that you're unable to answer it.\n"
    "Maintain a warm, natural tone in your responses. Do not mention phrases like 'according to the context' or 'based on the context'."
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
