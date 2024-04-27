import asyncio
from openai import OpenAI
import os
from dotenv import load_dotenv

# from config.config import OPENAI_API_KEY, GPT_MODEL

load_dotenv()

class OPENAI:
    def __init__(self, running_model=os.getenv("GPT_MODEL")):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = running_model

    def chat_completion(self, message, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            **kwargs,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    message = "This is a test message, return in JSON"
    client = OPENAI()
    response = client.chat_completion(message, temperature=0)
    print(response)