from openai import OpenAI
from dotenv import load_dotenv  
import os
from prompt import prompt
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def answer_question(context: str, question: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'user', 'content': prompt.format(context=context, question=question)}
        ],
        temperature=0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    return answer