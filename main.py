from openai import OpenAI
from dotenv import load_dotenv  
import os
from prompt import prompt

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def answer_question(context: str, question: str) -> str:
    response = client.completions.create(
        model=os.getenv("MODEL"),
        prompt=prompt.format(context=context, question=question),
        temperature=0,
        max_tokens=64,
        stop=["Контекст"]
    )

    return response.choices[0].text