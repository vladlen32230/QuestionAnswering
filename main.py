from openai import OpenAI
from dotenv import load_dotenv  
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def answer_question(context: str, question: str) -> str:

    prompt = """
    Контекст: {context}

    Ответь на вопрос пользователя, используя только контекст. 
    Ответ должен содержать только 1 непрерывный отрывок из контекста.
    """

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "system", "content": prompt.format(context=context)},
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    return response.choices[0].message.content