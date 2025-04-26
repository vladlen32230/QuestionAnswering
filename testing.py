from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from prompt import prompt
import asyncio

load_dotenv()

completion_client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

judgement_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# git clone https://huggingface.co/datasets/kuznetsoffandrey/sberquad
df = pd.read_parquet("sberquad/sberquad/validation-00000-of-00001.parquet").sample(frac=1)[:100]
questions = df["question"].tolist()
contexts = df["context"].tolist()
answers = df["answers"].apply(lambda x: x["text"][0]).tolist()

async def main():
    requests = []
    print("Sending requests...")
    for question, context in zip(questions, contexts):
        requests.append(completion_client.completions.create(
            model=os.getenv("MODEL"),
            prompt=prompt.format(context=context, question=question),
            temperature=0,
            max_tokens=64,
            stop=["Контекст"]
        ))

    responses = await asyncio.gather(*requests)
    responses = [response.choices[0].text for response in responses]
    print("Received responses")

    right, wrong = 0, 0
    judge_requests = []
    print("Sending judge requests...")
    for question, context, response in zip(questions, contexts, responses):
        judge_requests.append(judgement_client.chat.completions.create(
            model="mistral/ministral-8b",
            messages=[
                {"role": "system", "content": f"Вопрос: {question}\nКонтекст: {context}\nОтвет модели: {response}"},
                {"role": "user", "content": "Правильно ли ответила модель на вопрос, ответ на который содержится в контексте? Ответь только 'правильно' или 'неправильно'"}
            ],
            max_tokens=8,
            temperature=0
        ))

    judge_responses = await asyncio.gather(*judge_requests)
    judge_responses = [response.choices[0].message.content.strip().lower() for response in judge_responses]
    print("Received judge responses")

    for i, judge_response in enumerate(judge_responses):
        if judge_response == "правильно" or judge_response == "правильно.":
            right += 1
        else:
            print('-' * 50)
            print(f"Question {i}: {questions[i]}")
            print(f"Context: {contexts[i]}")
            print(f"Answer: {answers[i]}")
            print(f"Response: {responses[i]}")
            print(f"Judge response: {judge_response}")
            print("-" * 50)
            wrong += 1

    print(f"Right: {right}, Wrong: {wrong}")
    print(f"Accuracy: {right / (right + wrong)}")

if __name__ == "__main__":
    asyncio.run(main())