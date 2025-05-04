from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from prompt import prompt
import asyncio
import re
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

import nltk
nltk.download('punkt')

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
df = pd.read_parquet("sberquad/sberquad/validation-00000-of-00001.parquet").sample(frac=1)[:50]
questions = df["question"].tolist()
contexts = df["context"].tolist()
answers = df["answers"].apply(lambda x: x["text"][0]).tolist()

async def main():
    requests = []
    print("Sending requests...")
    for question, context in zip(questions, contexts):
        requests.append(completion_client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
                {'role': 'user', 'content': prompt.format(context=context, question=question)}
            ],
            temperature=0,
            max_tokens=1024
        ))

    responses = await asyncio.gather(*requests)
    responses = [response.choices[0].message.content for response in responses]
    responses = [re.sub(r"<think>.*?</think>\n\n", "", response, flags=re.DOTALL) for response in responses]
    print("Received responses")

    bleu_scores = []
    right, wrong = 0, 0
    judge_requests = []
    print("Sending judge requests...")
    for question, context, response in zip(questions, contexts, responses):
        judge_requests.append(judgement_client.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[
                {"role": "user", "content": f"/no_think\nВопрос: {question}\nКонтекст: {context}\nОтвет модели: {response}\n\nПравильно ли ответила модель на вопрос, ответ на который содержится в контексте? Ответь только 'правильно' или 'неправильно'"}
            ],
            max_tokens=8,
            temperature=0
        ))

    judge_responses = await asyncio.gather(*judge_requests)
    judge_responses = [response.choices[0].message.content.strip().lower() for response in judge_responses]
    print("Received judge responses")

    for i, judge_response in enumerate(judge_responses):
        print('-' * 50)
        print(f"Question {i}: {questions[i]}")
        print(f"Context: {contexts[i]}")
        print(f"Answer: {answers[i]}")
        print(f"Response: {responses[i]}")
        print(f"Judge response: {judge_response}")

        # Calculate BLEU score
        reference = [answers[i].split()] # Reference sentence needs to be a list of lists of tokens
        candidate = responses[i].split() # Candidate sentence is a list of tokens
        try:
            # Using smoothing function 1 for short sentences
            score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        except ZeroDivisionError:
            score = 0.0 # Handle cases where candidate is too short or has no overlap
        bleu_scores.append(score)
        print(f"BLEU Score: {score:.4f}")

        print("-" * 50)
        if judge_response.startswith("правильно"):
            right += 1
        else:
            wrong += 1

    # Calculate and print average BLEU score
    average_bleu = np.mean(bleu_scores) if bleu_scores else 0

    print(f"Right: {right}, Wrong: {wrong}")
    print(f"Accuracy: {right / (right + wrong):.4f}")
    print(f"Average BLEU Score: {average_bleu:.4f}")

if __name__ == "__main__":
    asyncio.run(main())