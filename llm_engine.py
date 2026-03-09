import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL")

def generate_explanation(result):

    factors = "\n".join(
        [f"{f} ({round(v,3)})" for f, v in result["top_factors"]]
    )

    prompt = f"""
You are an AI system that explains machine learning predictions.

Prediction: {result['prediction']}
Confidence: {round(result['confidence']*100,2)}%

Key factors influencing the decision:
{factors}

Explain this decision to a normal non-technical user in simple language.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    data = response.json()

    return data["choices"][0]["message"]["content"]