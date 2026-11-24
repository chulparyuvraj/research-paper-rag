import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Tried in order. If one fails (unavailable, rate-limited), the next is used.
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

SYSTEM_PROMPT = (
    "You are a research assistant. Answer ONLY using the numbered context "
    "passages provided. Cite the passages you use inline like [1] or [2]. "
    "If the passages do not contain the answer, reply exactly: "
    "'I could not find this in the provided documents.' Be concise."
)


def build_context(hits):
    return "\n\n".join(
        f"[{i + 1}] ({h['source']} p.{h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )

def clean(text):
    return re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)


def answer(question, hits, models=MODELS):
    user_msg = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    last_error = "no model tried"
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.1,
                max_tokens=1500,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            last_error = f"{model} returned an empty answer"
        except Exception as e:
            last_error = f"{model}: {e}"
    return f"LLM error: {last_error}"