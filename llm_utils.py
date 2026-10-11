
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(question, retrieved_chunks):
    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    if not retrieved_chunks:
        return "No relevant text was retrieved from the paper."

    context = "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a helpful AI Research Assistant.

Answer the user's question accurately using the research-paper
extracts provided below.

Instructions:
- Read all the supplied extracts before answering.
- Answer the actual question directly.
- Combine information from different extracts when necessary.
- Understand equivalent wording and paraphrases.
- Explain the answer in simple, clear English.
- Cite the relevant page numbers, such as [Page 3].
- Never invent facts, quotations, or page citations.
- Do not say something is absent merely because the exact
  words in the question do not appear in the paper.
- If the extracts genuinely do not contain enough evidence,
  explain what information is missing.

RESEARCH PAPER EXTRACTS:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 800
                }
            )

            if response.text and response.text.strip():
                return response.text.strip()

            return "The model returned an empty answer. Please try again."

        except errors.ServerError as error:
            if error.code != 503 or attempt == 2:
                raise

            time.sleep(2 * (attempt + 1))

    return "The AI service is temporarily busy. Please try again."
