import requests

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


def call_gemini(prompt: str) -> str:
    """
    Call Google's Gemini API using the REST API.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is not configured.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "key": GEMINI_API_KEY
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(
        url,
        headers=headers,
        params=params,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected Gemini response: {data}"
        )


def call_groq(prompt: str) -> str:
    """
    Call Groq's OpenAI-compatible chat API.
    """

    if not GROQ_API_KEY:
        raise RuntimeError("Groq API key is not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.1,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected Groq response: {data}"
        )


def call_openrouter(prompt: str) -> str:
    """
    Call OpenRouter's OpenAI-compatible API.
    """

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OpenRouter API key is not configured."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Research Paper Assistant",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.1,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected OpenRouter response: {data}"
        )


def generate_answer(prompt: str):
    """
    Try the configured providers in order.

    If one provider fails, the next provider is attempted.
    """

    providers = [
        ("Gemini", call_gemini),
        ("Groq", call_groq),
        ("OpenRouter", call_openrouter),
    ]

    errors = []

    for provider_name, provider_function in providers:

        try:
            answer = provider_function(prompt)

            if answer and answer.strip():
                return {
                    "provider": provider_name,
                    "answer": answer.strip(),
                }

        except Exception as error:
            errors.append(
                f"{provider_name}: {error}"
            )

    error_message = "\n".join(errors)

    raise RuntimeError(
        "All configured LLM providers failed.\n\n"
        + error_message
    )