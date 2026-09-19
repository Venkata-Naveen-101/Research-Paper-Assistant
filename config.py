import os

from dotenv import load_dotenv

load_dotenv()


# =========================================================
# RAG CONFIGURATION
# =========================================================

CHROMA_PATH = "data/chroma"

COLLECTION_NAME = "research_papers"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 150

TOP_K = 8


# =========================================================
# LLM CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)


OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "",
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free",
)