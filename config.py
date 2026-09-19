import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Paths
# -----------------------------

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "research_papers"

# -----------------------------
# Embedding model
# -----------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# RAG settings
# -----------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8

# -----------------------------
# LLM settings
# -----------------------------

MAX_CONTEXT_CHARS = 12000

# Provider 1
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# Provider 2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

# Provider 3
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct:free"
)