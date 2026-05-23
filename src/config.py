from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_MANUALS_DIR = DATA_DIR / "raw" / "manuals"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = PROJECT_ROOT / "eval"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "manuals.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "mistral"
