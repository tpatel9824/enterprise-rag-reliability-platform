from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    documents_dir: Path = ROOT / "data" / "documents"
    benchmark_path: Path = ROOT / "data" / "evaluation" / "benchmark.json"
    artifacts_dir: Path = ROOT / "artifacts"
    top_k: int = int(os.getenv("TOP_K", "5"))
    abstain_threshold: float = float(os.getenv("ABSTAIN_THRESHOLD", "0.08"))
    use_gemini: bool = os.getenv("RAG_USE_GEMINI", "false").lower() == "true"
    gemini_model: str = os.getenv("GENAI_MODEL", "")

settings = Settings()
