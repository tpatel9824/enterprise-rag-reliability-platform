import re
from .models import RetrievalResult

class OfflineGenerator:
    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        if not retrieved: return "I could not find enough evidence in the approved knowledge base."
        best=retrieved[0]
        sentences=re.split(r"(?<=[.!?])\s+", best.chunk.text.strip())
        answer=" ".join([s for s in sentences if s][:2])
        return f"{answer} [{best.chunk.document_id}]"

class GeminiGenerator:
    def __init__(self, model_name: str):
        from google import genai
        from google.genai.types import HttpOptions
        if not model_name: raise ValueError("GENAI_MODEL must be set when Gemini is enabled")
        self.client=genai.Client(http_options=HttpOptions(api_version="v1")); self.model_name=model_name
    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        if not retrieved: return "I could not find enough evidence in the approved knowledge base."
        context="\n\n".join(f"[{r.chunk.document_id} | {r.chunk.section}] {r.chunk.text}" for r in retrieved)
        prompt=("You are an enterprise knowledge assistant. Answer only from the supplied evidence. "
                "If evidence is insufficient, say so. Do not invent policy. Cite source document ids.\n\n"
                f"Evidence:\n{context}\n\nQuestion:\n{question}")
        response=self.client.models.generate_content(model=self.model_name, contents=prompt,
            config={"temperature":0.0,"max_output_tokens":400})
        return response.text.strip()

def build_generator(use_gemini: bool, model_name: str):
    return GeminiGenerator(model_name) if use_gemini else OfflineGenerator()
