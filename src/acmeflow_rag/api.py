from fastapi import FastAPI
from pydantic import BaseModel, Field
from .config import settings
from .service import RagService
app=FastAPI(title="AcmeFlow Enterprise RAG",version="0.1.0")
service=RagService()
class AskRequest(BaseModel):
    question:str=Field(min_length=3)
    top_k:int=Field(default=settings.top_k,ge=1,le=10)
    roles:list[str]=[]
    department:str|None=None
@app.get("/health")
def health():
    return {"status":"ok","documents":len({c.document_id for c in service.retriever.chunks}),"chunks":len(service.retriever.chunks)}
@app.post("/ask")
def ask(req:AskRequest):
    return service.ask(req.question,req.top_k,set(req.roles) if req.roles else None,req.department)
