import argparse
from .config import settings
from .service import RagService

def main():
    p=argparse.ArgumentParser(); p.add_argument("question"); p.add_argument("--top-k",type=int,default=settings.top_k); p.add_argument("--role",action="append",default=[]); p.add_argument("--department",default=None)
    a=p.parse_args(); result=RagService().ask(a.question,a.top_k,set(a.role) if a.role else None,a.department)
    print("\nANSWER\n"+result["answer"]); print("\nABSTAINED:",result["abstained"]); print("BEST SCORE:",result["best_score"]); print("\nSOURCES")
    for i,item in enumerate(result["retrieval"],1):
        c=item["chunk"]; print(f'{i}. {c["document_id"]} | {c["section"]} | score={item["score"]:.4f}')
if __name__ == "__main__": main()
