import json, sys
path="/home/youssef/Financial_RAG/uvicorn.log"
kw=("error","exception","traceback","failed","not available","hybrid","rerank","documents")
out=[]
with open(path) as f:
    for line in f:
        s=line.strip()
        if not s: continue
        try:
            o=json.loads(s)
            msg=o.get("message","")
        except Exception:
            msg=s
        low=msg.lower()
        if any(k in low for k in kw):
            out.append(msg)
# print last 50 matching
for m in out[-50:]:
    print(m[:300])
print("TOTAL_MATCHES", len(out))
