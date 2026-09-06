import json, urllib.request, urllib.error, sys
from pymongo import MongoClient

# ---------- B. MongoDB inspection (allowed: concurrent connections) ----------
try:
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    db = mc["financial_rag"]
    col = db["raw_chunks"]
    total = col.count_documents({})
    print("MONGO total raw_chunks =", total)
    # per ticker
    for t in col.aggregate([{"$group":{"_id":"$ticker","n":{"$sum":1}}}], allowDiskUse=True):
        print("  ticker", t["_id"], "->", t["n"])
    # per ticker+fy
    print("--- per ticker/fiscal_year ---")
    for t in col.aggregate([{"$group":{"_id":{"t":"$ticker","fy":"$fiscal_year"},"n":{"$sum":1}}}], allowDiskUse=True):
        print("  ", t["_id"], "->", t["n"])
    # sample schema
    s = col.find_one({})
    if s:
        print("--- sample doc keys ---")
        print("  ticker=", s.get("ticker"), "fiscal_year=", repr(s.get("fiscal_year")),
              "section=", s.get("section"), "chunk_type=", s.get("chunk_type"),
              "has_raw_text=", bool(s.get("raw_text")))
        # how many have empty raw_text
        empt = col.count_documents({"raw_text": {"$in":["", None]}})
        print("  docs with empty raw_text =", empt)
except Exception as e:
    print("MONGO ERROR:", repr(e))

# ---------- C. Live API retrieval behavior ----------
BASE="http://127.0.0.1:8000/api/v1"
def post(p,b,t=None):
    r=urllib.request.Request(BASE+p,data=json.dumps(b).encode(),method="POST")
    r.add_header("Content-Type","application/json")
    if t: r.add_header("Authorization","Bearer "+t)
    try:
        with urllib.request.urlopen(r,timeout=120) as x: return x.status,json.loads(x.read().decode())
    except urllib.error.HTTPError as e: return e.code,json.loads(e.read().decode())
try:
    s,resp=post("/auth/login",{"email":"youssefaboali122@gmail.com","password":"12345abcde"})
    tok=resp.get("access_token"); print("\nLOGIN", s)
except Exception as e:
    print("LOGIN ERROR", repr(e)); tok=None

if tok:
    qs=[
        ("AAPL","What was Apple's total net revenue in FY2025?"),
        ("AAPL","What is Apple's operating margin in FY2025?"),
        ("AAPL","How much did Apple spend on R&D in FY2025?"),
        ("MSFT","What was Microsoft's net income in FY2025?"),
        ("MSFT","What is Microsoft's operating margin in FY2025?"),
        ("MSFT","What was Microsoft's Intelligent Cloud segment revenue in FY2025?"),
        ("NVDA","What was NVIDIA's total revenue in FY2025?"),
        ("NVDA","What is NVIDIA's gross margin in FY2025?"),
        ("NVDA","What was NVIDIA's Data Center revenue in FY2025?"),
    ]
    for tk,q in qs:
        try:
            st,rp=post("/chat",{"user_query":q,"ticker":tk,"fiscal_year":"2025","session_id":"diag"},t=tok)
            ans=rp.get("answer","")[:160]
            ch=rp.get("cache_hit")
            na = "not available" in ans.lower()
            print(f"[{tk}] {q}\n   -> {st} cache={ch} NOTAVAIL={na}\n   {ans}\n")
        except Exception as e:
            print(f"[{tk}] {q} ERROR {repr(e)}")
