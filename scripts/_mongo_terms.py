from pymongo import MongoClient
c=MongoClient("mongodb://localhost:27017",serverSelectionTimeoutMS=5000)
col=c["financial_rag"]["raw_chunks"]
checks=[
    ("MSFT","2025","Intelligent Cloud"),
    ("NVDA","2025","Data Center"),
    ("NVDA","2025","gross margin"),
    ("NVDA","2025","Gross profit"),
]
for t,fy,term in checks:
    q={"ticker":t,"fiscal_year":fy,"raw_text":{"$regex":term,"$options":"i"}}
    n=col.count_documents(q)
    print(f"{t} {fy} contains '{term}': {n} chunks")
    if n:
        s=col.find_one(q)
        print("   example:", s["raw_text"][:160].replace("\n"," "))
