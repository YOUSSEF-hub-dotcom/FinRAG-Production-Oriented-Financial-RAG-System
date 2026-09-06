from pymongo import MongoClient
c=MongoClient("mongodb://localhost:27017",serverSelectionTimeoutMS=5000)
col=c["financial_rag"]["raw_chunks"]
print("MONGO raw_chunks total =", col.count_documents({}))
for t in col.aggregate([{"$group":{"_id":"$ticker","n":{"$sum":1}}}],allowDiskUse=True):
    print("  ticker", t["_id"], "->", t["n"])
for t in col.aggregate([{"$group":{"_id":{"t":"$ticker","fy":"$fiscal_year"},"n":{"$sum":1}}}],allowDiskUse=True):
    print("  ", t["_id"], "->", t["n"])
s=col.find_one({})
if s:
    print("sample keys:", list(s.keys()))
    print("  ticker=", s.get("ticker"), "fiscal_year=", repr(s.get("fiscal_year")),
          "section=", s.get("section"), "chunk_type=", s.get("chunk_type"),
          "has_raw_text=", bool(s.get("raw_text")))
    print("  empty raw_text count =", col.count_documents({"raw_text":{"$in":["",None]}}))
