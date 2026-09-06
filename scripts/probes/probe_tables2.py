from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
col = client[MONGODB_DB][MONGODB_COLLECTION]

doc = col.find_one({"ticker": "AAPL", "fiscal_year": "2025", "chunk_type": "table"})
print("=== AAPL 2025 TABLE CHUNK ===")
print("chunk_id:", doc.get("chunk_id"))
print("section:", doc.get("section"))
print("raw_text[:1500]:")
print((doc.get("raw_text") or "")[:1500])

print("\n\n=== TEXT CHUNK REFERENCING TABLE_6 ===")
doc2 = col.find_one({"ticker": "AAPL", "fiscal_year": "2025", "raw_text": {"$regex": "TABLE_6"}})
if doc2:
    print("chunk_id:", doc2.get("chunk_id"), "type:", doc2.get("chunk_type"))
    print((doc2.get("raw_text") or "")[:500])

print("\n\n=== TABLE CHUNK COUNT ===")
n = col.count_documents({"chunk_type": "table"})
t = col.count_documents({"chunk_type": "text"})
print("tables:", n, "text:", t)
client.close()
