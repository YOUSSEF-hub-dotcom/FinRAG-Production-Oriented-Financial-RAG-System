from pymongo import MongoClient

c = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=10000)
db = c["financial_rag"]
col = db["raw_chunks"]
print("collections:", db.list_collection_names())
print("docs:", col.count_documents({}))
print("tables:", col.count_documents({"chunk_type": "table"}))
