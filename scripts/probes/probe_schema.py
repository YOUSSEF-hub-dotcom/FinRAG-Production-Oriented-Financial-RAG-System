from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
col = client[MONGODB_DB][MONGODB_COLLECTION]

doc = col.find_one({})
print("=== FIRST DOC ===")
for k, v in doc.items():
    s = str(v)
    print(f"{k}: {s[:300]}")
client.close()
