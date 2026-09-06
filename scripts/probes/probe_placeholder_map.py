import re
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/1_ingestion")

from hybrid_chunker import _make_chunk_id, _TABLE_PLACEHOLDER
from pymongo import MongoClient
from config.settings import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
col = client[MONGODB_DB][MONGODB_COLLECTION]

# The AAPL 2025 income statement table (index 0014 = idx 14)
target = "AAPL_tbl_500b35644aa6_0014"
doc = col.find_one({"chunk_id": target})
print("target:", target, "->", "EXISTS" if doc else "MISSING")
print("source_file:", doc.get("source_file"), "ticker:", doc.get("ticker"), "year:", doc.get("fiscal_year"))

# Now test _make_chunk_id reconstruction
source = doc.get("source_file")
cid = _make_chunk_id("AAPL", "2025", "tbl", source, 14)
print("reconstructed:", cid, "-> MATCH" if cid == target else "-> MISMATCH")
print("placeholder regex finds:", _TABLE_PLACEHOLDER.findall("... (dollars in millions): %%TABLE_6%% Ame"))

# Find a text chunk containing a TABLE_ placeholder and list which indices
textchunk = col.find_one({"ticker": "AAPL", "fiscal_year": "2025", "chunk_type": "text", "raw_text": {"$regex": "%%TABLE_"}})
if textchunk:
    print("\ntext chunk:", textchunk["chunk_id"], "source:", textchunk.get("source_file"))
    idxs = [int(m) for m in re.findall(r"%%TABLE_(\d+)%%", textchunk["raw_text"])]
    print("referenced table indices:", idxs)
    for i in idxs[:6]:
        c = _make_chunk_id("AAPL", "2025", "tbl", textchunk.get("source_file"), i)
        d = col.find_one({"chunk_id": c}, {"chunk_id": 1, "chunk_type": 1})
        print("   idx", i, "->", c, "->", "FOUND" if d else "MISSING")
client.close()
