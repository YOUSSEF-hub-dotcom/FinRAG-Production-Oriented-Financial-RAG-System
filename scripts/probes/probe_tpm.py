import json
import re

with open("/tmp/stage1b_run3.log", encoding="utf-8") as f:
    lines = f.readlines()

for ln in lines:
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    try:
        obj = json.loads(ln)
    except Exception:
        continue
    msg = obj.get("message", "")
    if "Error code: 413" in msg and "Request too large" in msg:
        m = re.search(r"tokens per minute \(TPM\): Limit (\d+), Requested (\d+)", msg)
        model = re.search(r"model `([^`]+)`", msg)
        ts = obj.get("timestamp", "")
        if m:
            print(ts[11:19], model.group(1) if model else "?", "Limit=" + m.group(1), "Requested=" + m.group(2))
