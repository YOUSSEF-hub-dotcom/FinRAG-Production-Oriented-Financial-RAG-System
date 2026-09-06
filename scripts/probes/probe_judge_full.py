import json

with open("/tmp/stage1c_run5.log", encoding="utf-8") as f:
    lines = f.readlines()

for ln in lines:
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    try:
        o = json.loads(ln)
    except Exception:
        continue
    msg = o.get("message", "")
    if "Judge call failed with" in msg:
        print(msg[:1000])
        break
