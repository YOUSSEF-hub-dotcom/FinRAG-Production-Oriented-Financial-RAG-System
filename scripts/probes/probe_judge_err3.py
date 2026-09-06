import json

with open("/tmp/stage1c_run8.log", encoding="utf-8") as f:
    lines = f.readlines()

from collections import Counter
c = Counter()
full = []
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
        rest = msg.split("Judge call failed with", 1)[1]
        model = rest.split("(", 1)[0].strip()
        if "tokens per day" in msg:
            c[f"{model} | TPD"] += 1
        elif "tokens per minute" in msg:
            c[f"{model} | TPM"] += 1
        elif "429" in msg:
            c[f"{model} | 429-other"] += 1
        else:
            c[f"{model} | {msg[-120:]}"] += 1
        if len(full) < 4:
            full.append(msg[:600])
print("counts:", dict(c))
print()
for m in full:
    print(m)
    print("-" * 80)
