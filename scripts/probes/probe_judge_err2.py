import json

with open("/tmp/stage1c_run4.log", encoding="utf-8") as f:
    lines = f.readlines()

from collections import Counter
c = Counter()
examples = {}
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
        err = rest.split(": ", 1)[1][:160] if ": " in rest else rest[:160]
        key = f"{model} | {err}"
        c[key] += 1
        examples.setdefault(key, msg[:220])

print("failures:", len(c))
for k, v in c.most_common(15):
    print(" ", v, k)

print("\n== sample messages ==")
for k, v in list(examples.items())[:6]:
    print("-", k[:180])
