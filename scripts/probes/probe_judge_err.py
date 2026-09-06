import json

with open("/tmp/stage1c_run.log", encoding="utf-8") as f:
    lines = f.readlines()

from collections import Counter
c = Counter()
details = Counter()
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
        err = rest.split(": ", 1)[1][:120] if ": " in rest else rest[:120]
        c[model] += 1
        details[err] += 1
    elif "Judge evaluation failed" in msg:
        details["FINAL RUNTIME FAILURE"] += 1
        print("RUNTIME FAILURE:", msg[:200])
print("failed-with counts:", dict(c))
print("\nerror details:")
for k, v in details.most_common(12):
    print(" ", v, k)
