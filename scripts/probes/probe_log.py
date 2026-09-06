import json
import re

with open("/tmp/stage1b_run2.log", encoding="utf-8") as f:
    lines = f.readlines()

evts = []
for ln in lines:
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    try:
        obj = json.loads(ln)
    except Exception:
        continue
    ts = obj.get("timestamp", "")
    lvl = obj.get("level", "")
    logger = obj.get("logger", "")
    fn = obj.get("function", "")
    msg = obj.get("message", "")
    if "generation.engine" in logger or "generation.engine" in logger:
        evts.append((ts, lvl, logger, fn, msg))
    if "batch_runner" in logger and fn == "_run_one":
        evts.append((ts, lvl, logger, fn, msg[:160]))

print("=== generation-related events ===")
for e in evts:
    print(e[0][11:19], e[1], e[3], "|", e[4][:220])

print("\n=== 413 full messages ===")
seen = set()
for e in evts:
    if "413" in e[4]:
        m = re.search(r"413[^\\\\]*", e[4])
        key = e[4][:150]
        if key not in seen:
            seen.add(key)
            print(e[0][11:19], e[4][:500])
            print("-" * 80)
