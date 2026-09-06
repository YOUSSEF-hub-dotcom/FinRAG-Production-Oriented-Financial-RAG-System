import re
from pathlib import Path

filings = [
    ("AAPL", "2025", "data/AAPL/10-K/0000320193-25-000079/full-submission.txt"),
    ("MSFT", "2025", "data/MSFT/10-K/0000950170-25-100235/full-submission.txt"),
    ("NVDA", "2026", "data/NVDA/10-K/0001045810-26-000021/full-submission.txt"),
]

for tick, yr, path in filings:
    p = Path(path)
    if not p.exists():
        print(tick, yr, "MISSING", path)
        continue
    text = p.read_text(errors="replace")
    for num in ("416161", "281724", "215938", "416,161", "281,724", "215,938"):
        idx = text.find(num)
        if idx >= 0:
            print(f"{tick} {yr}: FOUND {num} at {idx}")
            print("   context:", re.sub(r"\s+", " ", text[max(0,idx-80):idx+80]))
        else:
            print(f"{tick} {yr}: NOT FOUND {num}")
