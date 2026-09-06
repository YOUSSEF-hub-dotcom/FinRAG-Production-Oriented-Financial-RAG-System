import asyncio, httpx, os
API_BASE = "http://127.0.0.1:8000"
EMAIL = "youssefaboali122@gmail.com"
PASSWORD = "12345abcde"

async def main():
    async with httpx.AsyncClient() as client:
        # login
        r = await client.post(f"{API_BASE}/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        print("login", r.status_code)
        if r.status_code != 200:
            print(r.text[:500])
            return
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        # health
        r = await client.get(f"{API_BASE}/health")
        print("health", r.status_code, r.json().get("status"))
        # Session A: Apple, Microsoft, NVIDIA order; first company follow-up
        sid = "test-mem-001"
        q1 = "Compare the operating margins and total net revenue between Apple, Microsoft, and NVIDIA for FY2025."
        r = await client.post(f"{API_BASE}/api/v1/chat", json={"user_query": q1, "ticker":"ALL", "fiscal_year":"2025", "session_id": sid}, headers=headers)
        print("Q1", r.status_code, r.json().get("answer","")[:120].replace("\n"," ")[:100])
        # follow-up first company
        q2 = "How much did the first company spend on Research and Development in that same fiscal year?"
        r = await client.post(f"{API_BASE}/api/v1/chat", json={"user_query": q2, "ticker":"ALL", "session_id": sid}, headers=headers)
        ans2 = r.json().get("answer","")
        print("Q2 first company:", ans2[:150].replace("\n"," ")[:120], "model", r.json().get("model_used"))
        # check if answer contains Apple and 34550/34,550
        has_apple = "Apple" in ans2 or "AAPL" in ans2
        has_34550 = "34,550" in ans2 or "34550" in ans2
        print(f"Q2 has_apple={has_apple} has_34550={has_34550}")
        # second company
        q3 = "How much did the second company spend on Research and Development?"
        r = await client.post(f"{API_BASE}/api/v1/chat", json={"user_query": q3, "ticker":"ALL", "session_id": sid}, headers=headers)
        ans3 = r.json().get("answer","")
        print("Q3 second:", ans3[:150].replace("\n"," ")[:120], "model", r.json().get("model_used"))
        has_msft = "Microsoft" in ans3 or "MSFT" in ans3
        has_32488 = "32,488" in ans3 or "32488" in ans3
        print(f"Q3 has_msft={has_msft} has_32488={has_32488}")
        # Session C: its revenue after Apple net income
        sid2 = "test-mem-002"
        r = await client.post(f"{API_BASE}/api/v1/chat", json={"user_query": "What is Apple's net income for FY2025?", "ticker":"AAPL","fiscal_year":"2025","session_id": sid2}, headers=headers)
        print("Q4a", r.json().get("answer","")[:100])
        r = await client.post(f"{API_BASE}/api/v1/chat", json={"user_query": "What is its revenue?", "ticker":"AAPL","session_id": sid2}, headers=headers)
        ans = r.json().get("answer","")
        print("Q4b its revenue:", ans[:150], "has 416,161?", "416,161" in ans or "416161" in ans)

asyncio.run(main())
