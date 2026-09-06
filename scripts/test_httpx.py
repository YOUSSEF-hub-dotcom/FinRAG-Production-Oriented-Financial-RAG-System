import httpx, asyncio
async def main():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://127.0.0.1:8000/health", timeout=10)
        print(r.status_code, r.text[:200])
asyncio.run(main())
