import asyncio
from backend.app.adapter import FakeAdapter
from backend.app.models import ChatRequest


async def main():
    req = ChatRequest(messages=[{"role": "user", "content": "hi there"}])
    async for event in FakeAdapter().stream(req):
        print(event)


asyncio.run(main())
