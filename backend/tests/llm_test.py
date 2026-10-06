import asyncio

from backend.app.config import get_settings
from backend.app.llm import get_llm


async def main():
    llm = get_llm(get_settings())
    async for chunk in llm.astream("whats the bayesein theorem?"):
        print(repr(chunk.content), chunk.usage_metadata)


asyncio.run(main())
