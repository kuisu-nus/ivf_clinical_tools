import asyncio
from e2b import AsyncSandbox

async def main():
    p = AsyncSandbox.list(limit=50)   # 注意：返回 paginator，不能 await
    for s in await p.next_items():
        print(s.sandbox_id, s.state, s.metadata)

asyncio.run(main())
