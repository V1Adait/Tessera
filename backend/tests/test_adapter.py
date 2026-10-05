from backend.app.adapter import FakeAdapter
from backend.app.models import ChatRequest,  Done, Delta

async def test_fake_adapter_streams():
    req = ChatRequest(messages=[{"role":"user", "content":"Hello"}])
    events = [e async for e in FakeAdapter().stream(req)]

    

    assert isinstance(events[-1], Done)

    assert all(isinstance(e,Delta) for e in events[:-1])

    assert "".join(e.text for e in events[:-1]).strip() == FakeAdapter.REPLY

    done = events[-1]
    assert done.usage.tokens_out == len(FakeAdapter.REPLY.split())
    

    

