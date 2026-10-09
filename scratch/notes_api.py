"""A tiny question-answer API. Same shape as the real /chat: JSON in, JSON out.

Run:   uv run uvicorn scratch.notes_api:app --reload
Docs:  http://127.0.0.1:8000/docs   (click "Try it out" on any route)
"""

import asyncio

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.config import get_settings
from backend.app.llm import get_llm

app = FastAPI()


# --- the shapes of data going in and out (Pydantic validates them for you) ---------------


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Answer(BaseModel):
    question: str
    answer: str


# remembers answers while the server runs; resets on restart
history: list[Answer] = []


@app.post("/ask/llm", response_model=Answer)
async def invoke_llm(body: Question) -> Answer:
    llm = get_llm(get_settings())
    reply = await llm.ainvoke(
        body.question
    )  # a plain string is sent as one user message
    text = reply.text  # ainvoke returns an AIMessage object; .text is the plain string
    answer = Answer(question=body.question, answer=text)
    history.append(answer)  # same as /ask
    return answer


# --- stand-in for a real LLM call: later this becomes stream_chat / get_llm -------------


def fake_llm(question: str) -> str:
    return f"You asked: {question}"


# --- routes: a decorator says which method + path, the function says what to do ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:

    answer = Answer(question=body.question, answer=fake_llm(body.question))
    history.append(answer)
    return answer


@app.get("/history", response_model=list[Answer])
def get_history(limit: int | None = Query(default=None, ge=1)) -> list[Answer]:
    if limit is None:
        return history
    return history[-limit:]


@app.get("/history/{n}", response_model=Answer)
def get_message(n: int) -> Answer:
    if n >= len(history) or n < 0:
        raise HTTPException(status_code=404, detail="message doesn't exist in history")

    return history[n]


@app.delete("/history", status_code=204)
def delete_history():
    history.clear()


@app.post("/ask/stream")
def stream(body: Question) -> StreamingResponse:
    async def lines():

        for word in fake_llm(body.question).split():
            yield f"data: {word}\n\n"
            await asyncio.sleep(0.2)
        yield "data: [DONE]\n\n"

    return StreamingResponse(lines(), media_type="text/event-stream")


# ==========================================================================================
# EXERCISES. Do them in order, one at a time. Docs open is fine; no AI.
# For each: remove the @exercise line above its test in test_notes_api.py, run
# `uv run pytest scratch -v`, watch it FAIL, then write code until it passes.
#
# 1. Validation.  An empty question ("") must be rejected with 422, and so must a question
#                 longer than 500 characters.
#                 Hint: Pydantic's Field(min_length=..., max_length=...) on the model.
#
# 2. Path parameter.  GET /history/{n} returns the n-th answer (0 is the first).
#                 If n is out of range, return 404 with a message.
#                 Hint: HTTPException(status_code=404, detail="...").
#
# 3. Query parameter.  GET /history?limit=2 returns only the last 2 answers.
#                 Without limit, it returns everything. limit=0 must give 422.
#                 Hint: a function parameter with a default is a query parameter;
#                 Query(ge=1) adds a rule.
#
# 4. Delete.      DELETE /history empties the history and returns 204 (no body).
#                 Hint: @app.delete(..., status_code=204); history.clear().
#
# 5. Streaming.   POST /ask/stream takes the same body as /ask and streams the answer
#                 one word at a time as Server-Sent Events:
#                     data: You\n\n
#                     data: asked:\n\n
#                     data: hi\n\n
#                     data: [DONE]\n\n
#                 This is exactly the shape the real /chat route will have.
#                 Hint: StreamingResponse(generator, media_type="text/event-stream"),
#                 where the generator is a function that `yield`s each line.
#
# 6. Real model.  Replace fake_llm with a call to your get_llm(...). Remember the route
#                 must now be `async def` and use `await llm.ainvoke(...)`.
# ==========================================================================================
