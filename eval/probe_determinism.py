"""Does this model give the same answer to the same question? Test it before you trust it.

    .venv/Scripts/python eval/probe_determinism.py MODEL [MODEL ...] [--n 8]

Why this exists: setting temperature=0 is a request, not a guarantee. Some models ignore it (the
endpoint lists no `temperature` parameter at all) and answer differently every time, which makes
"temperature 0" in the code a comforting lie. This sends the same open-ended prompt N times at
temperature 0 and counts the distinct answers. 1 of N is deterministic; N of N means the setting is
being ignored. It costs a fraction of a cent per model. Reads the key from the app settings and never
prints it.
"""

import argparse
import asyncio
import collections
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = "Write one sentence of exactly twelve words about the sea. Reply with the sentence only."


async def ask(
    client: httpx.AsyncClient, model: str, seed: int | None
) -> tuple[str, str, float]:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 80,
        "temperature": 0,
    }
    if seed is not None:
        body["seed"] = seed
    response = (await client.post(URL, json=body)).json()
    if "choices" not in response:
        raise RuntimeError(str(response.get("error", response))[:120])
    text = (
        response["choices"][0]["message"].get("content") or ""
    ).strip()  # may be empty
    return (
        text,
        str(response.get("provider")),
        float(response.get("usage", {}).get("cost") or 0),
    )


async def probe(
    client: httpx.AsyncClient, model: str, n: int, seed: int | None
) -> None:
    try:
        results = await asyncio.gather(*[ask(client, model, seed) for _ in range(n)])
    except RuntimeError as exc:
        print(f"{model:44s} ERROR {exc}")
        return
    distinct = len({text for text, _, _ in results})
    providers = dict(collections.Counter(p for _, p, _ in results))
    if distinct == 1:
        verdict = "deterministic"
    elif distinct <= n // 2:
        verdict = "partly stable"
    else:
        verdict = "IGNORES temperature 0"
    cost = sum(c for _, _, c in results)
    print(
        f"{model:44s} {distinct}/{n} distinct | {verdict:22s} | providers {providers} | ${cost:.6f}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    key = get_settings().openrouter_api_key
    if not key:
        sys.exit("no OPENROUTER_API_KEY in the settings")
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"}, timeout=90
    ) as client:
        for model in args.models:
            await probe(client, model, args.n, args.seed)


if __name__ == "__main__":
    asyncio.run(main())
