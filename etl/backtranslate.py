"""Backtranslate Adam's messages into (prompt, reply) pairs for fine-tuning.

Reads  data/clean/messages.jsonl          one row per burst of Adam's messages
Writes data/backtranslated/pairs.jsonl    {"row", "channel", "type", "time", "text", "prompt"}, where
                                          "prompt" is the friend message Claude wrote for Adam's
                                          real "text" to reply to
       data/backtranslated/failed.jsonl   {"row", "text", "stop_reason"} for rows Claude refused or cut off

"row" is the row's line number in messages.jsonl. Rows already in either output are skipped,
so rerunning after an interruption continues where it stopped. Rerunning clean.py renumbers
the rows, so delete data/backtranslated before backtranslating its new output.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic, transform_schema
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "clean" / "messages.jsonl"
OUT = DATA / "backtranslated"
PAIRS = OUT / "pairs.jsonl"
FAILED = OUT / "failed.jsonl"

PROMPT = (Path(__file__).parent / "backtranslate_prompt.md").read_text()

client = AsyncAnthropic(
    default_headers={"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]},
    max_retries=5,  # an hours-long run will hit rate limits, so back off longer before giving up
)
concurrency = asyncio.Semaphore(20)


class Context(BaseModel):
    prompt: str  # the friend message that Adam's text replies to


SCHEMA = transform_schema(Context)


def load_data(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def finished_rows() -> set[int]:
    """Row numbers that an earlier run already wrote to PAIRS or FAILED."""
    finished = set()
    for path in (PAIRS, FAILED):
        if path.exists():
            with open(path) as f:
                for line in f:
                    finished.add(json.loads(line)["row"])

    return finished


def append(path: Path, record: dict) -> None:
    """Append one JSON line right away, so an interrupted run keeps every finished row."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def backtranslate(i: int, row: dict) -> None:
    """Ask Claude for the message that row `i` replies to, and append the result to PAIRS or FAILED."""
    async with concurrency:
        message = await client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=2_000,
            # Every request shares this prompt, so cache it: after the first request it's billed
            # at a fraction of the normal input price.
            system=[{"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": row["text"]}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )

    # A refused or cut-off reply holds partial JSON, so only parse finished ones. This is why we
    # don't use messages.parse(): it raises on partial JSON, which would stop the whole run.
    if message.stop_reason != "end_turn":
        append(FAILED, {"row": i, "text": row["text"], "stop_reason": message.stop_reason})
        return

    # Haiku decides per request whether to think before answering, so the reply can open with a
    # thinking block. The JSON is always the last block.
    block = message.content[-1]
    assert block.type == "text"

    context = Context.model_validate_json(block.text)
    append(PAIRS, {"row": i, **row, "prompt": context.prompt})


async def main(limit: int | None) -> None:
    OUT.mkdir(exist_ok=True)

    rows = load_data(SOURCE)
    finished = finished_rows()

    todo = []
    for i, row in enumerate(rows):
        # Empty rows are bursts of images or removed messages: there's nothing to reply to,
        # and the API rejects empty messages.
        if i not in finished and row["text"].strip():
            todo.append((i, row))

    if limit is not None:
        todo = todo[:limit]

    print(f"{len(finished)} rows done by earlier runs, backtranslating {len(todo)}")

    calls = []
    for i, row in todo:
        calls.append(backtranslate(i, row))

    for count, call in enumerate(asyncio.as_completed(calls), start=1):
        await call

        if count % 1000 == 0 or count == len(calls):
            print(f"{count}/{len(calls)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtranslate Adam's messages into training pairs.")
    parser.add_argument("--limit", type=int, help="only backtranslate the next N rows, for a test run")
    args = parser.parse_args()

    asyncio.run(main(args.limit))
