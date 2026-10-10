"""Build the fine-tuning dataset from the backtranslated pairs, with a fixed 5% holdout.

Reads  data/backtranslated/pairs.jsonl   {"row", "channel", "type", "time", "text", "prompt"}
Writes data/sft/train.jsonl              about 95% of the pairs
       data/sft/holdout.jsonl            about 5%, kept out of training for evaluation

Each line is one conversation in the "messages" format that TRL's SFTTrainer, and so Unsloth,
reads directly. The friend message Claude wrote is the user turn, and Adam's real message is
the assistant turn the model learns to write:

    {"messages": [{"role": "user", "content": "how much sleep did u get"},
                  {"role": "assistant", "content": "like 3 hours i only fell asleep at 4"}]}

There is no system message, so at inference send a friend's message as a lone user turn to
match training. Load both splits with:

    from datasets import load_dataset
    dataset = load_dataset("json", data_files={"train": "data/sft/train.jsonl", "holdout": "data/sft/holdout.jsonl"})
"""

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "backtranslated" / "pairs.jsonl"
OUT = DATA / "sft"

HOLDOUT_FRACTION = 0.05


def to_messages(pair: dict) -> list[dict]:
    """The conversation the model trains on: a friend's message, then Adam's reply."""
    return [
        {"role": "user", "content": pair["prompt"]},
        {"role": "assistant", "content": pair["text"]},
    ]


def in_holdout(pair: dict, fraction: float = HOLDOUT_FRACTION) -> bool:
    """Whether a pair belongs to the holdout rather than training.

    The choice hashes the message's channel and time instead of drawing randomly, so every
    rebuild puts each message in the same split: rerunning clean.py renumbers rows but keeps
    these. New messages don't move old ones either, so holdout messages never leak into training.
    sha256 rather than hash(), which Python salts differently in every process.
    """
    key = f"{pair['channel']}/{pair['time']}".encode()
    position = int.from_bytes(hashlib.sha256(key).digest()[:8]) / 2**64  # uniform in [0, 1)

    return position < fraction


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    with open(SOURCE) as f:
        pairs = [json.loads(line) for line in f]

    # Backtranslation writes rows in the order they finish, so sort to make rebuilds identical.
    pairs.sort(key=lambda pair: pair["row"])

    train, holdout = [], []
    for pair in pairs:
        split = holdout if in_holdout(pair) else train
        split.append({"messages": to_messages(pair)})

    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "holdout.jsonl", holdout)

    print(f"{len(train)} train, {len(holdout)} holdout ({len(holdout) / len(pairs):.1%})")


if __name__ == "__main__":
    main()
