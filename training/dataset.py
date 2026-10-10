"""Render the SFT conversations with Qwen3's chat template for training and inference.

Reads data/sft/{train,holdout}.jsonl, written by etl/build_sft.py.

This module only needs a tokenizer, not a GPU, so the text the model trains on can be
built and checked anywhere. train.py, which needs CUDA, imports it.
"""

import json
from pathlib import Path

from datasets import Dataset

SFT = Path(__file__).resolve().parents[1] / "data" / "sft"

# Covers 99.9% of conversations. The few longer ones are dropped rather than truncated,
# since a truncated reply would teach the model to stop mid-message.
MAX_SEQ_LENGTH = 512

# Where Qwen3's template starts each turn. Training scores only the tokens after
# REPLY_START, so the model learns Adam's replies but not the friend messages or the
# empty thinking block the template puts before every reply.
USER_START = "<|im_start|>user\n"
REPLY_START = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def read(split: str) -> list[list[dict]]:
    """The conversations in data/sft/<split>.jsonl, each a [friend, Adam] message list."""
    with open(SFT / f"{split}.jsonl") as f:
        return [json.loads(line)["messages"] for line in f]


def generation_prompt(tokenizer, friend_message: str) -> str:
    """The text the model continues to write Adam's reply to a friend's message."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": friend_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def training_text(tokenizer, messages: list[dict]) -> str:
    """A whole conversation as the model trains on it: generation_prompt() plus the reply.

    The template ends every turn with "<|im_end|>\\n". The text stops at "<|im_end|>",
    the token generation stops on, because SFTTrainer appends that token to any text not
    already ending in it, which would teach the model a doubled stop token.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)

    return text.removesuffix("\n")


def training_rows(tokenizer, conversations: list[list[dict]]) -> Dataset:
    """Conversations as a "text" dataset for SFTTrainer, without any over MAX_SEQ_LENGTH tokens.

    The dataset has only a "text" column on purpose: given a "messages" column,
    SFTTrainer would apply the chat template itself, without enable_thinking=False.
    """
    texts = []
    for messages in conversations:
        text = training_text(tokenizer, messages)
        if len(tokenizer(text)["input_ids"]) <= MAX_SEQ_LENGTH:
            texts.append(text)

    dropped = len(conversations) - len(texts)
    print(f"kept {len(texts)} conversations, dropped {dropped} over {MAX_SEQ_LENGTH} tokens")

    return Dataset.from_dict({"text": texts})
