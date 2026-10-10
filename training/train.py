"""Fine-tune Qwen3-8B with a LoRA adapter so it texts like Adam.

Reads  data/sft/{train,holdout}.jsonl   written by etl/build_sft.py
Writes data/lora/<run>/adapter/          the LoRA adapter and tokenizer, where <run> is
                                         "smoke" or "full"

Needs a CUDA GPU, so it runs on Runpod rather than a Mac:

    pip install -r training/requirements.txt
    python training/train.py --smoke   # 50 steps, to check the setup before a full run
    python training/train.py           # one epoch over the training split

Both runs check that training scores only Adam's replies, report loss on a sample of the
holdout, and print replies to a few holdout messages.
"""

import argparse
from pathlib import Path

# Unsloth patches transformers and trl when imported, so it has to come first.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

# isort: split
import dataset
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/Qwen3-8B"
OUT = Path(__file__).resolve().parents[1] / "data" / "lora"

# Unsloth's recommended starting points for LoRA on all linear layers.
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2  # an effective batch of 16 conversations

# Holdout conversations scored during training. The full holdout is for the final evaluation.
EVAL_SIZE = 500
EVAL_STEPS = 250

SMOKE_STEPS = 50
SEED = 3407


def load_model():
    """Qwen3-8B in 16-bit with trainable LoRA adapters on every linear layer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=dataset.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    return model, tokenizer


def build_trainer(model, tokenizer, smoke: bool, output_dir: Path) -> SFTTrainer:
    """An SFTTrainer over the training split that scores only Adam's replies."""
    train = dataset.training_rows(tokenizer, dataset.read("train"))
    holdout = dataset.training_rows(tokenizer, dataset.read("holdout"))
    holdout = holdout.shuffle(seed=SEED).select(range(EVAL_SIZE))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=holdout,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=dataset.MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=1,
            max_steps=SMOKE_STEPS if smoke else -1,  # -1 trains for num_train_epochs
            warmup_ratio=0.05,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            optim="adamw_8bit",
            logging_steps=1 if smoke else 10,
            eval_strategy="steps",
            eval_steps=SMOKE_STEPS // 2 if smoke else EVAL_STEPS,
            save_strategy="no",  # a full run takes about an hour, so rerun rather than resume
            seed=SEED,
            report_to="none",
            output_dir=str(output_dir / "trainer"),
        ),
    )

    # Mask the friend's turn and the template scaffolding out of the loss, on both splits.
    return train_on_responses_only(
        trainer,
        instruction_part=dataset.USER_START,
        response_part=dataset.REPLY_START,
    )


def check_masking(trainer: SFTTrainer, tokenizer, rows: int = 100) -> None:
    """Fail before training unless the scored tokens are exactly each reply and its stop token."""
    for row in trainer.train_dataset.select(range(rows)):
        text = tokenizer.decode(row["input_ids"])
        reply = text.split(dataset.REPLY_START, 1)[1]
        scored = tokenizer.decode([token for token in row["labels"] if token != -100])

        if scored != reply:
            raise RuntimeError(f"training would score {scored!r} instead of the reply {reply!r}")

    print(f"masking ok: the first {rows} rows score only Adam's reply, e.g. {reply!r}")


def sample_replies(model, tokenizer, count: int = 8) -> None:
    """Print the model's replies next to Adam's real ones for a few holdout messages."""
    FastLanguageModel.for_inference(model)

    for friend, adam in dataset.read("holdout")[:count]:
        prompt = dataset.generation_prompt(tokenizer, friend["content"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Qwen's recommended sampling settings with thinking off.
        output = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.8, top_k=20)
        reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        print(f"friend: {friend['content']}\n  adam:  {adam['content']}\n  model: {reply.strip()}\n")


def main(smoke: bool) -> None:
    output_dir = OUT / ("smoke" if smoke else "full")

    model, tokenizer = load_model()
    trainer = build_trainer(model, tokenizer, smoke, output_dir)
    check_masking(trainer, tokenizer)

    trainer.train()

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    print(f"saved the adapter to {output_dir / 'adapter'}")

    sample_replies(model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B to text like Adam.")
    parser.add_argument("--smoke", action="store_true", help=f"train {SMOKE_STEPS} steps to check the setup")
    args = parser.parse_args()

    main(args.smoke)
