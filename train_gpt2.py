import argparse
import math
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.llm_model import BASE_MODEL, DEFAULT_MODEL_DIR, format_prompt

DATASET = "berkeley-nest/Nectar"
SEED = 42
MAX_TOKENS = 128


def training_examples(rows, tokenizer, count):
    examples = []
    seen_questions = set()
    for row in rows:
        if row["turns"] != 1 or not row["good_natured"]:
            continue

        question, separator, _ = row["prompt"].strip().removeprefix("Human:").rpartition(
            "\n\nAssistant:"
        )
        question = question.strip()
        if not separator or not question.endswith("?") or question in seen_questions:
            continue

        answer = min(row["answers"], key=lambda item: item["rank"])["answer"].strip()
        prompt_ids = tokenizer.encode(format_prompt(question), add_special_tokens=False, verbose=False)
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False, verbose=False)
        if not answer or len(prompt_ids) + len(answer_ids) + 1 > MAX_TOKENS:
            continue

        answer_ids.append(tokenizer.eos_token_id)
        examples.append({
            "input_ids": torch.tensor(prompt_ids + answer_ids),
            "labels": torch.tensor([-100] * len(prompt_ids) + answer_ids),
        })
        seen_questions.add(question)
        if len(examples) == count:
            return examples
    raise RuntimeError(f"Nectar yielded only {len(examples)} eligible examples.")


def collate_examples(examples, pad_token_id):
    labels = [example["labels"] for example in examples]
    input_ids = [example["input_ids"] for example in examples]
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(
            [torch.ones_like(tokens) for tokens in input_ids], batch_first=True
        ),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


@torch.inference_mode()
def test_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        tokens = (batch["labels"][:, 1:] != -100).sum().item()
        loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Nectar questions.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-examples", type=positive_int, default=256)
    parser.add_argument("--test-examples", type=positive_int, default=32)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    nectar_file = hf_hub_download(DATASET, "data/rlaif.parquet", repo_type="dataset")
    rows = load_dataset("parquet", data_files=nectar_file, split="train", streaming=True)
    rows = rows.shuffle(seed=SEED, buffer_size=2048)
    examples = training_examples(rows, tokenizer, args.train_examples + args.test_examples)
    collate = partial(collate_examples, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        examples[:args.train_examples], batch_size=args.batch_size, shuffle=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        examples[args.train_examples:], batch_size=args.batch_size,
        collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, use_safetensors=True).to(device)
    model.config.use_cache = False

    # Keep the optimizer state small enough for a classroom laptop.
    model.requires_grad_(False)
    model.transformer.h[-2:].requires_grad_(True)
    model.transformer.ln_f.requires_grad_(True)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=5e-5)
    print(f"Base test loss: {test_loss(model, test_loader, device):.4f}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
            if not math.isfinite(loss.item()):
                raise RuntimeError("Training loss is non-finite.")
            loss.backward()
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                print(f"Epoch {epoch + 1}, batch {step}/{len(train_loader)}: {loss.item():.4f}",
                      flush=True)
        print(f"Fine-tuned test loss: {test_loss(model, test_loader, device):.4f}", flush=True)

    model.config.use_cache = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        checkpoint.rename(args.output)
    print(f"Saved checkpoint to {args.output}", flush=True)


if __name__ == "__main__":
    main()
