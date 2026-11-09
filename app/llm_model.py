import os
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "openai-community/gpt2"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "gpt2-nectar"


def format_prompt(question):
    return f"\n\nHuman: {question.strip()}\n\nAssistant:"


class LLMTextGenerator:
    def __init__(self):
        self._lock = Lock()
        self._checkpoint = None

    def _load_checkpoint(self):
        with self._lock:
            if self._checkpoint is None:
                model_dir = Path(os.getenv("LLM_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
                if not model_dir.is_dir():
                    raise FileNotFoundError(
                        "GPT-2 checkpoint is missing. Run train_gpt2.py or set LLM_MODEL_DIR."
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
                self._checkpoint = (tokenizer, model.eval())
            return self._checkpoint

    @torch.inference_mode()
    def generate_text(self, question, max_new_tokens):
        tokenizer, model = self._load_checkpoint()
        inputs = tokenizer(format_prompt(question), return_tensors="pt")
        prompt_length = inputs["input_ids"].size(1)
        if prompt_length + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("The question and requested answer exceed GPT-2's token limit.")

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        return answer.split("\n\nHuman:", 1)[0].strip()
