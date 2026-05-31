from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    pattern = rf"\b([{''.join(choices)}])\b"
    m = re.search(pattern, text)
    return m.group(1) if m else None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
    from hw.dataset import MathVQADataset
    from hw.model import MathVLM, ModelConfig
    from hw.processor import MathVLMProcessor, ProcessorConfig

    mcfg = config["model"]
    pcfg = config["processor"]
    dcfg = config["data"]
    icfg = config["inference"]

    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    device = torch.device(icfg["device"])
    dtype = dtypes[icfg["dtype"]]

    tokenizer = AutoTokenizer.from_pretrained(mcfg["language_model"])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision = AutoModel.from_pretrained(mcfg["vision_encoder"])
    llm = AutoModelForCausalLM.from_pretrained(mcfg["language_model"], torch_dtype=dtype)
    llm.resize_token_embeddings(len(tokenizer))

    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    model = MathVLM(
        vision_encoder=vision,
        language_model=llm,
        config=ModelConfig(
            vision_hidden_size=vision.config.hidden_size,
            text_hidden_size=llm.config.hidden_size,
            num_image_tokens=pcfg["num_image_tokens"],
            image_token_id=image_token_id,
        ),
    )
    model.freeze_backbones()
    model.to(device)
    model.adapter.to(torch.float32)

    adapter_path = mcfg.get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        state = torch.load(adapter_path, map_location=device)
        model.adapter.load_state_dict(state)

    model.eval()

    processor = MathVLMProcessor(tokenizer, ProcessorConfig(**pcfg))
    manifest = dcfg["eval_manifest"]
    if toy:
        manifest = "assets/toy_math_vqa/manifest.jsonl"
    dataset = MathVQADataset(
        manifest,
        split=dcfg.get("split", "dev"),
        max_samples=dcfg.get("max_samples"),
    )

    rows: list[dict[str, Any]] = []
    for sample in tqdm(dataset, desc="benchmark"):
        item = processor(sample)
        batch = {
            "input_ids": item["input_ids"].unsqueeze(0).to(device),
            "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
            "pixel_values": item["pixel_values"].unsqueeze(0).to(device=device, dtype=dtype),
        }
        ids = model.generate(
            batch,
            max_new_tokens=int(icfg["max_new_tokens"]),
            do_sample=bool(icfg.get("do_sample", False)),
        )
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        pred = parse_mc_answer(text) or ""
        rows.append(
            {
                "id": sample.id,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": pred,
                "subject": sample.subject,
                "raw": text,
            }
        )

    output_path = icfg.get("output_path")
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
