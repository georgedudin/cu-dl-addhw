from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    out = model(batch)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.item())


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k == "pixel_values":
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device)
    return out


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
    from hw.dataset import MathVQADataset
    from hw.model import MathVLM, ModelConfig
    from hw.processor import MathVLMProcessor, ProcessorConfig

    tcfg = config["trainer"]
    mcfg = config["model"]
    pcfg = config["processor"]
    dcfg = config["data"]

    device = torch.device(tcfg["device"])
    dtype = _DTYPES[tcfg["dtype"]]

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
    if mcfg.get("freeze_vision", True) and mcfg.get("freeze_llm", True):
        model.freeze_backbones()
    model.to(device)
    model.adapter.to(torch.float32)

    processor = MathVLMProcessor(tokenizer, ProcessorConfig(**pcfg))
    dataset = MathVQADataset(
        dcfg["train_manifest"],
        split=dcfg.get("split", "train"),
        max_samples=dcfg.get("max_samples"),
    )

    def collate(samples):
        return processor.collate([processor(s) for s in samples])

    loader = DataLoader(
        dataset,
        batch_size=tcfg["local_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=tcfg.get("num_workers", 0),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    max_steps = 3 if fast_train else int(tcfg["max_steps"])
    pbar = tqdm(total=max_steps, desc="train")
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = _move_batch(batch, device, dtype)
            loss = train_one_step(model, batch, optimizer)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}")
    pbar.close()

    save_path = tcfg.get("save_checkpoint_path")
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.adapter.state_dict(), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
