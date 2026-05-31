from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        size = self.config.image_size
        image = image.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor.unsqueeze(0).repeat(self.config.num_tiles, 1, 1, 1)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_block = " ".join(
            [IMAGE_START_TOKEN] + [IMAGE_TOKEN] * self.config.num_image_tokens + [IMAGE_END_TOKEN]
        )
        options_text = "\n".join(sample.options)
        text = (
            f"{image_block}\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            f"Ответ:"
        )
        if include_answer:
            text = f"{text} {sample.answer}"
        return text

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt_ids = self.tokenizer(self.build_prompt(sample, include_answer=False))["input_ids"]
        full_ids = self.tokenizer(self.build_prompt(sample, include_answer=True))["input_ids"]
        full_ids = list(full_ids) + [self.tokenizer.eos_token_id]
        full_ids = full_ids[: self.config.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [self.config.ignore_index] * prompt_len + list(full_ids[prompt_len:])
        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        max_len = max(b["input_ids"].size(0) for b in batch)
        pad_id = self.tokenizer.pad_token_id
        ignore = self.config.ignore_index

        def pad(t: torch.Tensor, value: int) -> torch.Tensor:
            return F.pad(t, (0, max_len - t.size(0)), value=value)

        input_ids = torch.stack([pad(b["input_ids"], pad_id) for b in batch])
        attention_mask = torch.stack([pad(b["attention_mask"], 0) for b in batch])
        labels = torch.stack([pad(b["labels"], ignore) for b in batch])
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
