from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.pre_norm = nn.LayerNorm(vision_hidden_size)
        self.proj1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.pre_norm(vision_hidden_states)
        x = self.proj2(self.act(self.proj1(x)))
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_image_tokens).transpose(1, 2)
        return x


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    mask = input_ids == image_token_id
    out = input_embeds.clone()
    out[mask] = visual_embeds.reshape(-1, visual_embeds.shape[-1]).to(out.dtype)
    return out


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _visual_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, T = pixel_values.shape[:2]
        flat = pixel_values.flatten(0, 1)
        vout = self.vision_encoder(pixel_values=flat)
        vfeats = vout.last_hidden_state
        visual = self.adapter(vfeats)
        if T > 1:
            visual = visual.view(B, T, visual.shape[1], visual.shape[2]).mean(dim=1)
        return visual

    def _prepare_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        visual = self._visual_embeds(batch["pixel_values"])
        embeds = self.language_model.get_input_embeddings()(batch["input_ids"])
        return merge_visual_embeddings(embeds, batch["input_ids"], visual, self.config.image_token_id)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        merged = self._prepare_inputs(batch)
        out = self.language_model(
            inputs_embeds=merged,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        return {"loss": out.loss, "logits": out.logits}

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        merged = self._prepare_inputs(batch)
        return self.language_model.generate(
            inputs_embeds=merged,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )
