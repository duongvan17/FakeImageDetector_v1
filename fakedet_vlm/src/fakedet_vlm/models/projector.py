"""2-layer MLP projector mapping ViT-B patch tokens (768) to LLM space (1536)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProjectorMLP(nn.Module):
    """Linear → GELU → Linear with pre-activation LayerNorm on input.

    LayerNorm on the vision side stabilises training when the encoder was
    fine-tuned with a different objective (binary classification) than the LLM
    expects. Output dtype follows the LLM (cast in the VLM forward).
    """

    def __init__(self, vision_dim: int = 768, llm_dim: int = 1536) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, vision_dim) -> (B, N, llm_dim)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
