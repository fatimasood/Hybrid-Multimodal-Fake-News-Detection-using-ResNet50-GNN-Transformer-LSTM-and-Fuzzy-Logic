from __future__ import annotations

import torch
from torch import nn


class FeatureSlicer(nn.Module):
    """Slices the ResNet avgpool vector into K graph nodes, matching the paper text."""

    def __init__(self, total_dim: int, num_nodes: int, node_dim: int):
        super().__init__()
        if num_nodes * node_dim != total_dim:
            raise ValueError("num_nodes * node_dim must equal total_dim for strict avgpool slicing")
        self.num_nodes = num_nodes
        self.node_dim = node_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, 2048]; output vertices: [batch, K, d]
        return features.view(features.shape[0], self.num_nodes, self.node_dim)


class TrustifyGNN(nn.Module):
    """Implements the paper's node transform, learned adjacency, and message passing equations."""

    def __init__(self, node_dim: int, message_steps: int = 1):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, node_dim)
        self.message_transform = nn.Linear(node_dim, node_dim)
        self.relation_conv1 = nn.Conv1d(node_dim, 1, kernel_size=1)
        self.message_steps = message_steps

    def learned_adjacency(self, vertices: torch.Tensor) -> torch.Tensor:
        # vertices: [batch, nodes, node_dim]
        diff = torch.abs(vertices.unsqueeze(2) - vertices.unsqueeze(1))
        batch, nodes, _, dim = diff.shape
        relation_input = diff.reshape(batch * nodes * nodes, dim, 1)
        scores = self.relation_conv1(relation_input).reshape(batch, nodes, nodes)
        adjacency = torch.sigmoid(scores)
        eye = torch.eye(nodes, device=vertices.device).unsqueeze(0)
        return adjacency * (1.0 - eye)

    def forward(self, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. 3: vertex_a = tanh(Weight_a * vertex_a + bias_a)
        vertices = torch.tanh(self.node_transform(vertices))
        # h_a^0 = ReLU(vertex_a)
        hidden = torch.relu(vertices)
        adjacency = self.learned_adjacency(vertices)
        for _ in range(self.message_steps):
            messages = self.message_transform(hidden)
            aggregated = torch.bmm(adjacency, messages)
            hidden = torch.relu(aggregated)
        return hidden, adjacency

