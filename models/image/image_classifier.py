from __future__ import annotations

import torch
from torch import nn

from models.gnn.object_graph import FeatureSlicer, TrustifyGNN
from models.image.resnet_features import ResNet50AvgPoolExtractor
from models.transformer.captioner import GNNEmbeddedTransformerCaptioner


class TrustifyImageBranch(nn.Module):
    """ResNet-50 -> feature slicing -> GNN -> 3x3 Transformer -> image confidence."""

    def __init__(
        self,
        vocab_size: int,
        avgpool_dim: int,
        graph_nodes: int,
        node_feature_dim: int,
        transformer_heads: int,
        transformer_ff_dim: int,
        transformer_dropout: float,
        max_caption_length: int,
        freeze_resnet: bool = True,
    ):
        super().__init__()
        self.extractor = ResNet50AvgPoolExtractor(pretrained=True, freeze=freeze_resnet)
        self.slicer = FeatureSlicer(avgpool_dim, graph_nodes, node_feature_dim)
        self.gnn = TrustifyGNN(node_feature_dim)
        self.captioner = GNNEmbeddedTransformerCaptioner(
            vocab_size=vocab_size,
            d_model=node_feature_dim,
            nhead=transformer_heads,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
        )
        self.image_classifier = nn.Sequential(nn.Linear(node_feature_dim, 1), nn.Sigmoid())
        self.max_caption_length = max_caption_length

    def forward(self, images: torch.Tensor, caption_input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.extractor(images)
        vertices = self.slicer(features)
        gnn_nodes, adjacency = self.gnn(vertices)
        caption_logits, memory = self.captioner(gnn_nodes, caption_input_ids)
        image_confidence = self.image_classifier(memory.mean(dim=1)).squeeze(-1)
        return {
            "image_confidence": image_confidence,
            "caption_logits": caption_logits,
            "adjacency": adjacency,
            "gnn_nodes": gnn_nodes,
        }

