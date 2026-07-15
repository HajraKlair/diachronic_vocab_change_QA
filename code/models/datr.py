"""
DATR Model Architecture.

Diachronic-Aware Temporal Retrieval model with:
1. Bi-encoder architecture (separate query and passage encoders)
2. Era embeddings for temporal conditioning
3. Support for contrastive diachronic alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoConfig


class EraEmbedding(nn.Module):
    """
    Learnable era embeddings for temporal conditioning.
    """
    
    def __init__(
        self,
        num_eras: int,
        embedding_dim: int,
        include_unknown: bool = True
    ):
        """
        Args:
            num_eras: Number of distinct eras (e.g., 12 decades)
            embedding_dim: Dimension of era embeddings
            include_unknown: Whether to include an "unknown" era embedding
        """
        super().__init__()
        
        total_eras = num_eras + 1 if include_unknown else num_eras
        self.embeddings = nn.Embedding(total_eras, embedding_dim)
        self.unknown_idx = num_eras if include_unknown else None
        
        # Initialize with small values
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
    
    def forward(self, era_ids: torch.Tensor) -> torch.Tensor:
        """
        Get era embeddings.
        
        Args:
            era_ids: Tensor of era indices [batch_size]
            
        Returns:
            Era embeddings [batch_size, embedding_dim]
        """
        # Handle -1 (unknown) indices
        if self.unknown_idx is not None:
            era_ids = era_ids.clone()
            era_ids[era_ids < 0] = self.unknown_idx
        
        return self.embeddings(era_ids)


class DATREncoder(nn.Module):
    """
    DATR encoder that combines text encoding with era conditioning.
    """
    
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        era_embedding_dim: int = 64,
        num_eras: int = 12,
        pooling: str = "cls",
        normalize: bool = True
    ):
        """
        Args:
            encoder_name: HuggingFace model name
            era_embedding_dim: Dimension of era embeddings
            num_eras: Number of distinct eras
            pooling: Pooling strategy ("cls" or "mean")
            normalize: Whether to L2-normalize output embeddings
        """
        super().__init__()
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Era embedding
        self.era_embedding = EraEmbedding(num_eras, era_embedding_dim)
        
        # Projection to combine text and era representations
        self.projection = nn.Linear(
            self.hidden_size + era_embedding_dim,
            self.hidden_size
        )
        
        self.pooling = pooling
        self.normalize = normalize
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states to get sequence representation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        era_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text with optional era conditioning.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            era_ids: [batch_size] - era indices, None for no era conditioning
            
        Returns:
            Encoded representations [batch_size, hidden_size]
        """
        # Get text representations
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        
        # Add era conditioning if provided
        if era_ids is not None:
            era_emb = self.era_embedding(era_ids)
            combined = torch.cat([pooled, era_emb], dim=-1)
            pooled = self.projection(combined)
        
        # Layer norm
        pooled = self.layer_norm(pooled)
        
        # Normalize if requested
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled


class DATR(nn.Module):
    """
    Full DATR model with query and passage encoders.
    
    Supports:
    - Standard retrieval (query -> passage)
    - Diachronic alignment (modern query <-> historical query)
    """
    
    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        era_embedding_dim: int = 64,
        num_eras: int = 12,
        pooling: str = "cls",
        normalize: bool = True,
        shared_encoder: bool = False
    ):
        """
        Args:
            encoder_name: HuggingFace model name
            era_embedding_dim: Dimension of era embeddings
            num_eras: Number of distinct eras
            pooling: Pooling strategy
            normalize: Whether to L2-normalize embeddings
            shared_encoder: Whether query and passage share the same encoder
        """
        super().__init__()
        
        self.query_encoder = DATREncoder(
            encoder_name=encoder_name,
            era_embedding_dim=era_embedding_dim,
            num_eras=num_eras,
            pooling=pooling,
            normalize=normalize
        )
        
        if shared_encoder:
            self.passage_encoder = self.query_encoder
        else:
            self.passage_encoder = DATREncoder(
                encoder_name=encoder_name,
                era_embedding_dim=era_embedding_dim,
                num_eras=num_eras,
                pooling=pooling,
                normalize=normalize
            )
        
        self.shared_encoder = shared_encoder
    
    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        era_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode queries."""
        return self.query_encoder(input_ids, attention_mask, era_ids)
    
    def encode_passages(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        era_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode passages."""
        return self.passage_encoder(input_ids, attention_mask, era_ids)
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
        query_era_ids: Optional[torch.Tensor] = None,
        passage_era_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning query and passage embeddings.
        
        Returns:
            Tuple of (query_embeddings, passage_embeddings)
        """
        query_emb = self.encode_queries(
            query_input_ids, query_attention_mask, query_era_ids
        )
        
        passage_emb = self.encode_passages(
            passage_input_ids, passage_attention_mask, passage_era_ids
        )
        
        return query_emb, passage_emb
    
    def compute_similarity(
        self,
        query_emb: torch.Tensor,
        passage_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between queries and passages.
        
        Args:
            query_emb: [batch_size, hidden_size]
            passage_emb: [num_passages, hidden_size]
            
        Returns:
            Similarity scores [batch_size, num_passages]
        """
        # Dot product similarity (embeddings are already normalized)
        return torch.matmul(query_emb, passage_emb.transpose(0, 1))
    
    def save_pretrained(self, save_path: str):
        """Save model weights and config."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
        
        # Save config
        config = {
            "shared_encoder": self.shared_encoder,
            "hidden_size": self.query_encoder.hidden_size,
        }
        torch.save(config, os.path.join(save_path, "config.pt"))
    
    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load model from saved weights."""
        import os
        
        config = torch.load(os.path.join(load_path, "config.pt"))
        config.update(kwargs)
        
        model = cls(**config)
        model.load_state_dict(
            torch.load(os.path.join(load_path, "model.pt"))
        )
        
        return model


class DATRLoss(nn.Module):
    """
    Combined loss for DATR training:
    1. Contrastive retrieval loss (InfoNCE)
    2. Diachronic alignment loss
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        diachronic_weight: float = 0.3
    ):
        """
        Args:
            temperature: Temperature for softmax
            diachronic_weight: Weight for diachronic alignment loss
        """
        super().__init__()
        self.temperature = temperature
        self.diachronic_weight = diachronic_weight
    
    def retrieval_loss(
        self,
        query_emb: torch.Tensor,
        passage_emb: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive retrieval loss (InfoNCE).
        
        Args:
            query_emb: [batch_size, hidden_size]
            passage_emb: [batch_size * (1 + num_negatives), hidden_size]
            labels: [batch_size] - indices of positive passages
            
        Returns:
            Loss value
        """
        # Compute similarity scores
        scores = torch.matmul(query_emb, passage_emb.transpose(0, 1))
        scores = scores / self.temperature
        
        # Cross entropy loss
        loss = F.cross_entropy(scores, labels)
        
        return loss
    
    def diachronic_alignment_loss(
        self,
        modern_emb: torch.Tensor,
        historical_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diachronic alignment loss.
        
        Encourages modern and historical versions of the same query
        to have similar embeddings.
        
        Args:
            modern_emb: [batch_size, hidden_size] - modern paraphrase embeddings
            historical_emb: [batch_size, hidden_size] - historical query embeddings
            
        Returns:
            Loss value
        """
        # InfoNCE loss where positives are paired (modern, historical)
        batch_size = modern_emb.size(0)
        
        # Similarity matrix
        scores = torch.matmul(modern_emb, historical_emb.transpose(0, 1))
        scores = scores / self.temperature
        
        # Labels: diagonal entries are positives
        labels = torch.arange(batch_size, device=scores.device)
        
        # Bidirectional loss
        loss_m2h = F.cross_entropy(scores, labels)
        loss_h2m = F.cross_entropy(scores.transpose(0, 1), labels)
        
        return (loss_m2h + loss_h2m) / 2
    
    def forward(
        self,
        query_emb: torch.Tensor,
        passage_emb: torch.Tensor,
        labels: torch.Tensor,
        modern_emb: Optional[torch.Tensor] = None,
        historical_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with individual losses and total loss
        """
        # Retrieval loss
        retrieval_loss = self.retrieval_loss(query_emb, passage_emb, labels)
        
        losses = {
            "retrieval_loss": retrieval_loss,
            "total_loss": retrieval_loss
        }
        
        # Diachronic alignment loss (if provided)
        if modern_emb is not None and historical_emb is not None:
            diachronic_loss = self.diachronic_alignment_loss(modern_emb, historical_emb)
            losses["diachronic_loss"] = diachronic_loss
            losses["total_loss"] = retrieval_loss + self.diachronic_weight * diachronic_loss
        
        return losses
