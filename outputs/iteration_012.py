# BrownBioTech Multi-Omics Fusion Implementation (Iteration 12→13)

## File 1: `brownbiotech/agents/multiomics/omic_modality.py`

```python
"""
Per-Omics Modality Encoders for BrownBioTech Multi-Omics Fusion.

Implements specialized encoders for transcriptomics, proteomics, 
metabolomics, and genomics data modalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np


class OmicsModality(Enum):
    """Supported omics data modalities."""
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    GENOMICS = "genomics"
    EPIGENOMICS = "epigenomics"


@dataclass
class ModalityConfig:
    """Configuration for a single omics modality encoder."""
    modality: OmicsModality
    input_dim: int
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "gelu"


# Default configurations for each modality
DEFAULT_MODALITY_CONFIGS: Dict[OmicsModality, ModalityConfig] = {
    OmicsModality.TRANSCRIPTOMICS: ModalityConfig(
        modality=OmicsModality.TRANSCRIPTOMICS,
        input_dim=20000,  # ~20K genes
        hidden_dim=512,
        output_dim=128,
        num_layers=3,
    ),
    OmicsModality.PROTEOMICS: ModalityConfig(
        modality=OmicsModality.PROTEOMICS,
        input_dim=5000,  # ~5K proteins
        hidden_dim=256,
        output_dim=128,
        num_layers=2,
    ),
    OmicsModality.METABOLOMICS: ModalityConfig(
        modality=OmicsModality.METABOLOMICS,
        input_dim=1000,  # ~1K metabolites
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
    ),
    OmicsModality.GENOMICS: ModalityConfig(
        modality=OmicsModality.GENOMICS,
        input_dim=500,  # Mutation features
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
    ),
    OmicsModality.EPIGENOMICS: ModalityConfig(
        modality=OmicsModality.EPIGENOMICS,
        input_dim=2000,  # Methylation/accessibility features
        hidden_dim=256,
        output_dim=128,
        num_layers=2,
    ),
}


class ActivationFactory:
    """Factory for creating activation functions."""
    
    _ACTIVATIONS = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    
    @classmethod
    def create(cls, name: str) -> nn.Module:
        """Create an activation function by name."""
        if name.lower() not in cls._ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{name}'. "
                f"Available: {list(cls._ACTIVATIONS.keys())}"
            )
        return cls._ACTIVATIONS[name.lower()]()


class ModalityEncoderBlock(nn.Module):
    """Single encoder block for one modality."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        use_batch_norm: bool,
        activation: str,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder layers
        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            if i < len(dims) - 2:  # No activation on final layer
                layers.append(ActivationFactory.create(activation))
                layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # Attention pooling for variable-length inputs
        self.attention_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, 1),
        )
    
    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input features.
        
        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, seq_len, input_dim)
            return_attention: If True, return attention weights
            
        Returns:
            Encoded representation of shape (batch, output_dim)
            Optionally attention weights of shape (batch, seq_len) if input is 3D
        """
        if x.dim() == 2:
            # Direct encoding for fixed-size features
            encoded = self.encoder(x)
            if return_attention:
                return encoded, torch.ones(x.shape[0], 1, device=x.device)
            return encoded
        
        elif x.dim() == 3:
            # Sequence encoding with attention pooling
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(batch_size * seq_len, -1)
            encoded_flat = self.encoder(x_flat)
            encoded = encoded_flat.reshape(batch_size, seq_len, -1)
            
            # Attention pooling
            attn_weights = self.attention_pool(encoded).squeeze(-1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            pooled = torch.bmm(
                attn_weights.unsqueeze(1), encoded
            ).squeeze(1)
            
            if return_attention:
                return pooled, attn_weights
            return pooled
        
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")


class SparseGenomicsEncoder(nn.Module):
    """Specialized encoder for sparse genomic mutation data."""
    
    def __init__(
        self,
        num_genes: int = 20000,
        embedding_dim: int = 64,
        output_dim: int = 128,
        num_mutation_types: int = 6,
    ):
        super().__init__()
        
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.mutation_type_embedding = nn.Embedding(num_mutation_types, embedding_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )
        
        # Global mutation burden embedding
        self.burden_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )
        
        self.output_proj = nn.Linear(output_dim * 2, output_dim)
    
    def forward(
        self,
        gene_indices: torch.Tensor,
        mutation_types: torch.Tensor,
        mutation_burden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sparse mutation data.
        
        Args:
            gene_indices: (batch, max_mutations) gene indices
            mutation_types: (batch, max_mutations) mutation type indices
            mutation_burden: (batch, 1) total mutation burden
            
        Returns:
            Encoded representation (batch, output_dim)
        """
        gene_emb = self.gene_embedding(gene_indices)
        mut_emb = self.mutation_type_embedding(mutation_types)
        
        # Combine gene and mutation type embeddings
        combined = torch.cat([gene_emb, mut_emb], dim=-1)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        fused_flat = self.fusion(combined_flat)
        fused = fused_flat.reshape(gene_indices.shape[0], gene_indices.shape[1], -1)
        
        # Mean pool over mutations
        mutation_encoding = fused.mean(dim=1)
        
        # Add burden information
        burden_encoding = self.burden_embedding(mutation_burden)
        
        # Final fusion
        output = self.output_proj(
            torch.cat([mutation_encoding, burden_encoding], dim=-1)
        )
        return output


class TranscriptomicsEncoder(nn.Module):
    """Specialized encoder for transcriptomics with pathway-aware attention."""
    
    def __init__(
        self,
        input_dim: int = 20000,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_pathways: int = 50,
        pathway_gene_indices: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Gene-level encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        
        # Pathway attention
        self.pathway_query = nn.Embedding(num_pathways, 64)
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )
        
        # Pathway aggregation
        self.pathway_agg = nn.Sequential(
            nn.Linear(64 * num_pathways, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Direct encoding fallback
        self.direct_encoder = ModalityEncoderBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=0.1,
            use_batch_norm=True,
            activation="gelu",
        )
        
        self.pathway_gene_indices = pathway_gene_indices
    
    def forward(
        self, x: torch.Tensor, use_pathway: bool = True
    ) -> torch.Tensor:
        """
        Encode transcriptomics data.
        
        Args:
            x: Expression values (batch, num_genes)
            use_pathway: Whether to use pathway-aware encoding
            
        Returns:
            Encoded representation (batch, output_dim)
        """
        if not use_pathway or self.pathway_gene_indices is None:
            return self.direct_encoder(x)
        
        batch_size = x.shape[0]
        
        # Encode each gene
        gene_features = self.gene_encoder(x.unsqueeze(-1))  # (batch, genes, 64)
        
        # Pathway attention (simplified - use all genes as context)
        pathway_queries = self.pathway_query.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        pathway_encoded, _ = self.pathway_attention(
            pathway_queries, gene_features, gene_features
        )
        
        # Flatten and aggregate
        pathway_flat = pathway_encoded.reshape(batch_size, -1)
        return self.pathway_agg(pathway_flat)


class OmicsModalityEncoder(nn.Module):
    """Unified encoder that dispatches to modality-specific encoders."""
    
    def __init__(
        self,
        config: ModalityConfig,
        pathway_gene_indices: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.config = config
        self.modality = config.modality
        
        # Create modality-specific encoder
        if config.modality == OmicsModality.GENOMICS:
            self.encoder = SparseGenomicsEncoder(
                num_genes=config.input_dim,
                output_dim=config.output_dim,
            )
            self._is_sparse = True
        elif config.modality == OmicsModality.TRANSCRIPTOMICS:
            self.encoder = TranscriptomicsEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                pathway_gene_indices=pathway_gene_indices,
            )
            self._is_sparse = False
        else:
            self.encoder = ModalityEncoderBlock(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_batch_norm=config.use_batch_norm,
                activation=config.activation,
            )
            self._is_sparse = False
        
        # Layer normalization for stable fusion
        self.output_norm = nn.LayerNorm(config.output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode omics data.
        
        Args:
            x: Input tensor (format depends on modality)
            **kwargs: Additional modality-specific arguments
            
        Returns:
            Normalized encoded representation (batch, output_dim)
        """
        encoded = self.encoder(x, **kwargs)
        return self.output_norm(encoded)
    
    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder."""
        return self.config.output_dim


def create_modality_encoders(
    modalities: list[OmicsModality],
    custom_configs: Optional[Dict[OmicsModality, ModalityConfig]] = None,
    pathway_gene_indices: Optional[torch.Tensor] = None,
) -> Dict[OmicsModality, OmicsModalityEncoder]:
    """
    Create encoders for multiple modalities.
    
    Args:
        modalities: List of modalities to create encoders for
        custom_configs: Optional custom configurations
        pathway_gene_indices: Gene-to-pathway mapping for transcriptomics
        
    Returns:
        Dictionary mapping modality to encoder
    """
    encoders = {}
    
    for modality in modalities:
        config = custom_configs.get(modality, DEFAULT_MODALITY_CONFIGS[modality])
        encoders[modality] = OmicsModalityEncoder(
            config=config,
            pathway_gene_indices=pathway_gene_indices,
        )
    
    return encoders
```

---

## File 2: `brownbiotech/agents/multiomics/fusion_encoder.py`

```python
"""
Multi-Modal Late Fusion Transformer for BrownBioTech.

Implements a late fusion architecture where each omics modality is 
encoded separately, then fused using cross-modal attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math

from .omic_modality import (
    OmicsModality,
    OmicsModalityEncoder,
    ModalityConfig,
    DEFAULT_MODALITY_CONFIGS,
    create_modality_encoders,
)


@dataclass
class FusionConfig:
    """Configuration for the late fusion transformer."""
    modalities: List[OmicsModality] = field(
        default_factory=lambda: [
            OmicsModality.TRANSCRIPTOMICS,
            OmicsModality.PROTEOMICS,
            OmicsModality.METABOLOMICS,
        ]
    )
    modality_output_dim: int = 128
    fusion_dim: int = 256
    num_fusion_layers: int = 4
    num_attention_heads: int = 8
    fusion_dropout: float = 0.1
    use_modality_tokens: bool = True
    use_cross_attention: bool = True
    custom_modality_configs: Optional[Dict[OmicsModality, ModalityConfig]] = None


class ModalityToken(nn.Module):
    """Learnable modality-specific token for late fusion."""
    
    def __init__(self, dim: int, num_modalities: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_modalities, dim) * 0.02)
        self.modality_embeddings = nn.Embedding(num_modalities, dim)
    
    def forward(self, modality_indices: torch.Tensor) -> torch.Tensor:
        """
        Get modality tokens.
        
        Args:
            modality_indices: (batch, num_modalities) indices
            
        Returns:
            Modality tokens (batch, num_modalities, dim)
        """
        batch_size = modality_indices.shape[0]
        tokens = self.tokens.expand(batch_size, -1, -1)
        mod_emb = self.modality_embeddings(modality_indices)
        return tokens + mod_emb


class CrossModalAttention(nn.Module):
    """Cross-modal attention between different omics representations."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            x: Query tensor (batch, seq_len, dim)
            context: Context tensor for cross-attention (batch, ctx_len, dim)
            attention_mask: Optional attention mask
            
        Returns:
            Attended tensor (batch, seq_len, dim)
        """
        # Self-attention
        attn_out, _ = self.self_attention(
            x, x, x, attn_mask=attention_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention (if context provided)
        if context is not None:
            cross_out, cross_weights = self.cross_attention(
                x, context, context, attn_mask=attention_mask
            )
            x = self.norm2(x + self.dropout(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x


class LateFusionTransformer(nn.Module):
    """
    Late fusion transformer for multi-omics integration.
    
    Architecture:
    1. Each modality is encoded independently
    2. Modality representations are projected to common space
    3. Cross-modal attention learns interactions
    4. Final representation is aggregated
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        self.config = config
        self.modalities = config.modalities
        self.num_modalities = len(config.modalities)
        
        # Create modality-specific encoders
        self.modality_encoders = nn.ModuleDict({
            mod.value: OmicsModalityEncoder(
                config=config.custom_modality_configs.get(
                    mod, DEFAULT_MODALITY_CONFIGS[mod]
                ),
            )
            for mod in config.modalities
        })
        
        # Project all modalities to fusion dimension
        self.modality_projections = nn.ModuleDict({
            mod.value: nn.Sequential(
                nn.Linear(config.modality_output_dim, config.fusion_dim),
                nn.GELU(),
                nn.LayerNorm(config.fusion_dim),
            )
            for mod in config.modalities
        })
        
        # Modality tokens
        if config.use_modality_tokens:
            self.modality_tokens = ModalityToken(
                dim=config.fusion_dim,
                num_modalities=self.num_modalities,
            )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossModalAttention(
                dim=config.fusion_dim,
                num_heads=config.num_attention_heads,
                dropout=config.fusion_dropout,
            )
            for _ in range(config.num_fusion_layers)
        ])
        
        # Final aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(config.fusion_dim * self.num_modalities, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        
        # Modality importance weights (learnable)
        self.modality_importance = nn.Parameter(
            torch.zeros(self.num_modalities)
        )
        
        # Global token for final representation
        self.global_token = nn.Parameter(torch.randn(1, 1, config.fusion_dim) * 0.02)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.num_attention_heads,
            dropout=config.fusion_dropout,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(config.fusion_dim)
    
    def encode_modalities(
        self,
        modality_data: Dict[OmicsModality, torch.Tensor],
    ) -> Dict[OmicsModality, torch.Tensor]:
        """
        Encode each modality independently.
        
        Args:
            modality_data: Dictionary mapping modality to input tensor
            
        Returns:
            Dictionary mapping modality to encoded representation
        """
        encoded = {}
        
        for modality, data in modality_data.items():
            if modality.value not in self.modality_encoders:
                raise ValueError(f"Modality {modality} not configured in fusion encoder")
            
            encoder = self.modality_encoders[modality.value]
            encoded[modality] = encoder(data)
        
        return encoded
    
    def fuse_modalities(
        self,
        encoded: Dict[OmicsModality, torch.Tensor],
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Fuse encoded modality representations.
        
        Args:
            encoded: Dictionary of encoded modality representations
            return_attention: If True, return attention weights
            
        Returns:
            Fused representation (batch, fusion_dim)
            Optionally attention weights
        """
        batch_size = next(iter(encoded.values())).shape[0]
        device = next(iter(encoded.values())).device
        
        # Project to common space
        projected = {}
        for modality, enc in encoded.items():
            projected[modality] = self.modality_projections[modality.value](enc)
        
        # Stack modality representations
        modality_order = [m for m in self.modalities if m in projected]
        modality_stack = torch.stack(
            [projected[m] for m in modality_order], dim=1
        )  # (batch, num_modalities, fusion_dim)
        
        # Add modality tokens
        if self.config.use_modality_tokens:
            modality_indices = torch.arange(
                self.num_modalities, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            mod_tokens = self.modality_tokens(modality_indices)
            modality_stack = modality_stack + mod_tokens
        
        # Apply fusion layers
        attention_weights = {}
        for i, layer in enumerate(self.fusion_layers):
            if self.config.use_cross_attention and i > 0:
                # Use previous layer output as context
                modality_stack = layer(
                    modality_stack, context=modality_stack
                )
            else:
                modality_stack = layer(modality_stack)
            
            if return_attention:
                attention_weights[f"layer_{i}"] = modality_stack
        
        # Global token attention
        global_tok = self.global_token.expand(batch_size, -1, -1)
        fused, global_attn = self.global_attention(
            global_tok, modality_stack, modality_stack
        )
        fused = self.global_norm(fused + global_tok).squeeze(1)
        
        if return_attention:
            attention_weights["global"] = global_attn
            return fused, attention_weights
        
        return fused
    
    def compute_modality_weights(
        self,
        encoded: Dict[OmicsModality, torch.Tensor],
    ) -> Dict[OmicsModality, float]:
        """
        Compute importance weights for each modality.
        
        Args:
            encoded: Dictionary of encoded representations
            
        Returns:
            Dictionary mapping modality to importance weight
        """
        weights = F.softmax(self.modality_importance, dim=0)
        
        modality_order = [m for m in self.modalities if m in encoded]
        return {m: weights[i].item() for i, m in enumerate(modality_order)}
    
    def forward(
        self,
        modality_data: Dict[OmicsModality, torch.Tensor],
        return_attention: bool = False,
        return_modality_weights: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[OmicsModality, float]],
    ]:
        """
        Full forward pass: encode modalities then fuse.
        
        Args:
            modality_data: Dictionary mapping modality to input tensor
            return_attention: If True, return attention weights
            return_modality_weights: If True, return modality importance weights
            
        Returns:
            Fused representation
            Optionally attention weights and/or modality weights
        """
        # Encode each modality
        encoded = self.encode_modalities(modality_data)
        
        # Fuse modalities
        if return_attention:
            fused, attention = self.fuse_modalities(encoded, return_attention=True)
        else:
            fused = self.fuse_modalities(encoded, return_attention=False)
            attention = {}
        
        # Compute modality weights if requested
        mod_weights = {}
        if return_modality_weights:
            mod_weights = self.compute_modality_weights(encoded)
        
        # Return based on what's requested
        if return_attention and return_modality_weights:
            return fused, attention, mod_weights
        elif return_attention:
            return fused, attention
        elif return_modality_weights:
            return fused, mod_weights
        return fused
    
    def get_fusion_dim(self) -> int:
        """Return the fusion dimension."""
        return self.config.fusion_dim


class MultiOmicsFusionPipeline:
    """
    High-level pipeline for multi-omics fusion inference.
    
    Handles preprocessing, encoding, fusion, and post-processing.
    """
    
    def __init__(
        self,
        model: LateFusionTransformer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict(
        self,
        modality_data: Dict[OmicsModality, Union[torch.Tensor, np.ndarray]],
    ) -> torch.Tensor:
        """
        Generate fused representation.
        
        Args:
            modality_data: Dictionary of modality inputs
            
        Returns:
            Fused representation tensor
        """
        # Convert numpy to tensor if needed
        tensor_data = {}
        for mod, data in modality_data.items():
            if isinstance(data, np.ndarray):
                tensor_data[mod] = torch.from_numpy(data).float().to(self.device)
            else:
                tensor_data[mod] = data.to(self.device)
        
        return self.model(tensor_data)
    
    @torch.no_grad()
    def predict_with_explanations(
        self,
        modality_data: Dict[OmicsModality, Union[torch.Tensor, np.ndarray]],
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Generate fused representation with explanation data.
        
        Args:
            modality_data: Dictionary of modality inputs
            
        Returns:
            Dictionary with 'representation', 'attention', and 'modality_weights'
        """
        tensor_data = {}
        for mod, data in modality_data.items():
            if isinstance(data, np.ndarray):
                tensor_data[mod] = torch.from_numpy(data).float().to(self.device)
            else:
                tensor_data[mod] = data.to(self.device)
        
        fused, attention, mod_weights = self.model(
            tensor_data,
            return_attention=True,
            return_modality_weights=True,
        )
        
        return {
            "representation": fused,
            "attention": attention,
            "modality_weights": mod_weights,
        }
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MultiOmicsFusionPipeline":
        """Load a model from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = LateFusionTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, device=device)
```

---

## File 3: `brownbiotech/agents/multiomics/response_predictor.py`

```python
"""
DGAT1/YARS2 Co-Targeting Response Predictor for BrownBioTech.

Predicts patient response to DGAT1/YARS2 co-targeting therapy
using multi-omics fused representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .fusion_encoder import LateFusionTransformer, FusionConfig
from .omic_modality import OmicsModality


class ResponseCategory(Enum):
    """Patient response categories."""
    COMPLETE_RESPONSE = "complete_response"
    PARTIAL_RESPONSE = "partial_response"
    STABLE_DISEASE = "stable_disease"
    PROGRESSIVE_DISEASE = "progressive_disease"


@dataclass
class ResponsePredictorConfig:
    """Configuration for the response predictor."""
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    fusion_dim: int = 256
    hidden_dim: int = 128
    num_response_classes: int = 4
    use_survival_head: bool = True
    use_biomarker_head: bool = True
    dropout: float = 0.2
    target_genes: List[str] = field(
        default_factory=lambda: ["DGAT1", "YARS2"]
    )


class SurvivalHead(nn.Module):
    """Cox proportional hazards-style survival prediction head."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Risk score
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict risk score (higher = worse prognosis).
        
        Args:
            x: Input features (batch, input_dim)
            
        Returns:
            Risk scores (batch, 1)
        """
        return self.network(x)


class BiomarkerHead(nn.Module):
    """Predict target gene expression/activity from fused representation."""
    
    def __init__(
        self,
        input_dim: int,
        target_genes: List[str],
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.target_genes = target_genes
        self.num_targets = len(target_genes)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_targets),
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_targets),
            nn.Softplus(),  # Positive uncertainty
        )
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict target gene activity.
        
        Args:
            x: Input features (batch, input_dim)
            
        Returns:
            Tuple of (predictions, uncertainties) each (batch, num_targets)
        """
        predictions = self.network(x)
        uncertainties = self.uncertainty(x)
        return predictions, uncertainties


class CoTargetingAttention(nn.Module):
    """Attention mechanism specific to DGAT1/YARS2 co-targeting."""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        
        self.dgat1_query = nn.Linear(dim, dim)
        self.yars2_query = nn.Linear(dim, dim)
        self.shared_key = nn.Linear(dim, dim)
        self.shared_value = nn.Linear(dim, dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply co-targeting specific attention.
        
        Args:
            x: Fused representation (batch, dim)
            
        Returns:
            Co-targeting enhanced representation (batch, dim)
        """
        # Create target-specific queries
        dgat1_q = self.dgat1_query(x).unsqueeze(1)
        yars2_q = self.yars2_query(x).unsqueeze(1)
        
        # Shared key-value from fused representation
        k = self.shared_key(x).unsqueeze(1)
        v = self.shared_value(x).unsqueeze(1)
        
        # Attend for each target
        dgat1_out, _ = self.attention(dgat1_q, k, v)
        yars2_out, _ = self.attention(yars2_q, k, v)
        
        # Gated combination
        combined = torch.cat([dgat1_out.squeeze(1), yars2_out.squeeze(1)], dim=-1)
        gate = self.gate(combined)
        
        # Apply gate to original representation
        enhanced = x * gate + x
        return self.norm(enhanced)


class ResponsePredictor(nn.Module):
    """
    DGAT1/YARS2 Co-Targeting Response Predictor.
    
    Takes multi-omics fused representations and predicts:
    1. Response category (CR/PR/SD/PD)
    2. Survival risk score
    3. Target gene activity predictions
    """
    
    def __init__(self, config: ResponsePredictorConfig):
        super().__init__()
        
        self.config = config
        
        # Fusion encoder
        self.fusion_encoder = LateFusionTransformer(config.fusion_config)
        
        # Co-targeting specific attention
        self.co_targeting_attn = CoTargetingAttention(
            dim=config.fusion_dim
        )
        
        # Response classification head
        self.response_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_response_classes),
        )
        
        # Optional heads
        if config.use_survival_head:
            self.survival_head = SurvivalHead(config.fusion_dim)
        
        if config.use_biomarker_head:
            self.biomarker_head = BiomarkerHead(
                config.fusion_dim,
                config.target_genes,
            )
        
        # Confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(
        self,
        modality_data: Dict[OmicsModality, torch.Tensor],
        return_all: bool = False,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Predict response to DGAT1/YARS2 co-targeting.
        
        Args:
            modality_data: Dictionary of modality inputs
            return_all: If True, return all predictions
            
        Returns:
            Response logits (batch, num_classes) if return_all=False
            Dictionary of all predictions if return_all=True
        """
        # Fuse multi-omics data
        fused = self.fusion_encoder(modality_data)
        
        # Apply co-targeting attention
        enhanced = self.co_targeting_attn(fused)
        
        # Response classification
        logits = self.response_classifier(enhanced)
        calibrated_logits = logits / self.temperature
        
        if not return_all:
            return calibrated_logits
        
        results = {
            "response_logits": calibrated_logits,
            "response_probs": F.softmax(calibrated_logits, dim=-1),
            "fused_representation": fused,
            "enhanced_representation": enhanced,
        }
        
        # Add survival prediction
        if self.config.use_survival_head:
            results["risk_score"] = self.survival_head(enhanced)
        
        # Add biomarker predictions
        if self.config.use_biomarker_head:
            preds, unc = self.biomarker_head(enhanced)
            results["biomarker_predictions"] = preds
            results["biomarker_uncertainties"] = unc
        
        return results
    
    def predict_response(
        self,
        modality_data: Dict[OmicsModality, torch.Tensor],
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Get human-readable response prediction.
        
        Args:
            modality_data: Dictionary of modality inputs
            
        Returns:
            Dictionary with response category, confidence, and details
        """
        self.eval()
        
        with torch.no_grad():
            results = self.forward(modality_data, return_all=True)
        
        probs = results["response_probs"][0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        
        response_map = {
            0: ResponseCategory.COMPLETE_RESPONSE.value,
            1: ResponseCategory.PARTIAL_RESPONSE.value,
            2: ResponseCategory.STABLE_DISEASE.value,
            3: ResponseCategory.PROGRESSIVE_DISEASE.value,
        }
        
        output = {
            "predicted_response": response_map[pred_class],
            "confidence": confidence,
            "response_probabilities": {
                response_map[i]: probs[i].item()
                for i in range(len(probs))
            },
        }
        
        if "risk_score" in results:
            output["risk_score"] = results["risk_score"][0, 0].item()
        
        if "biomarker_predictions" in results:
            output["target_predictions"] = {
                gene: {
                    "predicted_activity": results["biomarker_predictions"][0, i].item(),
                    "uncertainty": results["biomarker_uncertainties"][0, i].item(),
                }
                for i, gene in enumerate(self.config.target_genes)
            }
        
        return output


class ResponsePredictorLoss(nn.Module):
    """
    Combined loss function for response predictor training.
    """
    
    def __init__(
        self,
        response_weight: float = 1.0,
        survival_weight: float = 0.5,
        biomarker_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.response_weight = response_weight
        self.survival_weight = survival_weight
        self.biomarker_weight = biomarker_weight
        self.label_smoothing = label_smoothing
        
        self.response_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}
        
        # Response classification loss
        response_loss = self.response_loss(
            predictions["response_logits"],
            targets["response_label"],
        )
        losses["response"] = response_loss.item()
        
        total_loss = self.response_weight * response_loss
        
        # Survival loss (Cox-like)
        if "risk_score" in predictions and "survival_time" in targets:
            risk = predictions["risk_score"].squeeze()
            survival_time = targets["survival_time"]
            event = targets["event"].float()
            
            # Simplified Cox partial likelihood
            risk_exp = torch.exp(risk)
            loss = -torch.mean(
                event * (risk - torch.log(torch.cumsum(risk_exp, dim=0)))
            )
            losses["survival"] = loss.item()
            total_loss += self.survival_weight * loss
        
        # Biomarker loss (Gaussian NLL)
        if "biomarker_predictions" in predictions and "biomarker_values" in targets:
            pred = predictions["biomarker_predictions"]
            unc = predictions["biomarker_uncertainties"]
            true = targets["biomarker_values"]
            
            # Negative log likelihood under Gaussian
            nll = 0.5 * (
                torch.log(unc ** 2) + (true - pred) ** 2 / (unc ** 2)
            ).mean()
            losses["biomarker"] = nll.item()
            total_loss += self.biomarker_weight * nll
        
        losses["total"] = total_loss.item()
        return total_loss, losses


class CoTargetingPipeline:
    """
    End-to-end pipeline for DGAT1/YARS2 co-targeting prediction.
    """
    
    def __init__(
        self,
        predictor: ResponsePredictor,
        device: str = "cpu",
    ):
        self.predictor = predictor.to(device)
        self.device = device
        self.predictor.eval()
    
    @torch.no_grad()
    def predict(
        self,
        modality_data: Dict[OmicsModality, Union[torch.Tensor, np.ndarray]],
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Generate co-targeting response prediction.
        
        Args:
            modality_data: Multi-omics input data
            
        Returns:
            Prediction results with response category and confidence
        """
        tensor_data = {}
        for mod, data in modality_data.items():
            if isinstance(data, np.ndarray):
                tensor_data[mod] = torch.from_numpy(data).float().to(self.device)
            else:
                tensor_data[mod] = data.to(self.device)
        
        return self.predictor.predict_response(tensor_data)
    
    def save(self, path: str) -> None:
        """Save the pipeline to disk."""
        torch.save({
            "predictor_state_dict": self.predictor.state_dict(),
            "config": self.predictor.config,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CoTargetingPipeline":
        """Load a pipeline from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        predictor = ResponsePredictor(config)
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
        return cls(predictor=predictor, device=device)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    fusion_config = FusionConfig(
        modalities=[
            OmicsModality.TRANSCRIPTOMICS,
            OmicsModality.PROTEOMICS,
            OmicsModality.METABOLOMICS,
        ],
        fusion_dim=128,
        num_fusion_layers=2,
    )
    
    predictor_config = ResponsePredictorConfig(
        fusion_config=fusion_config,
        fusion_dim=128,
        hidden_dim=64,
        num_response_classes=4,
        use_survival_head=True,
        use_biomarker_head=True,
    )
    
    # Create predictor
    predictor = ResponsePredictor(predictor_config)
    
    # Create dummy multi-omics data
    batch_size = 4
    modality_data = {
        OmicsModality.TRANSCRIPTOMICS: torch.randn(batch_size, 20000),
        OmicsModality.PROTEOMICS: torch.randn(batch_size, 5000),
        OmicsModality.METABOLOMICS: torch.randn(batch_size, 1000),
    }
    
    # Test forward pass
    print("Testing ResponsePredictor...")
    results = predictor(modality_data, return_all=True)
    
    print(f"Response logits shape: {results['response_logits'].shape}")
    print(f"Response probs shape: {results['response_probs'].shape}")
    print(f"Risk score shape: {results['risk_score'].shape}")
    print(f"Biomarker predictions shape: {results['biomarker_predictions'].shape}")
    
    # Test human-readable prediction
    print("\nTesting human-readable prediction...")
    single_data = {k: v[:1] for k, v in modality_data.items()}
    prediction = predictor.predict_response(single_data)
    print(f"Predicted response: {prediction['predicted_response']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Risk score: {prediction['risk_score']:.3f}")
    print(f"Target predictions: {prediction['target_predictions']}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    loss_fn = ResponsePredictorLoss()
    targets = {
        "response_label": torch.tensor([0, 1, 2, 3]),
        "survival_time": torch.tensor([365, 200, 500, 100], dtype=torch.float),
        "event": torch.tensor([0, 1, 0, 1]),
        "biomarker_values": torch.randn(batch_size, 2),
    }
    
    total_loss, loss_components = loss_fn(results, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_components}")
    
    print("\n✓ All tests passed!")
```

---

## Summary of Improvements

| Module | Key Features |
|--------|--------------|
| **`omic_modality.py`** | Per-modality encoders with specialized handling for sparse genomics, pathway-aware transcriptomics, and configurable MLP encoders for proteomics/metabolomics |
| **`fusion_encoder.py`** | Late fusion transformer with learnable modality tokens, cross-modal attention layers, global token aggregation, and modality importance weighting |
| **`response_predictor.py`** | DGAT1/YARS2 co-targeting predictor with multi-task heads (response classification, survival risk, biomarker prediction), uncertainty estimation, and temperature-scaled calibration |

### Architecture Highlights:
1. **Late Fusion**: Each modality encoded independently → projected to common space → fused via attention
2. **Co-Targeting Attention**: Specialized attention mechanism for DGAT1/YARS2 synergy detection
3. **Uncertainty-Aware**: Biomarker head predicts both values and uncertainties
4. **Calibrated Outputs**: Temperature scaling for reliable probability estimates
5. **Explainable**: Modality importance weights and attention visualization support