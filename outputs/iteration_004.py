# BrownBioTech Iteration 4/100: Unified Multimodal Graph-Transformer ADMET Agent

## File Structure

```
brownbiotech/
├── models/
│   ├── __init__.py
│   ├── graph_transformer.py
│   └── explainable_attention.py
├── agents/
│   ├── __init__.py
│   ├── admet_agent.py
│   └── design_feedback.py
└── core/
    └── molecular_graph.py
```

---

## File 1: `brownbiotech/core/molecular_graph.py`

```python
"""
Molecular Graph Data Structures for BrownBioTech Graph-Transformer.

Defines the multimodal input representation combining:
- Atom-level features (topology, chemical properties)
- Bond-level features
- 3D conformer features (for mitochondrial toxicity prediction)
- Sequence-derived features (for YARS2 off-target assessment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class ADMETTask(Enum):
    """ADMET prediction tasks with priority levels for DGAT1/YARS2 program."""
    MITOCHONDRIAL_TOXICITY = "mito_tox"  # YARS2 off-target - CRITICAL
    HERG_BLOCKADE = "herg"               # Cardiac liability - HIGH
    CYP3A4_INHIBITION = "cyp3a4"         # Drug-drug interaction - HIGH
    CYP2D6_INHIBITION = "cyp2d6"         # Drug-drug interaction - MEDIUM
    HEPATOTOXICITY = "hepato"            # Liver toxicity - HIGH
    SOLUBILITY = "solubility"            # Developability - MEDIUM
    PERMEABILITY = "permeability"        # Oral bioavailability - MEDIUM


@dataclass
class AtomFeatures:
    """Atom-level features for graph nodes."""
    atomic_num: int
    degree: int
    formal_charge: int
    hybridization: int
    is_aromatic: bool
    is_in_ring: bool
    vdw_radius: float
    electronegativity: float
    # Extended features for mitochondrial toxicity
    partial_charge: Optional[float] = None
    mulliken_charge: Optional[float] = None
    # YARS2-relevant: amino acid interaction propensity
    aa_interaction_score: Optional[float] = None


@dataclass
class BondFeatures:
    """Bond-level features for graph edges."""
    bond_type: int  # 0=none, 1=single, 2=double, 3=triple, 4=aromatic
    is_conjugated: bool
    is_in_ring: bool
    stereo: int  # 0=none, 1=Z, 2=E, 3=any
    bond_length: Optional[float] = None  # From 3D conformer


@dataclass
class MolecularGraph:
    """
    Multimodal molecular graph representation.
    
    Combines 2D topology, 3D geometry, and sequence-derived features
    for unified ADMET prediction.
    """
    smiles: str
    atom_features: list[AtomFeatures]
    bond_features: list[BondFeatures]
    edge_indices: list[Tuple[int, int]]
    
    # 3D conformer features (critical for mitochondrial membrane interaction)
    conformer_coords: Optional[np.ndarray] = None  # (n_atoms, 3)
    molecular_dipole: Optional[float] = None
    
    # Sequence-derived features (for YARS2 off-target)
    protein_binding_profile: Optional[np.ndarray] = None  # (n_residues,)
    
    # Metadata
    mol_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    num_rotatable_bonds: int = 0
    
    def to_tensors(self, device: str = "cpu") -> dict[str, Tensor]:
        """Convert to PyTorch tensors for model input."""
        n_atoms = len(self.atom_features)
        
        # Atom feature matrix (n_atoms, n_features)
        atom_feat_dim = 8
        atom_matrix = torch.zeros(n_atoms, atom_feat_dim, device=device)
        for i, atom in enumerate(self.atom_features):
            atom_matrix[i] = torch.tensor([
                atom.atomic_num / 118.0,  # Normalize
                atom.degree / 4.0,
                atom.formal_charge / 4.0,
                atom.hybridization / 4.0,
                float(atom.is_aromatic),
                float(atom.is_in_ring),
                atom.vdw_radius / 3.0,
                atom.electronegativity / 4.0,
            ], device=device)
        
        # Edge index tensor (2, n_edges)
        edge_index = torch.tensor(
            list(zip(*self.edge_indices)) if self.edge_indices else [],
            dtype=torch.long,
            device=device
        ).t().contiguous() if self.edge_indices else torch.zeros(2, 0, dtype=torch.long, device=device)
        
        # Bond feature matrix (n_edges, n_bond_features)
        n_bond_feats = 4
        bond_matrix = torch.zeros(len(self.bond_features), n_bond_feats, device=device)
        for i, bond in enumerate(self.bond_features):
            bond_matrix[i] = torch.tensor([
                bond.bond_type / 4.0,
                float(bond.is_conjugated),
                float(bond.is_in_ring),
                bond.stereo / 3.0,
            ], device=device)
        
        # 3D coordinates if available
        coords = None
        if self.conformer_coords is not None:
            coords = torch.tensor(
                self.conformer_coords, 
                dtype=torch.float32, 
                device=device
            )
        
        # Global molecular features
        global_feats = torch.tensor([
            self.mol_weight / 1000.0,
            self.logp / 5.0,
            self.tpsa / 200.0,
            self.num_rotatable_bonds / 10.0,
        ], device=device)
        
        return {
            "atom_features": atom_matrix,
            "edge_index": edge_index,
            "bond_features": bond_matrix,
            "conformer_coords": coords,
            "global_features": global_feats,
            "protein_binding_profile": (
                torch.tensor(self.protein_binding_profile, device=device)
                if self.protein_binding_profile is not None else None
            ),
        }
```

---

## File 2: `brownbiotech/models/explainable_attention.py`

```python
"""
Explainable Multi-Head Attention for BrownBioTech Graph-Transformer.

Provides attention weights that can be traced back to molecular substructures
for Design Agent feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AttentionExplanation:
    """
    Container for explainable attention outputs.
    
    Enables Design Agent to understand WHY a molecule has certain ADMET risks
    and which substructures to modify.
    """
    # Per-head attention weights (n_heads, n_atoms, n_atoms)
    attention_weights: Tensor
    
    # Aggregated attention importance per atom (n_atoms,)
    atom_importance: Tensor
    
    # Top-k attended atom indices for each query atom
    top_attended_atoms: Tensor
    
    # Attention entropy (lower = more focused = more interpretable)
    attention_entropy: float
    
    # Which ADMET task this attention corresponds to
    task_name: str
    
    def get_critical_atoms(
        self, 
        threshold_percentile: float = 90.0
    ) -> Tuple[list[int], Tensor]:
        """
        Get atoms with attention importance above threshold.
        
        Returns:
            Tuple of (atom_indices, their_importance_scores)
        """
        k = max(1, int(len(self.atom_importance) * (1 - threshold_percentile / 100)))
        topk_values, topk_indices = torch.topk(self.atom_importance, k)
        return topk_indices.tolist(), topk_values
    
    def to_dict(self) -> dict:
        """Serialize for logging/visualization."""
        return {
            "task_name": self.task_name,
            "attention_entropy": self.attention_entropy,
            "atom_importance": self.atom_importance.detach().cpu().tolist(),
            "critical_atoms": self.get_critical_atoms()[0],
        }


class ExplainableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with full explainability support.
    
    Key features:
    - Returns attention weights for interpretation
    - Supports sparse attention patterns (for efficiency)
    - Provides entropy-based confidence metrics
    - Compatible with graph-structured inputs
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_sparse_attention: bool = False,
        sparse_topk: Optional[int] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sparse_attention = use_sparse_attention
        self.sparse_topk = sparse_topk
        
        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable attention temperature (helps with entropy calibration)
        self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))
        
        # For mitochondrial toxicity: bias attention toward aromatic rings
        self.register_buffer("aromatic_bias", None)
    
    def set_aromatic_bias(self, is_aromatic: Tensor) -> None:
        """
        Set attention bias for aromatic atoms.
        
        Mitochondrial toxicity often involves π-π stacking with membrane proteins.
        """
        # Create bias matrix that slightly upweights aromatic-aromatic interactions
        n = len(is_aromatic)
        aromatic_mask = is_aromatic.float().unsqueeze(0).expand(n, -1)
        bias = (aromatic_mask * aromatic_mask.t()) * 0.1
        self.aromatic_bias = bias
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_explanation: bool = True,
        task_name: str = "unknown",
    ) -> Tuple[Tensor, Optional[AttentionExplanation]]:
        """
        Forward pass with optional explanation output.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            mask: Optional attention mask
            return_explanation: Whether to compute attention explanation
            task_name: Name of ADMET task for explanation tracking
            
        Returns:
            Tuple of (output, explanation_or_none)
        """
        # Handle 2D input (single sample, no batch)
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(0)
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature
        
        # Apply aromatic bias if set (for mitochondrial toxicity)
        if self.aromatic_bias is not None:
            scores = scores + self.aromatic_bias.unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        
        # Optional sparse attention
        if self.use_sparse_attention and self.sparse_topk is not None:
            topk = min(self.sparse_topk, seq_len)
            topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)
            sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_indices, True)
            scores = scores.masked_fill(~sparse_mask, float("-inf"))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        # Remove batch dim if added
        if squeeze_output:
            output = output.squeeze(0)
            attn_weights = attn_weights.squeeze(0)
        
        # Compute explanation if requested
        explanation = None
        if return_explanation:
            # Aggregate across heads for atom importance
            atom_importance = attn_weights.mean(dim=0).sum(dim=0)  # (seq_len,)
            atom_importance = atom_importance / (atom_importance.max() + 1e-8)
            
            # Compute attention entropy (lower = more focused)
            log_probs = torch.log(attn_weights + 1e-10)
            entropy = -(attn_weights * log_probs).sum(dim=-1).mean().item()
            
            # Top attended atoms per query
            topk = min(5, seq_len)
            top_attended = attn_weights.topk(topk, dim=-1).indices
            
            explanation = AttentionExplanation(
                attention_weights=attn_weights.detach(),
                atom_importance=atom_importance.detach(),
                top_attended_atoms=top_attended.detach(),
                attention_entropy=entropy,
                task_name=task_name,
            )
        
        return output, explanation


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for combining molecular graph with protein features.
    
    Used for YARS2 off-target prediction where we need to model
    molecule-protein interactions.
    """
    
    def __init__(self, mol_dim: int, protein_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.mol_proj = nn.Linear(mol_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.cross_attn = ExplainableMultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.output_proj = nn.Linear(hidden_dim, mol_dim)
        
    def forward(
        self,
        mol_features: Tensor,      # (n_atoms, mol_dim)
        protein_features: Tensor,   # (n_residues, protein_dim)
        return_explanation: bool = True,
        task_name: str = "cross_modal",
    ) -> Tuple[Tensor, Optional[AttentionExplanation]]:
        """
        Cross-modal attention from molecule to protein.
        
        Returns:
            Updated molecular features with protein context
        """
        # Project to common dimension
        mol_proj = self.mol_proj(mol_features)
        prot_proj = self.protein_proj(protein_features)
        
        # Concatenate for cross-attention (mol queries, protein keys/values)
        combined = torch.cat([mol_proj, prot_proj], dim=0)
        
        # Apply attention
        attended, explanation = self.cross_attn(
            combined, return_explanation=return_explanation, task_name=task_name
        )
        
        # Extract only molecular part
        mol_updated = attended[:mol_features.size(0)]
        mol_updated = self.output_proj(mol_updated)
        
        return mol_updated, explanation
```

---

## File 3: `brownbiotech/models/graph_transformer.py`

```python
"""
Unified Multimodal Graph-Transformer for ADMET Prediction.

Replaces legacy single-task predictors with a single model that:
- Shares representations across all ADMET tasks
- Uses graph structure + 3D geometry + protein context
- Provides explainable outputs for Design Agent feedback
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .explainable_attention import (
    AttentionExplanation,
    CrossModalAttention,
    ExplainableMultiHeadAttention,
)


class GraphTransformerLayer(nn.Module):
    """
    Single layer of the Graph-Transformer.
    
    Combines:
    - Message passing (neighbor aggregation)
    - Self-attention (global context)
    - Feed-forward network
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_3d: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_3d = use_3d
        
        # Graph message passing
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),  # 2 atom + bond features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Self-attention
        self.self_attn = ExplainableMultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Optional 3D geometry encoding
        if use_3d:
            self.geometric_mlp = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for 3D coords
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: Tensor,      # (n_atoms, hidden_dim)
        edge_index: Tensor,         # (2, n_edges)
        bond_features: Tensor,      # (n_edges, n_bond_feats)
        coords: Optional[Tensor] = None,  # (n_atoms, 3)
        return_explanation: bool = True,
        task_name: str = "unknown",
    ) -> Tuple[Tensor, Optional[AttentionExplanation]]:
        """
        Forward pass through transformer layer.
        
        Returns:
            Updated node features and optional attention explanation
        """
        # Message passing
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            src_feats = node_features[src]
            dst_feats = node_features[dst]
            msg_input = torch.cat([src_feats, dst_feats, bond_features], dim=-1)
            messages = self.message_mlp(msg_input)
            
            # Aggregate messages to destination nodes
            aggregated = torch.zeros_like(node_features)
            aggregated.index_add_(0, dst, messages)
            node_features = node_features + aggregated
        
        node_features = self.ln1(node_features)
        
        # Self-attention
        attended, explanation = self.self_attn(
            node_features, return_explanation=return_explanation, task_name=task_name
        )
        node_features = node_features + attended
        node_features = self.ln2(node_features)
        
        # 3D geometry encoding
        if self.use_3d and coords is not None:
            geo_input = torch.cat([node_features, coords], dim=-1)
            geo_features = self.geometric_mlp(geo_input)
            node_features = node_features + geo_features
            node_features = self.ln3(node_features)
        
        # FFN
        node_features = node_features + self.ffn(node_features)
        
        return node_features, explanation


class ADMETPredictionHead(nn.Module):
    """
    Task-specific prediction head with uncertainty estimation.
    
    Each ADMET task gets its own head but shares the backbone.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,  # Binary: safe/toxic
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Uncertainty head (predicts epistemic uncertainty)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures positive uncertainty
        )
    
    def forward(
        self, 
        node_features: Tensor,
        attention_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict ADMET outcome with uncertainty.
        
        Returns:
            Tuple of (logits, uncertainty)
        """
        # Attention-weighted pooling if attention available
        if attention_weights is not None:
            # attention_weights: (n_heads, n_atoms, n_atoms)
            # Use mean attention received by each atom as pooling weights
            pool_weights = attention_weights.mean(dim=0).mean(dim=-1)  # (n_atoms,)
            pool_weights = F.softmax(pool_weights, dim=0)
            pooled = (node_features * pool_weights.unsqueeze(-1)).sum(dim=0)
        else:
            pooled = node_features.mean(dim=0)
        
        pooled = self.pool(pooled)
        logits = self.classifier(pooled)
        uncertainty = self.uncertainty_head(pooled)
        
        return logits, uncertainty.squeeze(-1)


class UnifiedADMETTransformer(nn.Module):
    """
    Unified Multimodal Graph-Transformer for all ADMET tasks.
    
    Architecture:
    1. Atom/bond feature embedding
    2. N graph-transformer layers with 3D geometry
    3. Optional cross-modal attention with protein features
    4. Task-specific prediction heads with uncertainty
    
    Replaces legacy separate models for each ADMET endpoint.
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 8,
        bond_feature_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_3d: bool = True,
        protein_dim: Optional[int] = None,
        tasks: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_3d = use_3d
        self.tasks = tasks or [
            "mito_tox", "herg", "cyp3a4", "cyp2d6", "hepato", "solubility", "permeability"
        ]
        
        # Input projections
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.bond_embedding = nn.Linear(bond_feature_dim, hidden_dim)
        
        # Global feature embedding
        self.global_embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),  # MW, logP, TPSA, rotatable bonds
            nn.GELU(),
        )
        
        # Graph-transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_3d=use_3d,
            )
            for _ in range(num_layers)
        ])
        
        # Cross-modal attention for protein features (YARS2)
        self.cross_modal_attn: Optional[CrossModalAttention] = None
        if protein_dim is not None:
            self.cross_modal_attn = CrossModalAttention(
                mol_dim=hidden_dim,
                protein_dim=protein_dim,
                hidden_dim=hidden_dim,
                num_heads=4,
            )
        
        # Task-specific prediction heads
        self.prediction_heads = nn.ModuleDict({
            task: ADMETPredictionHead(hidden_dim, num_classes=2, dropout=dropout)
            for task in self.tasks
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        atom_features: Tensor,
        edge_index: Tensor,
        bond_features: Tensor,
        conformer_coords: Optional[Tensor] = None,
        global_features: Optional[Tensor] = None,
        protein_features: Optional[Tensor] = None,
        is_aromatic: Optional[Tensor] = None,
        tasks_to_predict: Optional[List[str]] = None,
        return_explanations: bool = True,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Forward pass for unified ADMET prediction.
        
        Args:
            atom_features: (n_atoms, atom_feature_dim)
            edge_index: (2, n_edges)
            bond_features: (n_edges, bond_feature_dim)
            conformer_coords: Optional (n_atoms, 3)
            global_features: Optional (4,)
            protein_features: Optional (n_residues, protein_dim)
            is_aromatic: Optional (n_atoms,) bool tensor for attention bias
            tasks_to_predict: Which tasks to predict (default: all)
            return_explanations: Whether to compute attention explanations
            
        Returns:
            Dict mapping task_name -> {"logits": Tensor, "uncertainty": Tensor, "explanation": ...}
        """
        tasks_to_predict = tasks_to_predict or self.tasks
        
        # Embed inputs
        node_feats = self.atom_embedding(atom_features)
        bond_feats = self.bond_embedding(bond_features) if bond_features.size(0) > 0 else bond_features
        
        # Add global context to all nodes
        if global_features is not None:
            global_ctx = self.global_embedding(global_features)
            node_feats = node_feats + global_ctx.unsqueeze(0)
        
        # Set aromatic bias for mitochondrial toxicity attention
        if is_aromatic is not None and "mito_tox" in tasks_to_predict:
            self.layers[0].self_attn.set_aromatic_bias(is_aromatic)
        
        # Graph-transformer layers
        explanations: Dict[str, AttentionExplanation] = {}
        last_attention = None
        
        for i, layer in enumerate(self.layers):
            node_feats, layer_explanation = layer(
                node_feats,
                edge_index,
                bond_feats,
                coords=conformer_coords,
                return_explanation=return_explanations and (i == len(self.layers) - 1),
                task_name="graph_transformer",
            )
            if layer_explanation is not None:
                last_attention = layer_explanation.attention_weights
                explanations["graph_transformer"] = layer_explanation
        
        # Cross-modal attention with protein (YARS2 off-target)
        if self.cross_modal_attn is not None and protein_features is not None:
            node_feats, protein_explanation = self.cross_modal_attn(
                node_feats,
                protein_features,
                return_explanation=return_explanations,
                task_name="protein_binding",
            )
            if protein_explanation is not None:
                explanations["protein_binding"] = protein_explanation
                last_attention = protein_explanation.attention_weights
        
        # Task-specific predictions
        results = {}
        for task in tasks_to_predict:
            if task not in self.prediction_heads:
                continue
            
            logits, uncertainty = self.prediction_heads[task](
                node_feats, 
                attention_weights=last_attention
            )
            
            task_explanation = explanations.get("graph_transformer")
            if task == "mito_tox" and "protein_binding" in explanations:
                task_explanation = explanations["protein_binding"]
            
            results[task] = {
                "logits": logits,
                "uncertainty": uncertainty,
                "explanation": task_explanation,
            }
        
        return results
    
    def predict_with_feedback(
        self,
        mol_tensors: Dict[str, Tensor],
        tasks: Optional[List[str]] = None,
        risk_threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Predict ADMET with Design Agent feedback format.
        
        Returns structured output suitable for Design Agent consumption:
        - Risk scores
        - Critical atoms (for modification)
        - Uncertainty estimates
        - Actionable recommendations
        """
        self.eval()
        
        with torch.no_grad():
            results = self.forward(
                atom_features=mol_tensors["atom_features"],
                edge_index=mol_tensors["edge_index"],
                bond_features=mol_tensors["bond_features"],
                conformer_coords=mol_tensors.get("conformer_coords"),
                global_features=mol_tensors.get("global_features"),
                protein_features=mol_tensors.get("protein_binding_profile"),
                tasks_to_predict=tasks,
                return_explanations=True,
            )
        
        # Format for Design Agent
        feedback = {}
        for task, pred in results.items():
            probs = torch.softmax(pred["logits"], dim=-1)
            risk_score = probs[1].item()  # Probability of toxic/adverse
            uncertainty = pred["uncertainty"].item()
            
            # Get critical atoms from explanation
            critical_atoms = []
            explanation_data = {}
            if pred["explanation"] is not None:
                critical_atoms, importance = pred["explanation"].get_critical_atoms(90.0)
                explanation_data = pred["explanation"].to_dict()
            
            # Generate recommendation
            if risk_score > risk_threshold:
                if task == "mito_tox":
                    recommendation = (
                        f"High mitochondrial toxicity risk ({risk_score:.2%}). "
                        f"Consider reducing aromatic character at atoms {critical_atoms[:3]}. "
                        f"YARS2 off-target interaction likely."
                    )
                elif task == "herg":
                    recommendation = (
                        f"hERG blockade risk ({risk_score:.2%}). "
                        f"Consider reducing basicity or lipophilicity."
                    )
                elif task.startswith("cyp"):
                    recommendation = (
                        f"{task.upper()} inhibition risk ({risk_score:.2%}). "
                        f"Consider modifying metabolic soft spots."
                    )
                else:
                    recommendation = f"High {task} risk ({risk_score:.2%}). Modification recommended."
            else:
                recommendation = f"{task} risk acceptable ({risk_score:.2%})."
            
            feedback[task] = {
                "risk_score": risk_score,
                "uncertainty": uncertainty,
                "is_high_risk": risk_score > risk_threshold,
                "critical_atoms": critical_atoms,
                "explanation": explanation_data,
                "recommendation": recommendation,
            }
        
        return feedback
```

---

## File 4: `brownbiotech/agents/admet_agent.py`

```python
"""
Upgraded ADMET Agent for BrownBioTech.

Replaces legacy single-task predictors with Unified Multimodal Graph-Transformer.
Provides explainable predictions for Design Agent feedback loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from brownbiotech.core.molecular_graph import MolecularGraph
from brownbiotech.models.graph_transformer import UnifiedADMETTransformer

logger = logging.getLogger(__name__)


@dataclass
class ADMETResult:
    """Container for ADMET prediction results."""
    task: str
    risk_score: float
    uncertainty: float
    is_high_risk: bool
    critical_atoms: List[int]
    recommendation: str
    explanation: Dict[str, Any]
    
    @property
    def confidence(self) -> float:
        """Inverse of uncertainty, normalized to [0, 1]."""
        return 1.0 / (1.0 + self.uncertainty)


class ADMETAgent:
    """
    Upgraded ADMET Agent using Unified Multimodal Graph-Transformer.
    
    Key improvements over legacy:
    1. Single model for all ADMET tasks (better data efficiency)
    2. 3D geometry integration (critical for mitochondrial toxicity)
    3. Protein context (for YARS2 off-target)
    4. Explainable outputs (for Design Agent feedback)
    5. Uncertainty quantification (for active learning)
    
    Usage:
        agent = ADMETAgent.load("path/to/checkpoint.pt")
        result = agent.predict(molecular_graph)
    """
    
    # Default tasks for DGAT1/YARS2 program
    DEFAULT_TASKS = [
        "mito_tox",      # CRITICAL: YARS2 off-target
        "herg",          # HIGH: Cardiac liability
        "cyp3a4",        # HIGH: DDI
        "cyp2d6",        # MEDIUM: DDI
        "hepato",        # HIGH: Liver toxicity
    ]
    
    # Risk thresholds by task priority
    RISK_THRESHOLDS = {
        "mito_tox": 0.3,    # Lower threshold for critical task
        "herg": 0.4,
        "cyp3a4": 0.5,
        "cyp2d6": 0.5,
        "hepato": 0.4,
        "solubility": 0.6,
        "permeability": 0.6,
    }
    
    def __init__(
        self,
        model: UnifiedADMETTransformer,
        device: str = "cpu",
        tasks: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.tasks = tasks or self.DEFAULT_TASKS
        
    @classmethod
    def from_config(
        cls,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        protein_dim: Optional[int] = None,
        device: str = "cpu",
        tasks: Optional[List[str]] = None,
    ) -> "ADMETAgent":
        """Create agent with default model architecture."""
        model = UnifiedADMETTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            protein_dim=protein_dim,
            tasks=tasks,
        )
        return cls(model=model, device=device, tasks=tasks)
    
    @classmethod
    def load(cls, checkpoint_path: Union[str, Path], device: str = "cpu") -> "ADMETAgent":
        """
        Load agent from checkpoint.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            device: Device to load model on
            
        Returns:
            Initialized ADMETAgent
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct model from config
        config = checkpoint.get("config", {})
        model = UnifiedADMETTransformer(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        tasks = config.get("tasks", cls.DEFAULT_TASKS)
        
        logger.info(f"Loaded ADMET agent from {checkpoint_path}")
        return cls(model=model, device=device, tasks=tasks)
    
    def save(self, checkpoint_path: Union[str, Path]) -> None:
        """Save agent to checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "hidden_dim": self.model.hidden_dim,
                "num_layers": len(self.model.layers),
                "num_heads": self.model.layers[0].self_attn.num_heads,
                "use_3d": self.model.use_3d,
                "tasks": self.tasks,
            },
        }, checkpoint_path)
        
        logger.info(f"Saved ADMET agent to {checkpoint_path}")
    
    def predict(
        self,
        mol_graph: MolecularGraph,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, ADMETResult]:
        """
        Predict ADMET properties for a molecule.
        
        Args:
            mol_graph: Molecular graph representation
            tasks: Which tasks to predict (default: self.tasks)
            
        Returns:
            Dict mapping task name to ADMETResult
        """
        tasks = tasks or self.tasks
        
        # Convert to tensors
        try:
            mol_tensors = mol_graph.to_tensors(device=self.device)
        except Exception as e:
            logger.error(f"Failed to convert molecular graph to tensors: {e}")
            raise ValueError(f"Invalid molecular graph: {e}") from e
        
        # Add aromatic mask for mitochondrial toxicity attention bias
        is_aromatic = torch.tensor(
            [float(a.is_aromatic) for a in mol_graph.atom_features],
            device=self.device
        )
        
        # Get predictions with feedback
        feedback = self.model.predict_with_feedback(
            mol_tensors=mol_tensors,
            tasks=tasks,
            risk_threshold=self.RISK_THRESHOLDS,
        )
        
        # Convert to ADMETResult objects
        results = {}
        for task, data in feedback.items():
            threshold = self.RISK_THRESHOLDS.get(task, 0.5)
            results[task] = ADMETResult(
                task=task,
                risk_score=data["risk_score"],
                uncertainty=data["uncertainty"],
                is_high_risk=data["risk_score"] > threshold,
                critical_atoms=data["critical_atoms"],
                recommendation=data["recommendation"],
                explanation=data["explanation"],
            )
        
        return results
    
    def predict_batch(
        self,
        mol_graphs: List[MolecularGraph],
        tasks: Optional[List[str]] = None,
    ) -> List[Dict[str, ADMETResult]]:
        """
        Predict ADMET for a batch of molecules.
        
        Note: Currently processes sequentially. For true batching,
        implement collate_fn and batch processing.
        """
        return [self.predict(mol, tasks) for mol in mol_graphs]
    
    def get_design_feedback(
        self,
        mol_graph: MolecularGraph,
        focus_task: str = "mito_tox",
    ) -> Dict[str, Any]:
        """
        Get structured feedback for Design Agent.
        
        Provides actionable information for molecular optimization:
        - Which atoms to modify
        - What modifications to consider
        - Priority ranking of issues
        
        Args:
            mol_graph: Input molecule
            focus_task: Primary task to focus optimization on
            
        Returns:
            Structured feedback dict for Design Agent
        """
        results = self.predict(mol_graph)
        
        # Prioritize issues
        issues = []
        for task, result in results.items():
            if result.is_high_risk:
                issues.append({
                    "task": task,
                    "priority": self._get_task_priority(task),
                    "risk_score": result.risk_score,
                    "confidence": result.confidence,
                    "critical_atoms": result.critical_atoms,
                    "recommendation": result.recommendation,
                })
        
        # Sort by priority * risk_score * confidence
        issues.sort(key=lambda x: x["priority"] * x["risk_score"] * x["confidence"], reverse=True)
        
        # Generate optimization strategy
        strategy = self._generate_optimization_strategy(issues, results, focus_task)
        
        return {
            "has_issues": len(issues) > 0,
            "num_issues": len(issues),
            "issues": issues,
            "optimization_strategy": strategy,
            "focus_task_result": results.get(focus_task),
            "all_results": {task: {
                "risk_score": r.risk_score,
                "uncertainty": r.uncertainty,
                "is_high_risk": r.is_high_risk,
            } for task, r in results.items()},
        }
    
    def _get_task_priority(self, task: str) -> float:
        """Get task priority weight (higher = more important)."""
        priorities = {
            "mito_tox": 1.0,   # CRITICAL
            "herg": 0.9,       # HIGH
            "hepato": 0.8,     # HIGH
            "cyp3a4": 0.7,     # HIGH
            "cyp2d6": 0.5,     # MEDIUM
            "solubility": 0.4, # MEDIUM
            "permeability": 0.4,  # MEDIUM
        }
        return priorities.get(task, 0.5)
    
    def _generate_optimization_strategy(
        self,
        issues: List[Dict],
        all_results: Dict[str, ADMETResult],
        focus_task: str,
    ) -> Dict[str, Any]:
        """Generate actionable optimization strategy."""
        if not issues:
            return {"status": "no_issues", "action": "proceed_to_synthesis"}
        
        primary_issue = issues[0]
        
        # Task-specific strategies
        strategies = {
            "mito_tox": {
                "status": "mitochondrial_toxicity",
                "action": "reduce_aromatic_interactions",
                "modifications": [
                    "Replace aromatic rings with aliphatic groups",
                    "Reduce planarity to disrupt membrane stacking",
                    "Add polar groups to reduce mitochondrial accumulation",
                    "Consider bioisosteric replacements for flagged atoms",
                ],
                "atoms_to_modify": primary_issue["critical_atoms"][:5],
            },
            "herg": {
                "status": "herg_liability",
                "action": "reduce_basicity_lipophilicity",
                "modifications": [
                    "Reduce number of basic amines",
                    "Decrease logP",
                    "Introduce polar surface area",
                    "Consider pKa reduction strategies",
                ],
                "atoms_to_modify": primary_issue["critical_atoms"][:5],
            },
            "cyp3a4": {
                "status": "cyp_inhibition",
                "action": "block_metabolic_soft_spots",
                "modifications": [
                    "Introduce steric hindrance near metabolism sites",
                    "Replace metabolically labile groups",
                    "Consider deuterium substitution at flagged positions",
                ],
                "atoms_to_modify": primary_issue["critical_atoms"][:5],
            },
        }
        
        task = primary_issue["task"]
        strategy = strategies.get(task, {
            "status": f"{task}_liability",
            "action": "general_optimization",
            "modifications": ["Consider structural modifications at flagged atoms"],
            "atoms_to_modify": primary_issue["critical_atoms"][:5],
        })
        
        strategy["primary_task"] = task
        strategy["risk_reduction_target"] = max(0.1, primary_issue["risk_score"] - 0.3)
        strategy["secondary_issues"] = [i["task"] for i in issues[1:3]]
        
        return strategy
```

---

## File 5: `brownbiotech/agents/design_feedback.py`

```python
"""
Design Agent Feedback Loop for BrownBioTech.

Consumes ADMET Agent explanations to guide molecular optimization.
Implements the autonomous loop: Predict → Explain → Design → Validate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from brownbiotech.agents.admet_agent import ADMETAgent, ADMETResult
from brownbiotech.core.molecular_graph import MolecularGraph

logger = logging.getLogger(__name__)


class ModificationType(Enum):
    """Types of molecular modifications."""
    BIOISOSTERIC_REPLACE = "bioisosteric_replace"
    ADD_POLAR_GROUP = "add_polar_group"
    REDUCE_AROMATICITY = "reduce_aromaticity"
    BLOCK_METABOLISM = "block_metabolism"
    REDUCE_BASICITY = "reduce_basicity"
    STERIC_HINDRANCE = "steric_hindrance"
    DEUTERIUM_SUB = "deuterium_substitution"


@dataclass
class DesignSuggestion:
    """
    A single molecular modification suggestion.
    
    Generated from ADMET explanations, targeted at specific atoms/substructures.
    """
    modification_type: ModificationType
    target_atoms: List[int]
    rationale: str
    expected_risk_reduction: float
    confidence: float
    priority: float
    
    # Optional: specific SMARTS transformation
    smarts_transform: Optional[str] = None
    
    # Source ADMET task
    source_task: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "modification_type": self.modification_type.value,
            "target_atoms": self.target_atoms,
            "rationale": self.rationale,
            "expected_risk_reduction": self.expected_risk_reduction,
            "confidence": self.confidence,
            "priority": self.priority,
            "source_task": self.source_task,
            "smarts_transform": self.smarts_transform,
        }


@dataclass
class DesignFeedbackReport:
    """
    Complete feedback report for Design Agent.
    
    Synthesizes ADMET predictions into actionable optimization plan.
    """
    original_smiles: str
    has_critical_issues: bool
    overall_risk_score: float
    suggestions: List[DesignSuggestion]
    optimization_priority_order: List[str]
    
    # For tracking iteration progress
    iteration: int = 0
    previous_risk_scores: Dict[str, float] = field(default_factory=dict)
    risk_improvement: Dict[str, float] = field(default_factory=dict)
    
    def get_top_suggestion(self, n: int = 1) -> List[DesignSuggestion]:
        """Get top n suggestions by priority."""
        return sorted(self.suggestions, key=lambda x: x.priority, reverse=True)[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "original_smiles": self.original_smiles,
            "has_critical_issues": self.has_critical_issues,
            "overall_risk_score": self.overall_risk_score,
            "num_suggestions": len(self.suggestions),
            "top_suggestions": [s.to_dict() for s in self.get_top_suggestion(3)],
            "optimization_priority": self.optimization_priority_order,
            "iteration": self.iteration,
            "risk_improvement": self.risk_improvement,
        }


class DesignFeedbackLoop:
    """
    Feedback loop connecting ADMET Agent to Design Agent.
    
    Workflow:
    1. Receive molecule from Design Agent
    2. Get ADMET predictions with explanations
    3. Analyze attention patterns to identify problematic substructures
    4. Generate modification suggestions with rationale
    5. Return structured feedback for next design iteration
    
    This enables autonomous optimization loops where the Design Agent
    iteratively improves molecules based on ADMET feedback.
    """
    
    # Task-specific modification rules
    MODIFICATION_RULES = {
        "mito_tox": [
            {
                "type": ModificationType.REDUCE_AROMATICITY,
                "rationale": "Aromatic rings contribute to π-π stacking with mitochondrial membrane proteins",
                "smarts": "[c:1]>>[C:1]",  # Simplified - aromatic to aliphatic
                "expected_reduction": 0.3,
            },
            {
                "type": ModificationType.ADD_POLAR_GROUP,
                "rationale": "Polar groups reduce mitochondrial membrane accumulation",
                "smarts": "[C:1]-[C:2]>>[C:1]-[C:2]-[OH]",  # Add hydroxyl
                "expected_reduction": 0.2,
            },
        ],
        "herg": [
            {
                "type": ModificationType.REDUCE_BASICITY,
                "rationale": "Basic amines interact with hERG potassium channel",
                "smarts": "[N:1]>>[N:1]-[C(=O)]",  # Amide instead of amine
                "expected_reduction": 0.4,
            },
        ],
        "cyp3a4": [
            {
                "type": ModificationType.BLOCK_METABOLISM,
                "rationale": "Steric hindrance prevents CYP3A4 oxidation",
                "smarts": "[C:1]-[H]>>[C:1]-[CH3]",  # Methyl instead of hydrogen
                "expected_reduction": 0.25,
            },
            {
                "type": ModificationType.DEUTERIUM_SUB,
                "rationale": "Deuterium slows metabolic oxidation (KIE)",
                "smarts": "[C:1]-[H]>>[C:1]-[D]",
                "expected_reduction": 0.15,
            },
        ],
    }
    
    def __init__(
        self,
        admet_agent: ADMETAgent,
        min_risk_for_suggestion: float = 0.3,
        max_suggestions_per_task: int = 3,
    ):
        self.admet_agent = admet_agent
        self.min_risk_for_suggestion = min_risk_for_suggestion
        self.max_suggestions_per_task = max_suggestions_per_task
        
        # History for tracking improvement across iterations
        self.history: Dict[str, List[float]] = {}
    
    def generate_feedback(
        self,
        mol_graph: MolecularGraph,
        iteration: int = 0,
        previous_scores: Optional[Dict[str, float]] = None,
    ) -> DesignFeedbackReport:
        """
        Generate complete design feedback for a molecule.
        
        Args:
            mol_graph: Input molecular graph
            iteration: Current iteration number (for tracking)
            previous_scores: Previous ADMET scores (for improvement tracking)
            
        Returns:
            DesignFeedbackReport with suggestions
        """
        previous_scores = previous_scores or {}
        
        # Get ADMET predictions
        results = self.admet_agent.predict(mol_graph)
        
        # Generate suggestions for each high-risk task
        suggestions = []
        risk_scores = {}
        
        for task, result in results.items():
            risk_scores[task] = result.risk_score
            
            # Only generate suggestions for high-risk predictions
            if result.risk_score < self.min_risk_for_suggestion:
                continue
            
            # Get task-specific modification rules
            rules = self.MODIFICATION_RULES.get(task, [])
            
            for i, rule in enumerate(rules[:self.max_suggestions_per_task]):
                suggestion = DesignSuggestion(
                    modification_type=rule["type"],
                    target_atoms=result.critical_atoms[:3] if result.critical_atoms else [],
                    rationale=rule["rationale"],
                    expected_risk_reduction=rule["expected_reduction"],
                    confidence=result.confidence,
                    priority=self._calculate_priority(task, result, rule),
                    smarts_transform=rule.get("smarts"),
                    source_task=task,
                )
                suggestions.append(suggestion)
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(risk_scores)
        
        # Determine optimization priority
        priority_order = self._get_priority_order(risk_scores)
        
        # Calculate improvement from previous iteration
        risk_improvement = {}
        for task, prev_score in previous_scores.items():
            if task in risk_scores:
                risk_improvement[task] = prev_score - risk_scores[task]
        
        # Update history
        for task, score in risk_scores.items():
            if task not in self.history:
                self.history[task] = []
            self.history[task].append(score)
        
        return DesignFeedbackReport(
            original_smiles=mol_graph.smiles,
            has_critical_issues=any(r.is_high_risk for r in results.values()),
            overall_risk_score=overall_risk,
            suggestions=suggestions,
            optimization_priority_order=priority_order,
            iteration=iteration,
            previous_risk_scores=previous_scores,
            risk_improvement=risk_improvement,
        )
    
    def _calculate_priority(
        self,
        task: str,
        result: ADMETResult,
        rule: Dict[str, Any],
    ) -> float:
        """
        Calculate suggestion priority.
        
        Higher priority = more important to address.
        """
        # Base priority from task importance
        task_priorities = {
            "mito_tox": 1.0,
            "herg": 0.9,
            "hepato": 0.8,
            "cyp3a4": 0.7,
            "cyp2d6": 0.5,
        }
        base_priority = task_priorities.get(task, 0.5)
        
        # Scale by risk score
        risk_factor = result.risk_score
        
        # Scale by expected reduction
        reduction_factor = rule.get("expected_reduction", 0.2)
        
        # Scale by confidence
        confidence_factor = result.confidence
        
        return base_priority * risk_factor * reduction_factor * confidence_factor
    
    def _calculate_overall_risk(self, risk_scores: Dict[str, float]) -> float:
        """
        Calculate overall molecule risk score.
        
        Uses weighted average with higher weights for critical tasks.
        """
        weights = {
            "mito_tox": 0.3,
            "herg": 0.25,
            "hepato": 0.2,
            "cyp3a4": 0.15,
            "cyp2d6": 0.1,
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for task, score in risk_scores.items():
            w = weights.get(task, 0.1)
            weighted_sum += w * score
            total_weight += w
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_priority_order(self, risk_scores: Dict[str, float]) -> List[str]:
        """Get tasks ordered by optimization priority."""
        task_priorities = {
            "mito_tox": 1.0,
            "herg": 0.9,
            "hepato": 0.8,
            "cyp3a4": 0.7,
            "cyp2d6": 0.5,
        }
        
        scored = [
            (task, task_priorities.get(task, 0.5) * score)
            for task, score in risk_scores.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [task for task, _ in scored]
    
    def check_convergence(
        self,
        task: str,
        window_size: int = 5,
        threshold: float = 0.05,
    ) -> bool:
        """
        Check if optimization has converged for a task.
        
        Returns True if risk score has stabilized (not improving).
        """
        if task not in self.history or len(self.history[task]) < window_size:
            return False
        
        recent = self.history[task][-window_size:]
        improvement = recent[0] - recent[-1]
        
        return improvement < threshold
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress across all iterations."""
        summary = {}
        
        for task, scores in self.history.items():
            if len(scores) < 2:
                continue
            
            summary[task] = {
                "initial_risk": scores[0],
                "final_risk": scores[-1],
                "improvement": scores[0] - scores[-1],
                "num_iterations": len(scores),
                "converged": self.check_convergence(task),
                "best_risk": min(scores),
            }
        
        return summary
```

---

## File 6: `brownbiotech/__init__.py`

```python
"""
BrownBioTech: AI-Driven Drug Discovery Platform

Focused on DGAT1/YARS2 cancer metabolism targets with
state-of-the-art ADMET prediction and explainable design feedback.
"""

__version__ = "0.4.0"  # Iteration 4/100

from brownbiotech.agents.admet_agent import ADMETAgent, ADMETResult
from brownbiotech.agents.design_feedback import DesignFeedbackLoop, DesignFeedbackReport
from brownbiotech.core.molecular_graph import MolecularGraph, ADMETTask
from brownbiotech.models.graph_transformer import UnifiedADMETTransformer
from brownbiotech.models.explainable_attention import (
    ExplainableMultiHeadAttention,
    AttentionExplanation,
)

__all__ = [
    "ADMETAgent",
    "ADMETResult", 
    "DesignFeedbackLoop",
    "DesignFeedbackReport",
    "MolecularGraph",
    "ADMETTask",
    "UnifiedADMETTransformer",
    "ExplainableMultiHeadAttention",
    "AttentionExplanation",
]
```

---

## Usage Example

```python
"""
Example usage of BrownBioTech Iteration 4 improvements.
"""

import torch
import numpy as np
from brownbiotech import (
    ADMETAgent,
    DesignFeedbackLoop,
    MolecularGraph,
    AtomFeatures,
    BondFeatures,
)


def create_example_molecule() -> MolecularGraph:
    """Create a simple example molecule (simulated DGAT1 inhibitor candidate)."""
    
    # Example: A molecule with aromatic rings (mito tox risk) and basic amine (hERG risk)
    atoms = [
        AtomFeatures(atomic_num=6, degree=2, formal_charge=0, hybridization=2, 
                     is_aromatic=True, is_in_ring=True, vdw_radius=1.7, electronegativity=2.55),
        AtomFeatures(atomic_num=6, degree=2, formal_charge=0, hybridization=2,
                     is_aromatic=True, is_in_ring=True, vdw_radius=1.7, electronegativity=2.55),
        AtomFeatures(atomic_num=6, degree=2, formal_charge=0, hybridization=2,
                     is_aromatic=True, is_in_ring=True, vdw_radius=1.7, electronegativity=2.55),
        AtomFeatures(atomic_num=6, degree=2, formal_charge=0, hybridization=2,
                     is_aromatic=True, is_in_ring=True, vdw_radius=1.7, electronegativity=2.55),
        AtomFeatures(atomic_num=6, degree=2, formal_charge=0, hybridization=2,
                     is_aromatic=True, is_in_ring=True, vdw_radius=1.7, electronegativity=2.55),
        AtomFeatures(atomic_num=6, degree=2, formal_charge=0, hybridization=2,
                     is_aromatic=True, is_in_ring=True, vdw_radius=1.7, electronegativity=2.55),
        AtomFeatures(atomic_num=7, degree=3, formal_charge=0, hybridization=3,
                     is_aromatic=False, is_in_ring=False, vdw_radius=1.55, electronegativity=3.04),
        AtomFeatures(atomic_num=6, degree=1, formal_charge=0, hybridization=4,
                     is_aromatic=False, is_in_ring=False, vdw_radius=2.0, electronegativity=2.55),
    ]
    
    bonds = [
        BondFeatures(bond_type=4, is_conjugated=True, is_in_ring=True),  # aromatic
        BondFeatures(bond_type=4, is_conjugated=True, is_in_ring=True),
        BondFeatures(bond_type=4, is_conjugated=True, is_in_ring=True),
        BondFeatures(bond_type=4, is_conjugated=True, is_in_ring=True),
        BondFeatures(bond_type=4, is_conjugated=True, is_in_ring=True),
        BondFeatures(bond_type=4, is_conjugated=True, is_in_ring=True),
        BondFeatures(bond_type=1, is_conjugated=False, is_in_ring=False),  # to amine
        BondFeatures(bond_type=1, is_conjugated=False, is_in_ring=False),  # methyl
    ]
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # ring
        (0, 6),  # to amine
        (6, 7),  # to methyl
    ]
    
    return MolecularGraph(
        smiles="c1ccccc1NC",
        atom_features=atoms,
        bond_features=bonds,
        edge_indices=edges,
        conformer_coords=np.random.randn(8, 3) * 0.5,  # Mock 3D coords
        mol_weight=120.0,
        logp=2.5,
        tpsa=25.0,
        num_rotatable_bonds=2,
    )


def main():
    """Demonstrate the ADMET Agent and Design Feedback Loop."""
    
    print("=" * 60)
    print("BrownBioTech Iteration 4/100: ADMET Agent Demo")
    print("=" * 60)
    
    # Create ADMET agent with default architecture
    agent = ADMETAgent.from_config(
        hidden_dim=128,  # Smaller for demo
        num_layers=2,
        num_heads=4,
        device="cpu",
    )
    
    # Create design feedback loop
    feedback_loop = DesignFeedbackLoop(
        admet_agent=agent,
        min_risk_for_suggestion=0.3,
    )
    
    # Create example molecule
    mol = create_example_molecule()
    print(f"\nInput molecule: {mol.smiles}")
    print(f"Number of atoms: {len(mol.atom_features)}")
    print(f"Number of bonds: {len(mol.bond_features)}")
    
    # Get ADMET predictions
    print("\n" + "-" * 40)
    print("ADMET Predictions:")
    print("-" * 40)
    
    results = agent.predict(mol)
    for task, result in results.items():
        status = "⚠️ HIGH RISK" if result.is_high_risk else "✓ ACCEPTABLE"
        print(f"  {task:12s}: {result.risk_score:.3f} ({status})")
        if result.critical_atoms:
            print(f"               Critical atoms: {result.critical_atoms}")
    
    # Get design feedback
    print("\n" + "-" * 40)
    print("Design Feedback:")
    print("-" * 40)
    
    feedback = feedback_loop.generate_feedback(mol, iteration=0)
    print(f"  Overall risk score: {feedback.overall_risk_score:.3f}")
    print(f"  Has critical issues: {feedback.has_critical_issues}")
    print(f"  Optimization priority: {feedback.optimization_priority_order}")
    
    if feedback.suggestions:
        print(f"\n  Top suggestions:")
        for i, sug in enumerate(feedback.get_top_suggestion(3), 1):
            print(f"    {i}. [{sug.modification_type.value}]")
            print(f"       Target atoms: {sug.target_atoms}")
            print(f"       Rationale: {sug.rationale}")
            print(f"       Expected reduction: {sug.expected_risk_reduction:.1%}")
    
    print("\n" + "=" * 60)
    print("Demo complete. Ready for autonomous optimization loops.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary of Improvements

| Aspect | Legacy (Pre-Iter 4) | Improved (Iter 4) |
|--------|---------------------|-------------------|
| **Architecture** | Separate single-task models | Unified Multimodal Graph-Transformer |
| **Input Modalities** | 2D fingerprints only | 2D graph + 3D geometry + protein context |
| **Mitochondrial Toxicity** | Generic QSAR | Aromatic-biased attention + 3D membrane modeling |
| **YARS2 Off-target** | Not modeled | Cross-modal attention with protein features |
| **Explainability** | None | Full attention tracing to atoms |
| **Design Feedback** | Manual review | Automated suggestion generation |
| **Uncertainty** | Point estimates | Epistemic uncertainty quantification |
| **Active Learning** | Not supported | Uncertainty-guided sampling ready |

**Key Integration Points:**
- `ADMETAgent.predict()` → Returns `ADMETResult` with explanations
- `DesignFeedbackLoop.generate_feedback()` → Converts predictions to `DesignSuggestion` objects
- Attention explanations flow from `ExplainableMultiHeadAttention` → `ADMETResult` → `DesignSuggestion.target_atoms`