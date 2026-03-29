```python
"""
brownbiotech/explainable_metabolism.py

Explainable-AI-driven Metabolism Mapping via Graph Attention Networks (GAT).

This module provides attention-weighted metabolic pathway analysis, enabling
the Design and VirtualScreen agents to understand *why* certain metabolic
routes are predicted as favorable for target compound production.

Integration Points:
    - MultiOmics agent: receives pathway embeddings for multi-omics fusion
    - Design agent: uses attention explanations to guide metabolic engineering
    - VirtualScreen agent: scores compounds based on pathway feasibility
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class NodeType(Enum):
    """Metabolite node types in the pathway graph."""
    METABOLITE = "metabolite"
    ENZYME = "enzyme"
    GENE = "gene"
    REACTION = "reaction"


class EdgeType(Enum):
    """Edge relationship types."""
    REACTANT_TO_REACTION = "reactant_to_reaction"
    REACTION_TO_PRODUCT = "reaction_to_product"
    ENZYME_CATALYZES = "enzyme_catalyzes"
    GENE_ENCODES = "gene_encodes"
    REGULATES = "regulates"


@dataclass
class MetabolicNode:
    """Single node in a metabolic pathway graph."""
    id: str
    node_type: NodeType
    features: np.ndarray  # Shape: (feature_dim,)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetabolicEdge:
    """Single edge in a metabolic pathway graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathwayExplanation:
    """Explainable output from the GAT model."""
    pathway_id: str
    importance_score: float
    node_attentions: Dict[str, float]  # node_id -> attention weight
    edge_attentions: Dict[Tuple[str, str], float]  # (src, tgt) -> weight
    critical_nodes: List[str]  # Top-k important nodes
    critical_edges: List[Tuple[str, str]]  # Top-k important edges
    textual_explanation: str


# =============================================================================
# Pathway Graph Builder
# =============================================================================

class MetabolicPathwayGraph:
    """Constructs and manages metabolic pathway graphs for GAT input."""
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.nodes: Dict[str, MetabolicNode] = {}
        self.edges: List[MetabolicEdge] = []
        self._node_idx_map: Dict[str, int] = {}
        self._edge_type_map = {e: i for i, e in enumerate(EdgeType)}
        
    def add_node(self, node: MetabolicNode) -> None:
        """Add a node to the pathway graph."""
        if node.features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Node {node.id} has feature dim {node.features.shape[0]}, "
                f"expected {self.feature_dim}"
            )
        self.nodes[node.id] = node
        
    def add_edge(self, edge: MetabolicEdge) -> None:
        """Add an edge to the pathway graph."""
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found")
        self.edges.append(edge)
        
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if not self.nodes:
            raise ValueError("Cannot convert empty graph")
            
        # Build index mapping
        self._node_idx_map = {nid: idx for idx, nid in enumerate(self.nodes)}
        
        # Node features with type encoding
        node_features = []
        node_type_onehot = np.zeros((len(NodeType),))
        for node in self.nodes.values():
            node_type_onehot[:] = 0
            node_type_onehot[list(NodeType).index(node.node_type)] = 1
            combined = np.concatenate([node.features, node_type_onehot])
            node_features.append(combined)
            
        x = torch.tensor(np.stack(node_features), dtype=torch.float32)
        
        # Edge indices and attributes
        edge_index = []
        edge_attr = []
        edge_type_dim = len(EdgeType)
        
        for edge in self.edges:
            src_idx = self._node_idx_map[edge.source_id]
            tgt_idx = self._node_idx_map[edge.target_id]
            edge_index.append([src_idx, tgt_idx])
            
            # Edge type one-hot + weight
            type_onehot = np.zeros(edge_type_dim)
            type_onehot[self._edge_type_map[edge.edge_type]] = 1
            attr = np.concatenate([type_onehot, [edge.weight]])
            edge_attr.append(attr)
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    @classmethod
    def from_kegg_json(cls, json_path: Path, feature_dim: int = 64) -> "MetabolicPathwayGraph":
        """Load pathway from KEGG-style JSON export."""
        graph = cls(feature_dim=feature_dim)
        
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load KEGG JSON: {e}")
            
        # Parse nodes
        for node_data in data.get("nodes", []):
            features = np.random.randn(feature_dim).astype(np.float32) * 0.1
            if "embedding" in node_data:
                emb = np.array(node_data["embedding"])
                features[:min(emb.shape[0], feature_dim)] = emb[:feature_dim]
                
            node = MetabolicNode(
                id=node_data["id"],
                node_type=NodeType(node_data.get("type", "metabolite")),
                features=features,
                metadata={k: v for k, v in node_data.items() 
                         if k not in ("id", "type", "embedding")}
            )
            graph.add_node(node)
            
        # Parse edges
        for edge_data in data.get("edges", []):
            edge = MetabolicEdge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=EdgeType(edge_data.get("type", "reactant_to_reaction")),
                weight=edge_data.get("weight", 1.0)
            )
            graph.add_edge(edge)
            
        logger.info(f"Loaded pathway: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph


# =============================================================================
# Graph Attention Network Model
# =============================================================================

class MetabolismGAT(nn.Module):
    """
    Graph Attention Network for metabolic pathway analysis.
    
    Uses multi-head attention to learn which nodes and edges are critical
    for predicting pathway viability and compound production potential.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 32,
        dropout: float = 0.2,
        edge_attr_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.edge_attr_dim = edge_attr_dim or len(EdgeType) + 1
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers with edge attributes
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=self.edge_attr_dim,
                    concat=True
                )
            )
            
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Pathway scoring head
        self.scoring_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self._attention_weights: Dict[int, torch.Tensor] = {}
        
    def forward(
        self,
        data: Data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        """
        Forward pass through the GAT.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr
            return_attention: If True, collect attention weights for explanation
            
        Returns:
            Tuple of (pathway_embedding, attention_weights_dict)
        """
        self._attention_weights = {}
        
        x = self.input_proj(data.x)
        
        for i, gat_layer in enumerate(self.gat_layers):
            x, (edge_idx, attn_weights) = gat_layer(
                x, 
                data.edge_index, 
                edge_attr=data.edge_attr,
                return_attention_weights=True
            )
            
            if return_attention:
                self._attention_weights[i] = attn_weights
                
        # Global pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )
        graph_embedding = global_mean_pool(x, batch)
        
        # Output projection
        output = self.output_proj(graph_embedding)
        
        # Pathway score
        score = self.scoring_head(output)
        
        attentions = self._attention_weights if return_attention else None
        return output, attentions
    
    def get_pathway_score(self, data: Data) -> float:
        """Get scalar pathway viability score."""
        self.eval()
        with torch.no_grad():
            _, _ = self.forward(data)
            # Re-run to get score (simpler than modifying forward)
            x = self.input_proj(data.x)
            for gat_layer in self.gat_layers:
                x = gat_layer(x, data.edge_index, edge_attr=data.edge_attr)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            graph_emb = global_mean_pool(x, batch)
            output = self.output_proj(graph_emb)
            score = self.scoring_head(output)
            return torch.sigmoid(score).item()


# =============================================================================
# Explainability Module
# =============================================================================

class MetabolismExplainer:
    """
    Generates human-readable explanations from GAT attention weights.
    
    Transforms raw attention distributions into actionable insights for
    metabolic engineering decisions.
    """
    
    def __init__(
        self,
        model: MetabolismGAT,
        graph: MetabolicPathwayGraph,
        top_k: int = 5
    ):
        self.model = model
        self.graph = graph
        self.top_k = top_k
        self._idx_to_node = {v: k for k, v in graph._node_idx_map.items()}
        
    def explain(
        self,
        data: Data,
        pathway_id: str = "unknown"
    ) -> PathwayExplanation:
        """Generate explanation for a pathway prediction."""
        self.model.eval()
        
        with torch.no_grad():
            embedding, attentions = self.model(data, return_attention=True)
            score = self.model.get_pathway_score(data)
            
        # Aggregate attention across layers and heads
        node_attentions = self._aggregate_node_attentions(attentions)
        edge_attentions = self._compute_edge_attentions(attentions)
        
        # Identify critical components
        critical_nodes = self._get_top_nodes(node_attentions)
        critical_edges = self._get_top_edges(edge_attentions)
        
        # Generate textual explanation
        textual = self._generate_text(
            pathway_id, score, critical_nodes, critical_edges
        )
        
        return PathwayExplanation(
            pathway_id=pathway_id,
            importance_score=score,
            node_attentions=node_attentions,
            edge_attentions=edge_attentions,
            critical_nodes=critical_nodes,
            critical_edges=critical_edges,
            textual_explanation=textual
        )
    
    def _aggregate_node_attentions(
        self, 
        attentions: Dict[int, torch.Tensor]
    ) -> Dict[str, float]:
        """Aggregate attention weights per node across layers."""
        node_scores: Dict[str, float] = {}
        
        for layer_idx, attn in attentions.items():
            # attn shape: (num_edges, num_heads)
            edge_index = self.graph.to_pyg_data().edge_index
            
            for edge_idx in range(attn.shape[0]):
                src_node_idx = edge_index[0, edge_idx].item()
                node_id = self._idx_to_node.get(src_node_idx)
                if node_id:
                    avg_attn = attn[edge_idx].mean().item()
                    # Weight later layers more (they capture higher-level patterns)
                    layer_weight = 1.0 + 0.5 * layer_idx
                    node_scores[node_id] = node_scores.get(node_id, 0) + avg_attn * layer_weight
                    
        # Normalize
        total = sum(node_scores.values()) or 1.0
        return {k: v / total for k, v in node_scores.items()}
    
    def _compute_edge_attentions(
        self,
        attentions: Dict[int, torch.Tensor]
    ) -> Dict[Tuple[str, str], float]:
        """Compute attention weights per edge."""
        edge_scores: Dict[Tuple[str, str], float] = {}
        edge_index = self.graph.to_pyg_data().edge_index
        
        for layer_idx, attn in attentions.items():
            for edge_idx in range(attn.shape[0]):
                src = self._idx_to_node.get(edge_index[0, edge_idx].item(), "")
                tgt = self._idx_to_node.get(edge_index[1, edge_idx].item(), "")
                if src and tgt:
                    key = (src, tgt)
                    avg_attn = attn[edge_idx].mean().item()
                    layer_weight = 1.0 + 0.5 * layer_idx
                    edge_scores[key] = edge_scores.get(key, 0) + avg_attn * layer_weight
                    
        total = sum(edge_scores.values()) or 1.0
        return {k: v / total for k, v in edge_scores.items()}
    
    def _get_top_nodes(self, attentions: Dict[str, float]) -> List[str]:
        """Get top-k most attended nodes."""
        sorted_nodes = sorted(attentions.items(), key=lambda x: -x[1])
        return [n[0] for n in sorted_nodes[:self.top_k]]
    
    def _get_top_edges(self, attentions: Dict[Tuple[str, str], float]) -> List[Tuple[str, str]]:
        """Get top-k most attended edges."""
        sorted_edges = sorted(attentions.items(), key=lambda x: -x[1])
        return [e[0] for e in sorted_edges[:self.top_k]]
    
    def _generate_text(
        self,
        pathway_id: str,
        score: float,
        critical_nodes: List[str],
        critical_edges: List[Tuple[str, str]]
    ) -> str:
        """Generate human-readable explanation."""
        node_details = []
        for nid in critical_nodes[:3]:
            node = self.graph.nodes.get(nid)
            if node:
                node_details.append(
                    f"{nid} ({node.node_type.value}: {node.metadata.get('name', 'N/A')})"
                )
                
        edge_details = []
        for src, tgt in critical_edges[:2]:
            edge = next((e for e in self.graph.edges if e.source_id == src and e.target_id == tgt), None)
            if edge:
                edge_details.append(f"{src} → {tgt} ({edge.edge_type.value})")
                
        explanation = (
            f"Pathway {pathway_id} has a viability score of {score:.3f}. "
            f"Key metabolic nodes driving this prediction: {', '.join(node_details) if node_details else 'N/A'}. "
            f"Critical reactions: {', '.join(edge_details) if edge_details else 'N/A'}. "
        )
        
        if score > 0.7:
            explanation += "High confidence: this pathway is predicted to be favorable for target production."
        elif score > 0.4:
            explanation += "Moderate confidence: consider optimizing the identified bottleneck nodes."
        else:
            explanation += "Low confidence: significant metabolic engineering may be required."
            
        return explanation


# =============================================================================
# Integration Layer for BrownBioTech Agents
# =============================================================================

class MetabolismMappingInterface:
    """
    Integration interface for BrownBioTech agents.
    
    Provides clean APIs for MultiOmics, Design, and VirtualScreen agents
    to leverage explainable metabolism mapping.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_dim: int = 64,
        device: str = "auto"
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        
        # Initialize model
        input_dim = feature_dim + len(NodeType)  # features + type one-hot
        self.model = MetabolismGAT(input_dim=input_dim).to(self.device)
        
        if model_path and model_path.exists():
            self._load_model(model_path)
        else:
            logger.warning("No model weights found, using random initialization")
            
        self._pathway_cache: Dict[str, Tuple[Data, MetabolicPathwayGraph]] = {}
        
    def _load_model(self, path: Path) -> None:
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def save_model(self, path: Path) -> None:
        """Save model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": self.model.state_dict()},
            path
        )
        logger.info(f"Saved model to {path}")
        
    def load_pathway(self, pathway_id: str, json_path: Path) -> None:
        """Load and cache a metabolic pathway."""
        graph = MetabolicPathwayGraph.from_kegg_json(json_path)
        data = graph.to_pyg_data().to(self.device)
        self._pathway_cache[pathway_id] = (data, graph)
        logger.info(f"Cached pathway: {pathway_id}")
        
    def get_pathway_embedding(self, pathway_id: str) -> np.ndarray:
        """
        Get pathway embedding for MultiOmics fusion.
        
        Returns:
            numpy array of shape (output_dim,)
        """
        if pathway_id not in self._pathway_cache:
            raise KeyError(f"Pathway {pathway_id} not loaded. Call load_pathway first.")
            
        data, _ = self._pathway_cache[pathway_id]
        self.model.eval()
        
        with torch.no_grad():
            embedding, _ = self.model(data)
            
        return embedding.cpu().numpy().flatten()
    
    def get_pathway_score(self, pathway_id: str) -> float:
        """
        Get pathway viability score for VirtualScreen agent.
        
        Returns:
            float between 0 and 1 indicating predicted pathway viability
        """
        if pathway_id not in self._pathway_cache:
            raise KeyError(f"Pathway {pathway_id} not loaded")
            
        data, _ = self._pathway_cache[pathway_id]
        return self.model.get_pathway_score(data)
    
    def explain_pathway(self, pathway_id: str) -> PathwayExplanation:
        """
        Get full explanation for Design agent metabolic engineering decisions.
        
        Returns:
            PathwayExplanation with attention weights and textual summary
        """
        if pathway_id not in self._pathway_cache:
            raise KeyError(f"Pathway {pathway_id} not loaded")
            
        data, graph = self._pathway_cache[pathway_id]
        explainer = MetabolismExplainer(self.model, graph)
        return explainer.explain(data, pathway_id)
    
    def compare_pathways(
        self, 
        pathway_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple pathways for compound production potential.
        
        Useful for VirtualScreen agent when evaluating alternative biosynthetic routes.
        """
        results = []
        for pid in pathway_ids:
            try:
                score = self.get_pathway_score(pid)
                explanation = self.explain_pathway(pid)
                results.append({
                    "pathway_id": pid,
                    "score": score,
                    "critical_nodes": explanation.critical_nodes,
                    "explanation": explanation.textual_explanation
                })
            except Exception as e:
                logger.error(f"Failed to analyze pathway {pid}: {e}")
                results.append({
                    "pathway_id": pid,
                    "score": 0.0,
                    "error": str(e)
                })
                
        # Sort by score descending
        results.sort(key=lambda x: -x["score"])
        
        return {
            "ranked_pathways": results,
            "recommended": results[0]["pathway_id"] if results else None,
            "score_delta": results[0]["score"] - results[1]["score"] if len(results) > 1 else 0
        }


# =============================================================================
# Example Usage & Testing
# =============================================================================

def create_example_pathway() -> MetabolicPathwayGraph:
    """Create a small example pathway for testing."""
    graph = MetabolicPathwayGraph(feature_dim=64)
    
    # Add metabolite nodes
    for i, name in enumerate(["glucose", "g6p", "f6p", "pyruvate", "acetyl_coa", "target_compound"]):
        features = np.random.randn(64).astype(np.float32) * 0.1
        graph.add_node(MetabolicNode(
            id=f"met_{i}",
            node_type=NodeType.METABOLITE,
            features=features,
            metadata={"name": name}
        ))
    
    # Add enzyme nodes
    for i, name in enumerate(["hexokinase", "pfk", "pyruvate_kinase", "pdh", "synthase"]):
        features = np.random.randn(64).astype(np.float32) * 0.1
        graph.add_node(MetabolicNode(
            id=f"enz_{i}",
            node_type=NodeType.ENZYME,
            features=features,
            metadata={"name": name}
        ))
    
    # Add edges (simplified linear pathway)
    graph.add_edge(MetabolicEdge("met_0", "enz_0", EdgeType.REACTANT_TO_REACTION, weight=1.0))
    graph.add_edge(MetabolicEdge("enz_0", "met_1", EdgeType.REACTION_TO_PRODUCT, weight=0.9))
    graph.add_edge(MetabolicEdge("met_1", "enz_1", EdgeType.REACTANT_TO_REACTION, weight=0.8))
    graph.add_edge(MetabolicEdge("enz_1", "met_2", EdgeType.REACTION_TO_PRODUCT, weight=0.85))
    graph.add_edge(MetabolicEdge("met_2", "enz_2", EdgeType.REACTANT_TO_REACTION, weight=0.7))
    graph.add_edge(MetabolicEdge("enz_2", "met_3", EdgeType.REACTION_TO_PRODUCT, weight=0.75))
    graph.add_edge(MetabolicEdge("met_3", "enz_3", EdgeType.REACTANT_TO_REACTION, weight=0.9))
    graph.add_edge(MetabolicEdge("enz_3", "met_4", EdgeType.REACTION_TO_PRODUCT, weight=0.95))
    graph.add_edge(MetabolicEdge("met_4", "enz_4", EdgeType.REACTANT_TO_REACTION, weight=0.6))  # Bottleneck
    graph.add_edge(MetabolicEdge("enz_4", "met_5", EdgeType.REACTION_TO_PRODUCT, weight=0.5))
    
    return graph


def demo_usage():
    """Demonstrate the explainable metabolism mapping module."""
    print("=" * 60)
    print("BrownBioTech Explainable Metabolism Mapping Demo")
    print("=" * 60)
    
    # Create interface
    interface = MetabolismMappingInterface(feature_dim=64)
    
    # Create and load example pathway
    graph = create_example_pathway()
    data = graph.to_pyg_data()
    interface._pathway_cache["glycolysis_to_target"] = (data.to(interface.device), graph)
    
    # 1. Get pathway embedding (for MultiOmics agent)
    print("\n[MultiOmics Integration] Pathway Embedding:")
    embedding = interface.get_pathway_embedding("glycolysis_to_target")
    print(f"  Shape: {embedding.shape}")
    print(f"  Sample values: {embedding[:5].round(4)}")
    
    # 2. Get pathway score (for VirtualScreen agent)
    print("\n[VirtualScreen Integration] Pathway Viability Score:")
    score = interface.get_pathway_score("glycolysis_to_target")
    print(f"  Score: {score:.4f}")
    
    # 3. Get full explanation (for Design agent)
    print("\n[Design Agent Integration] Pathway Explanation:")
    explanation = interface.explain_pathway("glycolysis_to_target")
    print(f"  {explanation.textual_explanation}")
    print(f"\n  Critical Nodes (attention-weighted):")
    for node_id in explanation.critical_nodes:
        node = graph.nodes[node_id]
        attn = explanation.node_attentions[node_id]
        print(f"    - {node_id}: {node.metadata.get('name', 'N/A')} (attention: {attn:.4f})")
    
    print(f"\n  Critical Edges:")
    for src, tgt in explanation.critical_edges:
        attn = explanation.edge_attentions[(src, tgt)]
        print(f"    - {src} → {tgt} (attention: {attn:.4f})")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_usage()
```

## Explanation of Improvement

**What this implements:**

This module adds **Explainable-AI-driven metabolism mapping** to BrownBioTech using Graph Attention Networks (GATs). At Iteration 10/100, this is the highest-ROI improvement because:

1. **Immediate Value**: Provides actionable insights without requiring massive compute or legal agreements
2. **Agent Integration**: Clean interfaces for all three downstream agents:
   - `MultiOmics`: `get_pathway_embedding()` returns vectors for multi-omics fusion
   - `VirtualScreen`: `get_pathway_score()` returns 0-1 viability scores
   - `Design`: `explain_pathway()` returns attention-weighted explanations identifying metabolic bottlenecks

3. **Explainability**: The `MetabolismExplainer` class transforms raw attention weights into:
   - Ranked critical nodes (which metabolites/enzymes matter most)
   - Ranked critical edges (which reactions are bottlenecks)
   - Human-readable textual explanations

**Key design decisions:**
- Uses PyTorch Geometric's `GATConv` with edge attributes (captures reaction types)
- Multi-head attention with layer-weighted aggregation (later layers weighted higher)
- Caching system for efficient repeated pathway queries
- Graceful fallback to random initialization if no pretrained weights exist

**To integrate with existing codebase:**
```python
from brownbiotech.explainable_metabolism import MetabolismMappingInterface

# In your MultiOmics agent
metabolism_interface = MetabolismMappingInterface(model_path=Path("models/metabolism_gat.pt"))
metabolism_interface.load_pathway("pathway_001", Path("data/pathways/pathway_001.json"))
pathway_emb = metabolism_interface.get_pathway_embedding("pathway_001")
# Use pathway_emb in your multi-omics fusion layer
```