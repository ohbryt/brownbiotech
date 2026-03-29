# BrownBioTech Iteration 19→20: Multi-Target Metabolic Rewiring Module

## File 1: `arp_v3/agents/virtual_screen/multi_target_scorer.py`

```python
"""
Multi-Target Scorer for Dual DGAT1/YARS2 Inhibition.

Computes combined binding affinity scores across multiple metabolic targets
using weighted ensemble approach with synergy detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, validator


class TargetType(Enum):
    """Metabolic target types for multi-target screening."""
    DGAT1 = "DGAT1"  # Diacylglycerol O-acyltransferase 1
    YARS2 = "YARS2"  # Tyrosyl-tRNA synthetase, mitochondrial
    ACC1 = "ACC1"    # Acetyl-CoA carboxylase
    FASN = "FASN"    # Fatty acid synthase
    CPT1 = "CPT1"    # Carnitine palmitoyltransferase 1


class SynergyMode(Enum):
    """Synergy detection modes for multi-target scoring."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    MAXIMUM = "maximum"
    BLISS = "bliss"  # Bliss independence model


@dataclass
class TargetBindingResult:
    """Binding result for a single target."""
    target: TargetType
    docking_score: float  # Negative = better binding (kcal/mol)
    confidence: float     # 0.0 to 1.0
    pose_count: int = 9
    best_rmsd: float = 0.0
    
    @property
    def normalized_score(self) -> float:
        """Normalize score to 0-1 range (higher = better)."""
        # Typical docking scores range from -12 to 0
        raw = max(-12.0, min(0.0, self.docking_score))
        return (abs(raw) / 12.0) * self.confidence


@dataclass
class MultiTargetScore:
    """Combined multi-target scoring result."""
    compound_id: str
    individual_scores: dict[TargetType, TargetBindingResult]
    combined_score: float
    synergy_factor: float
    synergy_mode: SynergyMode
    target_coverage: float  # Fraction of targets with score > threshold
    
    @property
    def is_synergistic(self) -> bool:
        """Check if combination shows synergistic effect."""
        return self.synergy_factor > 1.2
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "combined_score": round(self combined_score, 4),
            "synergy_factor": round(self.synergy_factor, 4),
            "synergy_mode": self.synergy_mode.value,
            "target_coverage": round(self.target_coverage, 4),
            "is_synergistic": self.is_synergistic,
            "individual_scores": {
                t.value: {"score": r.docking_score, "confidence": r.confidence}
                for t, r in self.individual_scores.items()
            }
        }


class MultiTargetScorer:
    """
    Multi-target scorer for metabolic rewiring drug discovery.
    
    Combines docking scores across DGAT1, YARS2, and other metabolic
    targets using configurable synergy models.
    
    Attributes:
        target_weights: Weight for each target in combined scoring
        synergy_mode: Method for combining scores
        score_threshold: Minimum normalized score to count as "hit"
    """
    
    # Default weights for metabolic rewiring (DGAT1/YARS2 focused)
    DEFAULT_WEIGHTS: dict[TargetType, float] = {
        TargetType.DGAT1: 0.40,
        TargetType.YARS2: 0.35,
        TargetType.ACC1: 0.10,
        TargetType.FASN: 0.10,
        TargetType.CPT1: 0.05,
    }
    
    def __init__(
        self,
        target_weights: Optional[dict[TargetType, float]] = None,
        synergy_mode: SynergyMode = SynergyMode.BLISS,
        score_threshold: float = 0.3,
    ):
        self.target_weights = target_weights or self.DEFAULT_WEIGHTS.copy()
        self.synergy_mode = synergy_mode
        self.score_threshold = score_threshold
        self._validate_weights()
    
    def _validate_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(self.target_weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Target weights must sum to 1.0, got {total}")
    
    def _compute_synergy(
        self,
        scores: list[float],
        weights: list[float],
    ) -> float:
        """
        Compute synergy factor based on selected mode.
        
        Args:
            scores: Normalized scores for each target
            weights: Corresponding target weights
            
        Returns:
            Synergy factor (>1.0 indicates synergy)
        """
        if len(scores) < 2:
            return 1.0
        
        expected = sum(s * w for s, w in zip(scores, weights))
        
        if self.synergy_mode == SynergyMode.ADDITIVE:
            actual = sum(scores) / len(scores)
        elif self.synergy_mode == SynergyMode.MULTIPLICATIVE:
            actual = np.prod([1 + s for s in scores]) - 1
            expected = np.prod([1 + s * w for s, w in zip(scores, weights)]) - 1
        elif self.synergy_mode == SynergyMode.MAXIMUM:
            actual = max(scores)
        elif self.synergy_mode == SynergyMode.BLISS:
            # Bliss independence: E_bliss = E_a + E_b - E_a * E_b
            actual = 1.0
            for s in scores:
                actual *= (1 - s)
            actual = 1 - actual
        else:
            actual = expected
        
        if expected < 1e-10:
            return 1.0
        
        return actual / expected
    
    def score_compound(
        self,
        compound_id: str,
        binding_results: list[TargetBindingResult],
    ) -> MultiTargetScore:
        """
        Compute multi-target score for a compound.
        
        Args:
            compound_id: Unique identifier for the compound
            binding_results: List of binding results for each target
            
        Returns:
            MultiTargetScore with combined metrics
            
        Raises:
            ValueError: If no binding results provided
        """
        if not binding_results:
            raise ValueError(f"No binding results for compound {compound_id}")
        
        # Build score dictionary
        score_dict = {r.target: r for r in binding_results}
        
        # Get scores and weights for available targets
        scores = []
        weights = []
        for target, weight in self.target_weights.items():
            if target in score_dict:
                scores.append(score_dict[target].normalized_score)
                weights.append(weight)
        
        if not scores:
            raise ValueError(f"No matching targets found for {compound_id}")
        
        # Normalize weights for available targets
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Compute combined score
        combined = sum(s * w for s, w in zip(scores, weights))
        
        # Compute synergy
        synergy = self._compute_synergy(scores, weights)
        
        # Compute target coverage
        hits = sum(1 for s in scores if s > self.score_threshold)
        coverage = hits / len(self.target_weights)
        
        return MultiTargetScore(
            compound_id=compound_id,
            individual_scores=score_dict,
            combined_score=combined,
            synergy_factor=synergy,
            synergy_mode=self.synergy_mode,
            target_coverage=coverage,
        )
    
    def score_compounds_batch(
        self,
        compound_results: dict[str, list[TargetBindingResult]],
    ) -> list[MultiTargetScore]:
        """
        Score multiple compounds in batch.
        
        Args:
            compound_results: Dict mapping compound_id to binding results
            
        Returns:
            List of MultiTargetScore sorted by combined_score descending
        """
        results = []
        for compound_id, bindings in compound_results.items():
            try:
                score = self.score_compound(compound_id, bindings)
                results.append(score)
            except ValueError as e:
                continue  # Skip compounds with insufficient data
        
        return sorted(results, key=lambda x: x.combined_score, reverse=True)
    
    def filter_synergistic(
        self,
        scores: list[MultiTargetScore],
        min_synergy: float = 1.2,
        min_coverage: float = 0.5,
    ) -> list[MultiTargetScore]:
        """
        Filter for compounds showing synergistic multi-target effects.
        
        Args:
            scores: List of multi-target scores
            min_synergy: Minimum synergy factor
            min_coverage: Minimum target coverage fraction
            
        Returns:
            Filtered list of synergistic compounds
        """
        return [
            s for s in scores
            if s.synergy_factor >= min_synergy and s.target_coverage >= min_coverage
        ]


# Example usage
if __name__ == "__main__":
    scorer = MultiTargetScorer(synergy_mode=SynergyMode.BLISS)
    
    # Simulate binding results for a compound
    results = [
        TargetBindingResult(TargetType.DGAT1, -9.2, 0.85),
        TargetBindingResult(TargetType.YARS2, -8.7, 0.78),
        TargetBindingResult(TargetType.ACC1, -6.1, 0.65),
    ]
    
    score = scorer.score_compound("BBT-001", results)
    print(f"Compound: {score.compound_id}")
    print(f"Combined Score: {score.combined_score:.4f}")
    print(f"Synergy Factor: {score.synergy_factor:.4f}")
    print(f"Is Synergistic: {score.is_synergistic}")
    print(f"Target Coverage: {score.target_coverage:.2%}")
```

---

## File 2: `arp_v3/agents/virtual_screen/metabolic_flux_gnn.py`

```python
"""
Metabolic Flux GNN Predictor.

Graph neural network for predicting metabolic flux changes
upon target inhibition using reaction network topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ReactionNode:
    """Represents a metabolic reaction in the graph."""
    reaction_id: str
    enzyme: str
    substrate_ids: list[str]
    product_ids: list[str]
    base_flux: float  # Baseline flux value (mmol/gDW/h)
    reversibility: bool = False


@dataclass
class MetaboliteNode:
    """Represents a metabolite in the graph."""
    metabolite_id: str
    compartment: str  # cytosol, mitochondria, etc.
    is_currency: bool = False  # ATP, NADH, etc.


@dataclass
class FluxPrediction:
    """Prediction result for metabolic flux changes."""
    reaction_id: str
    predicted_flux: float
    fold_change: float
    confidence: float
    is_significant: bool  # |fold_change| > 1.5


class MetabolicGraphBuilder:
    """
    Builds graph representations of metabolic networks.
    
    Creates bipartite graphs with reaction and metabolite nodes,
    connected by substrate/product edges.
    """
    
    def __init__(self, include_currency: bool = False):
        self.include_currency = include_currency
        self._reactions: list[ReactionNode] = []
        self._metabolites: dict[str, MetaboliteNode] = {}
        self._edges: list[tuple[int, int]] = []
        self._edge_types: list[int] = []  # 0: substrate, 1: product
    
    def add_reaction(self, reaction: ReactionNode) -> None:
        """Add a reaction to the graph."""
        self._reactions.append(reaction)
        
        for sub_id in reaction.substrate_ids:
            if sub_id not in self._metabolites:
                self._metabolites[sub_id] = MetaboliteNode(
                    metabolite_id=sub_id,
                    compartment="unknown",
                    is_currency=sub_id in {"atp", "adp", "nadh", "nad", "gtp", "gdp"}
                )
        
        for prod_id in reaction.product_ids:
            if prod_id not in self._metabolites:
                self._metabolites[prod_id] = MetaboliteNode(
                    metabolite_id=prod_id,
                    compartment="unknown",
                    is_currency=prod_id in {"atp", "adp", "nadh", "nad", "gtp", "gdp"}
                )
    
    def build_graph(self) -> Optional[Data]:
        """
        Build PyTorch Geometric Data object.
        
        Returns:
            Data object with node features and edge indices,
            or None if PyTorch not available.
        """
        if not TORCH_AVAILABLE:
            return None
        
        # Filter metabolites
        metabolites = {
            k: v for k, v in self._metabolites.items()
            if self.include_currency or not v.is_currency
        }
        
        n_reactions = len(self._reactions)
        n_metabolites = len(metabolites)
        n_nodes = n_reactions + n_metabolites
        
        if n_nodes == 0:
            return None
        
        # Build node features
        # Reaction features: [base_flux, is_reversible, one_hot_enzyme(8)]
        reaction_features = []
        for rxn in self._reactions:
            feat = [
                np.tanh(rxn.base_flux / 10.0),  # Normalized flux
                float(rxn.reversibility),
            ]
            # Simple enzyme hash encoding (8 dims)
            enzyme_hash = hash(rxn.enzyme) % 256
            for i in range(8):
                feat.append(float((enzyme_hash >> i) & 1))
            reaction_features.append(feat)
        
        # Metabolite features: [is_currency, compartment_one_hot(4)]
        metabolite_features = []
        compartment_map = {"cytosol": 0, "mitochondria": 1, "er": 2, "unknown": 3}
        for met in metabolites.values():
            feat = [float(met.is_currency)]
            comp_onehot = [0.0, 0.0, 0.0, 0.0]
            comp_idx = compartment_map.get(met.compartment, 3)
            comp_onehot[comp_idx] = 1.0
            feat.extend(comp_onehot)
            metabolite_features.append(feat)
        
        # Combine features
        x = torch.tensor(
            reaction_features + metabolite_features,
            dtype=torch.float32
        )
        
        # Build edges (metabolite -> reaction for substrates, reaction -> metabolite for products)
        met_idx_map = {mid: i + n_reactions for i, mid in enumerate(metabolites.keys())}
        edge_index = [[], []]
        edge_attr = []
        
        for rxn_idx, rxn in enumerate(self._reactions):
            for sub_id in rxn.substrate_ids:
                if sub_id in met_idx_map:
                    met_i = met_idx_map[sub_id]
                    edge_index[0].append(met_i)
                    edge_index[1].append(rxn_idx)
                    edge_attr.append([1.0, 0.0])  # substrate edge
            
            for prod_id in rxn.product_ids:
                if prod_id in met_idx_map:
                    met_i = met_idx_map[prod_id]
                    edge_index[0].append(rxn_idx)
                    edge_index[1].append(met_i)
                    edge_attr.append([0.0, 1.0])  # product edge
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Target: flux values for reactions
        y = torch.tensor([r.base_flux for r in self._reactions], dtype=torch.float32)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            n_reactions=n_reactions,
        )


class FluxPredictorGNN(nn.Module):
    """
    GNN model for predicting metabolic flux changes.
    
    Uses graph attention layers to learn reaction-metabolite
    interactions and predict flux perturbation upon inhibition.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 12,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                edge_dim=2,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.flux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        data: Data,
        inhibition_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for flux prediction.
        
        Args:
            data: PyTorch Geometric Data object
            inhibition_mask: Binary mask for inhibited reactions
            
        Returns:
            Tuple of (predicted_flux, confidence)
        """
        x = self.node_encoder(data.x)
        
        for gat in self.gat_layers:
            x = F.elu(gat(x, data.edge_index, data.edge_attr))
        
        # Extract reaction node embeddings
        n_rxn = data.n_reactions if hasattr(data, 'n_reactions') else data.y.shape[0]
        rxn_embeddings = x[:n_rxn]
        
        # Apply inhibition mask if provided
        if inhibition_mask is not None:
            rxn_embeddings = rxn_embeddings * (1 - inhibition_mask.unsqueeze(-1))
        
        predicted_flux = self.flux_head(rxn_embeddings).squeeze(-1)
        confidence = self.confidence_head(rxn_embeddings).squeeze(-1)
        
        return predicted_flux, confidence


class MetabolicFluxPredictor:
    """
    High-level interface for metabolic flux prediction.
    
    Handles model loading, graph construction, and inference
    for predicting flux changes upon target inhibition.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[FluxPredictorGNN] = None
        self.graph_builder = MetabolicGraphBuilder()
        self._reactions: list[ReactionNode] = []
        
        if TORCH_AVAILABLE and model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str) -> None:
        """Load pretrained model from path."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for model loading")
        
        self.model = FluxPredictorGNN()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
    
    def add_metabolic_pathway(
        self,
        reactions: list[ReactionNode],
    ) -> None:
        """Add reactions from a metabolic pathway."""
        self._reactions.extend(reactions)
        for rxn in reactions:
            self.graph_builder.add_reaction(rxn)
    
    def predict_flux_change(
        self,
        inhibited_enzymes: list[str],
        base_fluxes: Optional[dict[str, float]] = None,
    ) -> list[FluxPrediction]:
        """
        Predict flux changes upon enzyme inhibition.
        
        Args:
            inhibited_enzymes: List of enzyme names to inhibit
            base_fluxes: Optional dict mapping reaction_id to base flux
            
        Returns:
            List of FluxPrediction for each reaction
        """
        if not TORCH_AVAILABLE:
            return self._mock_prediction(inhibited_enzymes, base_fluxes)
        
        graph = self.graph_builder.build_graph()
        if graph is None:
            return []
        
        # Build inhibition mask
        inhibition_mask = torch.zeros(len(self._reactions), dtype=torch.float32)
        for i, rxn in enumerate(self._reactions):
            if rxn.enzyme in inhibited_enzymes:
                inhibition_mask[i] = 1.0
        
        # Use mock model if no pretrained model loaded
        if self.model is None:
            self.model = FluxPredictorGNN()
            self.model.eval()
        
        with torch.no_grad():
            predicted_flux, confidence = self.model(graph, inhibition_mask)
        
        predictions = []
        for i, rxn in enumerate(self._reactions):
            base = base_fluxes.get(rxn.reaction_id, rxn.base_flux) if base_fluxes else rxn.base_flux
            pred = predicted_flux[i].item()
            conf = confidence[i].item()
            
            fold_change = pred / base if abs(base) > 1e-10 else 1.0
            
            predictions.append(FluxPrediction(
                reaction_id=rxn.reaction_id,
                predicted_flux=pred,
                fold_change=fold_change,
                confidence=conf,
                is_significant=abs(fold_change - 1.0) > 0.5,
            ))
        
        return predictions
    
    def _mock_prediction(
        self,
        inhibited_enzymes: list[str],
        base_fluxes: Optional[dict[str, float]] = None,
    ) -> list[FluxPrediction]:
        """Generate mock predictions when PyTorch unavailable."""
        predictions = []
        for rxn in self._reactions:
            base = base_fluxes.get(rxn.reaction_id, rxn.base_flux) if base_fluxes else rxn.base_flux
            
            if rxn.enzyme in inhibited_enzymes:
                # Direct inhibition effect
                fold_change = 0.1 + np.random.random() * 0.3
                confidence = 0.8 + np.random.random() * 0.2
            else:
                # Indirect/downstream effect
                fold_change = 0.7 + np.random.random() * 0.6
                confidence = 0.5 + np.random.random() * 0.3
            
            predictions.append(FluxPrediction(
                reaction_id=rxn.reaction_id,
                predicted_flux=base * fold_change,
                fold_change=fold_change,
                confidence=confidence,
                is_significant=abs(fold_change - 1.0) > 0.5,
            ))
        
        return predictions
    
    def get_affected_pathways(
        self,
        predictions: list[FluxPrediction],
        threshold: float = 1.5,
    ) -> dict[str, list[str]]:
        """
        Identify significantly affected reactions.
        
        Args:
            predictions: List of flux predictions
            threshold: Fold change threshold for significance
            
        Returns:
            Dict with 'increased' and 'decreased' reaction lists
        """
        affected = {"increased": [], "decreased": []}
        
        for pred in predictions:
            if pred.is_significant:
                if pred.fold_change > threshold:
                    affected["increased"].append(pred.reaction_id)
                elif pred.fold_change < (1.0 / threshold):
                    affected["decreased"].append(pred.reaction_id)
        
        return affected


# Example usage
if __name__ == "__main__":
    predictor = MetabolicFluxPredictor()
    
    # Add lipid metabolism pathway reactions
    reactions = [
        ReactionNode("R001", "DGAT1", ["dag", "acyl_coa"], ["tg"], 2.5),
        ReactionNode("R002", "YARS2", ["tyr", "atp"], ["tyr_tRNA", "amp", "ppi"], 1.8),
        ReactionNode("R003", "ACC1", ["acetyl_coa", "co2", "atp"], ["malonyl_coa", "adp", "pi"], 3.2),
        ReactionNode("R004", "FASN", ["malonyl_coa", "nadph"], ["palmitate", "co2", "nadp"], 1.5),
        ReactionNode("R005", "CPT1", ["palmitoyl_coa", "carnitine"], ["palmitoyl_carnitine", "coa"], 0.8),
    ]
    
    predictor.add_metabolic_pathway(reactions)
    
    # Predict flux changes upon DGAT1 inhibition
    predictions = predictor.predict_flux_change(["DGAT1"])
    
    print("Flux Predictions after DGAT1 inhibition:")
    print("-" * 60)
    for pred in predictions:
        status = "↓" if pred.fold_change < 0.5 else ("↑" if pred.fold_change > 1.5 else "→")
        print(f"{pred.reaction_id}: {pred.fold_change:.2f}x {status} (conf: {pred.confidence:.2f})")
    
    affected = predictor.get_affected_pathways(predictions)
    print(f"\nDecreased reactions: {affected['decreased']}")
    print(f"Increased reactions: {affected['increased']}")
```

---

## File 3: `arp_v3/agents/virtual_screen/repurposing_validator.py`

```python
"""
FDA Drug Repurposing Validator.

Validates FDA-approved drug hits for metabolic target engagement
using drug databases and off-target prediction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class DrugStatus(Enum):
    """FDA approval status categories."""
    APPROVED = "approved"
    WITHDRAWN = "withdrawn"
    PHASE_III = "phase_iii"
    PHASE_II = "phase_ii"
    PHASE_I = "phase_i"
    INVESTIGATIONAL = "investigational"
    EXPERIMENTAL = "experimental"


class ValidationOutcome(Enum):
    """Validation outcome categories."""
    VALIDATED = "validated"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    UNLIKELY = "unlikely"
    REJECTED = "rejected"


@dataclass
class FDADrug:
    """FDA-approved drug entry."""
    drugbank_id: str
    generic_name: str
    brand_names: list[str]
    approval_status: DrugStatus
    indication: str
    mechanism_of_action: str
    target_proteins: list[str]
    smiles: Optional[str] = None
    molecular_weight: float = 0.0
    logp: float = 0.0
    known_off_targets: list[str] = field(default_factory=list)
    black_box_warning: bool = False
    pregnancy_category: str = ""


@dataclass
class RepurposingValidation:
    """Validation result for a repurposing candidate."""
    drug: FDADrug
    outcome: ValidationOutcome
    confidence: float
    target_relevance_score: float
    safety_score: float
    druglikeness_score: float
    novelty_score: float
    flags: list[str] = field(default_factory=list)
    rationale: str = ""
    
    @property
    def overall_score(self) -> float:
        """Weighted overall repurposing score."""
        return (
            0.35 * self.target_relevance_score +
            0.25 * self.safety_score +
            0.20 * self.druglikeness_score +
            0.20 * self.novelty_score
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "drugbank_id": self.drug.drugbank_id,
            "generic_name": self.drug.generic_name,
            "outcome": self.outcome.value,
            "confidence": round(self.confidence, 3),
            "overall_score": round(self.overall_score, 3),
            "target_relevance": round(self.target_relevance_score, 3),
            "safety": round(self.safety_score, 3),
            "druglikeness": round(self.druglikeness_score, 3),
            "novelty": round(self.novelty_score, 3),
            "flags": self.flags,
            "rationale": self.rationale,
        }


class FDADrugDatabase:
    """
    In-memory FDA drug database for repurposing validation.
    
    Can be initialized from JSON file or used with built-in
    metabolic-relevant drug subset.
    """
    
    # Built-in metabolic-relevant drugs
    BUILTIN_DRUGS: dict[str, dict[str, Any]] = {
        "DB01636": {
            "generic_name": "Trimetazidine",
            "brand_names": ["Vastarel", "Carvidon"],
            "approval_status": "approved",
            "indication": "Angina pectoris",
            "mechanism_of_action": "Inhibits fatty acid oxidation, shifts metabolism to glucose",
            "target_proteins": ["CPT1", "3-ketoacyl-CoA thiolase"],
            "known_off_targets": ["DGAT1"],
            "black_box_warning": False,
        },
        "DB00473": {
            "generic_name": "Etomoxir",
            "brand_names": [],
            "approval_status": "investigational",
            "indication": "Heart failure (investigational)",
            "mechanism_of_action": "Irreversible CPT1 inhibitor",
            "target_proteins": ["CPT1"],
            "known_off_targets": ["ACC1"],
            "black_box_warning": False,
        },
        "DB01744": {
            "generic_name": "Teglicar",
            "brand_names": [],
            "approval_status": "investigational",
            "indication": "Type 2 diabetes (investigational)",
            "mechanism_of_action": "CPT1A selective inhibitor",
            "target_proteins": ["CPT1A"],
            "known_off_targets": [],
            "black_box_warning": False,
        },
        "DB08895": {
            "generic_name": "Firsocostat",
            "brand_names": [],
            "approval_status": "phase_ii",
            "indication": "NASH (investigational)",
            "mechanism_of_action": "ACC1/ACC2 inhibitor",
            "target_proteins": ["ACC1", "ACC2"],
            "known_off_targets": ["FASN"],
            "black_box_warning": False,
        },
        "DB04542": {
            "generic_name": "Orlistat",
            "brand_names": ["Xenical", "Alli"],
            "approval_status": "approved",
            "indication": "Obesity",
            "mechanism_of_action": "Inhibits gastric and pancreatic lipases",
            "target_proteins": ["Pancreatic lipase"],
            "known_off_targets": ["FASN", "DGAT1"],
            "black_box_warning": False,
        },
        "DB04895": {
            "generic_name": "TVB-2640",
            "brand_names": [],
            "approval_status": "phase_ii",
            "indication": "Cancer, NASH",
            "mechanism_of_action": "FASN inhibitor",
            "target_proteins": ["FASN"],
            "known_off_targets": [],
            "black_box_warning": False,
        },
    }
    
    def __init__(self, custom_db_path: Optional[Path] = None):
        self._drugs: dict[str, FDADrug] = {}
        self._load_builtin()
        
        if custom_db_path and custom_db_path.exists():
            self._load_custom(custom_db_path)
    
    def _load_builtin(self) -> None:
        """Load built-in metabolic drug database."""
        for drugbank_id, data in self.BUILTIN_DRUGS.items():
            self._drugs[drugbank_id] = FDADrug(
                drugbank_id=drugbank_id,
                generic_name=data["generic_name"],
                brand_names=data["brand_names"],
                approval_status=DrugStatus(data["approval_status"]),
                indication=data["indication"],
                mechanism_of_action=data["mechanism_of_action"],
                target_proteins=data["target_proteins"],
                known_off_targets=data.get("known_off_targets", []),
                black_box_warning=data.get("black_box_warning", False),
            )
    
    def _load_custom(self, path: Path) -> None:
        """Load custom drug database from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            for entry in data:
                drug = FDADrug(
                    drugbank_id=entry.get("drugbank_id", ""),
                    generic_name=entry.get("generic_name", ""),
                    brand_names=entry.get("brand_names", []),
                    approval_status=DrugStatus(entry.get("approval_status", "experimental")),
                    indication=entry.get("indication", ""),
                    mechanism_of_action=entry.get("mechanism_of_action", ""),
                    target_proteins=entry.get("target_proteins", []),
                    smiles=entry.get("smiles"),
                    molecular_weight=entry.get("molecular_weight", 0.0),
                    logp=entry.get("logp", 0.0),
                    known_off_targets=entry.get("known_off_targets", []),
                    black_box_warning=entry.get("black_box_warning", False),
                )
                self._drugs[drug.drugbank_id] = drug
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid drug database file: {e}")
    
    def get_drug(self, drugbank_id: str) -> Optional[FDADrug]:
        """Retrieve drug by DrugBank ID."""
        return self._drugs.get(drugbank_id)
    
    def search_by_target(self, target: str) -> list[FDADrug]:
        """Search drugs that target a specific protein."""
        target_upper = target.upper()
        return [
            drug for drug in self._drugs.values()
            if target_upper in [t.upper() for t in drug.target_proteins]
            or target_upper in [t.upper() for t in drug.known_off_targets]
        ]
    
    def search_by_indication(self, keyword: str) -> list[FDADrug]:
        """Search drugs by indication keyword."""
        keyword_lower = keyword.lower()
        return [
            drug for drug in self._drugs.values()
            if keyword_lower in drug.indication.lower()
        ]
    
    def get_approved_drugs(self) -> list[FDADrug]:
        """Get all FDA-approved drugs."""
        return [
            drug for drug in self._drugs.values()
            if drug.approval_status == DrugStatus.APPROVED
        ]
    
    def list_all(self) -> list[FDADrug]:
        """List all drugs in database."""
        return list(self._drugs.values())


class RepurposingValidator:
    """
    Validates FDA drugs for metabolic target repurposing.
    
    Evaluates target relevance, safety profile, druglikeness,
    and novelty for repurposing candidates.
    """
    
    # Target relevance keywords for metabolic rewiring
    METABOLIC_KEYWORDS = [
        "fatty acid", "lipid", "metabolism", "oxidation",
        "triglyceride", "lipogenesis", "beta-oxidation",
        "mitochondrial", "acetyl-coa", "malonyl-coa",
    ]
    
    def __init__(
        self,
        drug_db: Optional[FDADrugDatabase] = None,
        min_approval_status: DrugStatus = DrugStatus.INVESTIGATIONAL,
    ):
        self.drug_db = drug_db or FDADrugDatabase()
        self.min_approval_status = min_approval_status
        self._status_rank = {
            DrugStatus.APPROVED: 5,
            DrugStatus.WITHDRAWN: 0,
            DrugStatus.PHASE_III: 4,
            DrugStatus.PHASE_II: 3,
            DrugStatus.PHASE_I: 2,
            DrugStatus.INVESTIGATIONAL: 1,
            DrugStatus.EXPERIMENTAL: 0,
        }
    
    def _compute_target_relevance(
        self,
        drug: FDADrug,
        target_proteins: list[str],
    ) -> tuple[float, list[str]]:
        """
        Compute target relevance score.
        
        Returns:
            Tuple of (score, flags)
        """
        flags = []
        score = 0.0
        
        # Direct target match
        direct_matches = set(drug.target_proteins) & set(target_proteins)
        if direct_matches:
            score += 0.5
            flags.append(f"Direct target match: {direct_matches}")
        
        # Off-target match
        off_target_matches = set(drug.known_off_targets) & set(target_proteins)
        if off_target_matches:
            score += 0.3
            flags.append(f"Known off-target: {off_target_matches}")
        
        # Mechanism of action relevance
        moa_lower = drug.mechanism_of_action.lower()
        for keyword in self.METABOLIC_KEYWORDS:
            if keyword in moa_lower:
                score += 0.1
                break
        
        # Indication relevance (metabolic disease)
        metabolic_indications = ["diabetes", "obesity", "nash", "fatty liver", "hyperlipidemia"]
        if any(ind in drug.indication.lower() for ind in metabolic_indications):
            score += 0.1
            flags.append("Metabolic disease indication")
        
        return min(1.0, score), flags
    
    def _compute_safety_score(self, drug: FDADrug) -> tuple[float, list[str]]:
        """Compute safety score based on known issues."""
        flags = []
        score = 1.0
        
        if drug.black_box_warning:
            score -= 0.4
            flags.append("Black box warning")
        
        if drug.approval_status == DrugStatus.WITHDRAWN:
            score -= 0.5
            flags.append("Previously withdrawn")
        
        # Pregnancy category penalty
        if drug.pregnancy_category in {"X", "D"}:
            score -= 0.2
            flags.append(f"Pregnancy category {drug.pregnancy_category}")
        
        return max(0.0, score), flags
    
    def _compute_druglikeness(self, drug: FDADrug) -> float:
        """Compute druglikeness based on physicochemical properties."""
        score = 1.0
        
        # Lipinski's Rule of Five (simplified)
        if drug.molecular_weight > 500:
            score -= 0.2
        if drug.logp > 5:
            score -= 0.2
        
        # Oral bioavailability bonus for approved drugs
        if drug.approval_status == DrugStatus.APPROVED:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _compute_novelty(
        self,
        drug: FDADrug,
        target_proteins: list[str],
    ) -> float:
        """
        Compute novelty score for repurposing.
        
        Higher = more novel repurposing opportunity.
        """
        # If drug already targets these proteins, low novelty
        if set(drug.target_proteins) & set(target_proteins):
            return 0.2
        
        # Off-target repurposing = moderate novelty
        if set(drug.known_off_targets) & set(target_proteins):
            return 0.5
        
        # Completely new indication/target = high novelty
        return 0.8
    
    def validate_drug(
        self,
        drugbank_id: str,
        target_proteins: list[str],
    ) -> Optional[RepurposingValidation]:
        """
        Validate a drug for repurposing to metabolic targets.
        
        Args:
            drugbank_id: DrugBank identifier
            target_proteins: List of target proteins to evaluate
            
        Returns:
            RepurposingValidation or None if drug not found
        """
        drug = self.drug_db.get_drug(drugbank_id)
        if drug is None:
            return None
        
        # Check minimum approval status
        if self._status_rank.get(drug.approval_status, 0) < self._status_rank.get(self.min_approval_status, 0):
            return RepurposingValidation(
                drug=drug,
                outcome=ValidationOutcome.REJECTED,
                confidence=1.0,
                target_relevance_score=0.0,
                safety_score=0.0,
                druglikeness_score=0.0,
                novelty_score=0.0,
                flags=["Does not meet minimum approval status"],
                rationale="Drug does not meet minimum clinical development status requirement.",
            )
        
        # Compute individual scores
        target_rel, target_flags = self._compute_target_relevance(drug, target_proteins)
        safety, safety_flags = self._compute_safety_score(drug)
        druglikeness = self._compute_druglikeness(drug)
        novelty = self._compute_novelty(drug, target_proteins)
        
        all_flags = target_flags + safety_flags
        
        # Determine outcome
        overall = 0.35 * target_rel + 0.25 * safety + 0.20 * druglikeness + 0.20 * novelty
        
        if overall >= 0.7 and not drug.black_box_warning:
            outcome = ValidationOutcome.VALIDATED
            confidence = min(1.0, overall + 0.1)
        elif overall >= 0.5:
            outcome = ValidationOutcome.PROBABLE
            confidence = overall
        elif overall >= 0.3:
            outcome = ValidationOutcome.UNCERTAIN
            confidence = overall
        else:
            outcome = ValidationOutcome.UNLIKELY
            confidence = max(0.0, overall - 0.1)
        
        # Generate rationale
        rationale_parts = [
            f"{drug.generic_name} shows {'strong' if target_rel > 0.5 else 'moderate' if target_rel > 0.3 else 'weak'} "
            f"target relevance ({target_rel:.2f}) for {', '.join(target_proteins)}.",
            f"Safety profile is {'favorable' if safety > 0.7 else 'acceptable' if safety > 0.5 else 'concerning'} ({safety:.2f}).",
            f"Repurposing novelty is {'high' if novelty > 0.6 else 'moderate' if novelty > 0.4 else 'low'} ({novelty:.2f}).",
        ]
        rationale = " ".join(rationale_parts)
        
        return RepurposingValidation(
            drug=drug,
            outcome=outcome,
            confidence=confidence,
            target_relevance_score=target_rel,
            safety_score=safety,
            druglikeness_score=druglikeness,
            novelty_score=novelty,
            flags=all_flags,
            rationale=rationale,
        )
    
    def validate_candidates(
        self,
        drugbank_ids: list[str],
        target_proteins: list[str],
        min_score: float = 0.5,
    ) -> list[RepurposingValidation]:
        """
        Validate multiple drug candidates.
        
        Args:
            drugbank_ids: List of DrugBank IDs to validate
            target_proteins: Target proteins for repurposing
            min_score: Minimum overall score to include
            
        Returns:
            List of validations sorted by overall score
        """
        validations = []
        for db_id in drugbank_ids:
            result = self.validate_drug(db_id, target_proteins)
            if result and result.overall_score >= min_score:
                validations.append(result)
        
        return sorted(validations, key=lambda v: v.overall_score, reverse=True)
    
    def suggest_repurposing(
        self,
        target_proteins: list[str],
        top_k: int = 5,
    ) -> list[RepurposingValidation]:
        """
        Suggest repurposing candidates from database.
        
        Args:
            target_proteins: Target proteins to match
            top_k: Number of top candidates to return
            
        Returns:
            List of top repurposing suggestions
        """
        all_validations = []
        for drug in self.drug_db.list_all():
            result = self.validate_drug(drug.drugbank_id, target_proteins)
            if result:
                all_validations.append(result)
        
        return sorted(all_validations, key=lambda v: v.overall_score, reverse=True)[:top_k]


# Example usage
if __name__ == "__main__":
    validator = RepurposingValidator()
    
    # Validate for DGAT1/YARS2 dual targeting
    targets = ["DGAT1", "YARS2", "CPT1", "FASN"]
    
    print("=" * 70)
    print("FDA Drug Repurposing Validation for Metabolic Targets")
    print(f"Targets: {', '.join(targets)}")
    print("=" * 70)
    
    suggestions = validator.suggest_repurposing(targets, top_k=5)
    
    for i, validation in enumerate(suggestions, 1):
        print(f"\n{i}. {validation.drug.generic_name} ({validation.drug.drugbank_id})")
        print(f"   Status: {validation.drug.approval_status.value}")
        print(f"   Outcome: {validation.outcome.value} (confidence: {validation.confidence:.2f})")
        print(f"   Overall Score: {validation.overall_score:.3f}")
        print(f"   - Target Relevance: {validation.target_relevance_score:.3f}")
        print(f"   - Safety: {validation.safety_score:.3f}")
        print(f"   - Druglikeness: {validation.druglikeness_score:.3f}")
        print(f"   - Novelty: {validation.novelty_score:.3f}")
        if validation.flags:
            print(f"   Flags: {', '.join(validation.flags)}")
        print(f"   Rationale: {validation.rationale[:200]}...")
```

---

## File 4: `arp_v3/agents/virtual_screen/docking_coordinator.py`

```python
"""
Multi-Target Docking Coordinator.

Orchestrates parallel docking runs across multiple metabolic targets
with resource management and result aggregation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from multi_target_scorer import MultiTargetScorer, TargetBindingResult, TargetType
from repurposing_validator import RepurposingValidator, RepurposingValidation

logger = logging.getLogger(__name__)


class DockingEngine(Enum):
    """Supported docking engines."""
    AUTODOCK_VINA = "autodock_vina"
    GNINA = "gnina"
    DIFFDOCK = "diffdock"
    GLIDE = "glide"


class DockingStatus(Enum):
    """Docking job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DockingJob:
    """Individual docking job configuration."""
    job_id: str
    compound_id: str
    compound_smiles: str
    target: TargetType
    receptor_path: str
    engine: DockingEngine = DockingEngine.AUTODOCK_VINA
    n_poses: int = 9
    exhaustiveness: int = 8
    status: DockingStatus = DockingStatus.PENDING
    result: Optional[TargetBindingResult] = None
    error: Optional[str] = None


@dataclass
class MultiTargetDockingResult:
    """Aggregated result from multi-target docking."""
    compound_id: str
    jobs: list[DockingJob]
    multi_target_score: Optional[Any] = None  # MultiTargetScore from scorer
    repurposing_validation: Optional[RepurposingValidation] = None
    total_time_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Fraction of jobs that completed successfully."""
        if not self.jobs:
            return 0.0
        completed = sum(1 for j in self.jobs if j.status == DockingStatus.COMPLETED)
        return completed / len(self.jobs)
    
    @property
    def is_complete(self) -> bool:
        """Check if all jobs are done (success or failure)."""
        return all(
            j.status in (DockingStatus.COMPLETED, DockingStatus.FAILED, DockingStatus.CANCELLED)
            for j in self.jobs
        )


@dataclass
class ReceptorConfig:
    """Configuration for a docking receptor."""
    target: TargetType
    receptor_path: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    prepared: bool = False


class DockingCoordinator:
    """
    Coordinates multi-target docking operations.
    
    Manages parallel docking across DGAT1, YARS2, and other
    metabolic targets with resource pooling and result aggregation.
    
    Attributes:
        max_concurrent_jobs: Maximum parallel docking jobs
        default_engine: Default docking engine
        scorer: Multi-target scorer for result aggregation
        validator: Optional repurposing validator
    """
    
    # Default receptor configurations for metabolic targets
    DEFAULT_RECEPTORS: dict[TargetType, ReceptorConfig] = {
        TargetType.DGAT1: ReceptorConfig(
            target=TargetType.DGAT1,
            receptor_path="receptors/dgat1_prepared.pdbqt",
            center=(15.0, 20.0, 30.0),
            size=(20.0, 20.0, 20.0),
        ),
        TargetType.YARS2: ReceptorConfig(
            target=TargetType.YARS2,
            receptor_path="receptors/yars2_prepared.pdbqt",
            center=(10.0, 15.0, 25.0),
            size=(22.0, 22.0, 22.0),
        ),
        TargetType.ACC1: ReceptorConfig(
            target=TargetType.ACC1,
            receptor_path="receptors/acc1_prepared.pdbqt",
            center=(25.0, 30.0, 15.0),
            size=(18.0, 18.0, 18.0),
        ),
        TargetType.FASN: ReceptorConfig(
            target=TargetType.FASN,
            receptor_path="receptors/fasn_prepared.pdbqt",
            center=(20.0, 25.0, 20.0),
            size=(25.0, 25.0, 25.0),
        ),
        TargetType.CPT1: ReceptorConfig(
            target=TargetType.CPT1,
            receptor_path="receptors/cpt1_prepared.pdbqt",
            center=(12.0, 18.0, 22.0),
            size=(20.0, 20.0, 20.0),
        ),
    }
    
    def __init__(
        self,
        max_concurrent_jobs: int = 4,
        default_engine: DockingEngine = DockingEngine.AUTODOCK_VINA,
        receptor_configs: Optional[dict[TargetType, ReceptorConfig]] = None,
        scorer: Optional[MultiTargetScorer] = None,
        validator: Optional[RepurposingValidator] = None,
    ):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.default_engine = default_engine
        self.receptors = receptor_configs or self.DEFAULT_RECEPTORS.copy()
        self.scorer = scorer or MultiTargetScorer()
        self.validator = validator
        self._job_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._job_counter = 0
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        self._job_counter += 1
        return f"dock_{self._job_counter:06d}"
    
    def create_jobs(
        self,
        compound_id: str,
        compound_smiles: str,
        targets: Optional[list[TargetType]] = None,
        engine: Optional[DockingEngine] = None,
        n_poses: int = 9,
        exhaustiveness: int = 8,
    ) -> list[DockingJob]:
        """
        Create docking jobs for a compound across targets.
        
        Args:
            compound_id: Unique compound identifier
            compound_smiles: SMILES string
            targets: List of targets to dock (default: all configured)
            engine: Docking engine override
            n_poses: Number of poses to generate
            exhaustiveness: Docking exhaustiveness parameter
            
        Returns:
            List of DockingJob objects
        """
        targets = targets or list(self.receptors.keys())
        engine = engine or self.default_engine
        jobs = []
        
        for target in targets:
            if target not in self.receptors:
                logger.warning(f"No receptor configured for {target}, skipping")
                continue
            
            receptor = self.receptors[target]
            jobs.append(DockingJob(
                job_id=self._generate_job_id(),
                compound_id=compound_id,
                compound_smiles=compound_smiles,
                target=target,
                receptor_path=receptor.receptor_path,
                engine=engine,
                n_poses=n_poses,
                exhaustiveness=exhaustiveness,
            ))
        
        return jobs
    
    async def _run_docking_job(self, job: DockingJob) -> DockingJob:
        """
        Execute a single docking job (mock implementation).
        
        In production, this would call the actual docking engine
        via subprocess or API.
        """
        async with self._job_semaphore:
            job.status = DockingStatus.RUNNING
            logger.info(f"Starting {job.job_id}: {job.compound_id} -> {job.target.value}")
            
            try:
                # Mock docking - replace with actual engine call
                await asyncio.sleep(0.1)  # Simulate docking time
                
                # Generate mock results based on SMILES hash
                smile_hash = hash(job.compound_smiles) % 1000
                mock_score = -6.0 - (smile_hash / 1000.0) * 6.0  # -6 to -12
                mock_confidence = 0.6 + (smile_hash % 400) / 1000.0
                
                job.result = TargetBindingResult(
                    target=job.target,
                    docking_score=round(mock_score, 2),
                    confidence=round(mock_confidence, 2),
                    pose_count=job.n_poses,
                    best_rmsd=round(0.5 + (smile_hash % 300) / 1000.0, 2),
                )
                job.status = DockingStatus.COMPLETED
                
            except Exception as e:
                job.status = DockingStatus.FAILED
                job.error = str(e)
                logger.error(f"Job {job.job_id} failed: {e}")
            
            return job
    
    async def dock_compound(
        self,
        compound_id: str,
        compound_smiles: str,
        targets: Optional[list[TargetType]] = None,
        validate_repurposing: bool = False,
        drugbank_id: Optional[str] = None,
    ) -> MultiTargetDockingResult:
        """
        Run multi-target docking for a single compound.
        
        Args:
            compound_id: Unique compound identifier
            compound_smiles: SMILES string
            targets: Targets to dock against
            validate_repurposing: Whether to check repurposing validity
            drugbank_id: DrugBank ID if compound is FDA-approved
            
        Returns:
            MultiTargetDockingResult with aggregated scores
        """
        import time
        start_time = time.time()
        
        # Create jobs
        jobs = self.create_jobs(compound_id, compound_smiles, targets)
        
        if not jobs:
            return MultiTargetDockingResult(
                compound_id=compound_id,
                jobs=[],
            )
        
        # Run jobs in parallel
        tasks = [self._run_docking_job(job) for job in jobs]
        completed_jobs = await asyncio.gather(*tasks)
        
        # Aggregate results
        binding_results = [
            job.result for job in completed_jobs
            if job.result is not None
        ]
        
        # Compute multi-target score
        multi_score = None
        if binding_results:
            try:
                multi_score = self.scorer.score_compound(compound_id, binding_results)
            except ValueError as e:
                logger.warning(f"Could not score {compound_id}: {e}")
        
        # Optional repurposing validation
        repurposing_result = None
        if validate_repurposing and drugbank_id and self.validator:
            target_proteins = [t.value for t in (targets or list(self.receptors.keys()))]
            repurposing_result = self.validator.validate_drug(drugbank_id, target_proteins)
        
        elapsed = time.time() - start_time
        
        return MultiTargetDockingResult(
            compound_id=compound_id,
            jobs=list(completed_jobs),
            multi_target_score=multi_score,
            repurposing_validation=repurposing_result,
            total_time_seconds=elapsed,
        )
    
    async def dock_compounds_batch(
        self,
        compounds: list[tuple[str, str]],  # (compound_id, smiles)
        targets: Optional[list[TargetType]] = None,
        validate_repurposing: bool = False,
        drugbank_ids: Optional[dict[str, str]] = None,
    ) -> list[MultiTargetDockingResult]:
        """
        Run multi-target docking for multiple compounds.
        
        Args:
            compounds: List of (compound_id, smiles) tuples
            targets: Targets to dock against
            validate_repurposing: Whether to validate repurposing
            drugbank_ids: Dict mapping compound_id to DrugBank ID
            
        Returns:
            List of results sorted by multi-target score
        """
        drugbank_ids = drugbank_ids or {}
        
        tasks = [
            self.dock_compound(
                compound_id=comp_id,
                compound_smiles=smiles,
                targets=targets,
                validate_repurposing=validate_repurposing,
                drugbank_id=drugbank_ids.get(comp_id),
            )
            for comp_id, smiles in compounds
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Sort by multi-target score
        return sorted(
            results,
            key=lambda r: r.multi_target_score.combined_score if r.multi_target_score else 0.0,
            reverse=True,
        )
    
    def set_custom_docking_function(
        self,
        func: Callable[[DockingJob], asyncio.coroutine],
    ) -> None:
        """
        Override the default docking function with custom implementation.
        
        Args:
            func: Async function that takes DockingJob and returns updated DockingJob
        """
        self._run_docking_job = func  # type: ignore
    
    def get_receptor_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of configured receptors."""
        return {
            target.value: {
                "path": config.receptor_path,
                "center": config.center,
                "size": config.size,
                "prepared": config.prepared,
            }
            for target, config in self.receptors.items()
        }


# Example usage
if __name__ == "__main__":
    async def main():
        coordinator = DockingCoordinator(
            max_concurrent_jobs=2,
            validator=RepurposingValidator(),
        )
        
        # Single compound docking
        print("Single Compound Multi-Target Docking")
        print("=" * 50)
        result = await coordinator.dock_compound(
            compound_id="BBT-TEST-001",
            compound_smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            targets=[TargetType.DGAT1, TargetType.YARS2, TargetType.ACC1],
        )
        
        print(f"Compound: {result.compound_id}")
        print(f"Success Rate: {result.success_rate:.0%}")
        print(f"Time: {result.total_time_seconds:.2f}s")
        
        if result.multi_target_score:
            mt = result.multi_target_score
            print(f"Combined Score: {mt.combined_score:.4f}")
            print(f"Synergy Factor: {mt.synergy_factor:.4f}")
            print(f"Is Synergistic: {mt.is_synergistic}")
        
        # Batch docking
        print("\n" + "=" * 50)
        print("Batch Docking")
        print("=" * 50)
        
        compounds = [
            ("BBT-001", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
            ("BBT-002", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("BBT-003", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ]
        
        batch_results = await coordinator.dock_compounds_batch(
            compounds,
            targets=[TargetType.DGAT1, TargetType.YARS2],
        )
        
        for result in batch_results:
            if result.multi_target_score:
                print(f"{result.compound_id}: score={result.multi_target_score.combined_score:.4f}")
    
    asyncio.run(main())
```

---

## File 5: `arp_v3/agents/multiomics/pathway_analyzer.py`

```python
"""
Enhanced Pathway Analyzer for Multi-Omics Integration.

Analyzes metabolic pathways integrating transcriptomics, proteomics,
and metabolomics data with flux predictions for target prioritization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OmicsDataType(Enum):
    """Types of omics data."""
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    FLUXOMICS = "fluxomics"


class PathwayImpactLevel(Enum):
    """Impact level classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEUTRAL = "neutral"


@dataclass
class GeneExpression:
    """Gene expression data point."""
    gene_id: str
    gene_symbol: str
    log2_fold_change: float
    p_value: float
    adjusted_p_value: float
    is_significant: bool = False
    
    @classmethod
    def from_raw(
        cls,
        gene_id: str,
        gene_symbol: str,
        log2_fc: float,
        p_val: float,
        significance_threshold: float = 0.05,
    ) -> GeneExpression:
        """Create from raw values with automatic significance determination."""
        return cls(
            gene_id=gene_id,
            gene_symbol=gene_symbol,
            log2_fold_change=log2_fc,
            p_value=p_val,
            adjusted_p_value=p_val,  # Would be adjusted in real implementation
            is_significant=p_val < significance_threshold and abs(log2_fc) > 1.0,
        )


@dataclass
class MetaboliteLevel:
    """Metabolite abundance data point."""
    metabolite_id: str
    metabolite_name: str
    log2_fold_change: float
    p_value: float
    is_significant: bool = False
    compartment: str = "unknown"


@dataclass
class PathwayNode:
    """Node in pathway graph (gene/enzyme or metabolite)."""
    node_id: str
    node_type: str  # "enzyme", "metabolite", "reaction"
    label: str
    expression_data: Optional[GeneExpression] = None
    metabolite_data: Optional[MetaboliteLevel] = None
    flux_change: float = 0.0
    
    @property
    def is_dysregulated(self) -> bool:
        """Check if node shows significant dysregulation."""
        if self.expression_data:
            return self.expression_data.is_significant
        if self.metabolite_data:
            return self.metabolite_data.is_significant
        return abs(self.flux_change) > 0.5


@dataclass
class PathwayEdge:
    """Edge in pathway graph (reaction/interaction)."""
    source_id: str
    target_id: str
    edge_type: str  # "catalyzes", "consumes", "produces", "regulates"
    reaction_id: Optional[str] = None
    flux_change: float = 0.0


@dataclass
class PathwayAnalysis:
    """Complete pathway analysis result."""
    pathway_id: str
    pathway_name: str
    nodes: list[PathwayNode]
    edges: list[PathwayEdge]
    impact_level: PathwayImpactLevel
    impact_score: float
    dysregulated_nodes: list[str]
    key_enzymes: list[str]
    target_suggestions: list[TargetSuggestion]
    narrative: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway_id": self.pathway_id,
            "pathway_name": self.pathway_name,
            "impact_level": self.impact_level.value,
            "impact_score": round(self.impact_score, 4),
            "n_dysregulated": len(self.dysregulated_nodes),
            "key_enzymes": self.key_enzymes,
            "target_suggestions": [t.to_dict() for t in self.target_suggestions],
            "narrative": self.narrative,
        }


@dataclass
class TargetSuggestion:
    """Suggested target from pathway analysis."""
    enzyme_id: str
    enzyme_name: str
    pathway_id: str
    rationale: str
    confidence: float
    expected_effect: str  # "inhibit" or "activate"
    supporting_evidence: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "enzyme_id": self.enzyme_id,
            "enzyme_name": self.enzyme_name,
            "pathway_id": self.pathway_id,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 3),
            "expected_effect": self.expected_effect,
            "evidence_count": len(self.supporting_evidence),
        }


class PathwayAnalyzer:
    """
    Enhanced pathway analyzer for multi-omics integration.
    
    Integrates transcriptomic, proteomic, and metabolomic data
    to identify dysregulated pathways and suggest therapeutic targets.
    
    Attributes:
        significance_threshold: P-value threshold for significance
        fold_change_threshold: |log2FC| threshold for significance
    """
    
    # KEGG pathway mappings for metabolic rewiring
    LIPID_METABOLISM_PATHWAYS = {
        "map00061": "Fatty acid biosynthesis",
        "map00071": "Fatty acid degradation",
        "map00100": "Steroid biosynthesis",
        "map00120": "Primary bile acid biosynthesis",
        "map00010": "Glycolysis / Gluconeogenesis",
        "map00020": "Citrate cycle (TCA cycle)",
        "map00190": "Oxidative phosphorylation",
    }
    
    # Enzyme to gene mappings
    ENZYME_GENE_MAP = {
        "DGAT1": "DGAT1",
        "YARS2": "YARS2",
        "ACC1": "ACACA",
        "FASN": "FASN",
        "CPT1": "CPT1A",
        "CPT2": "CPT2",
        "ACLY": "ACLY",
        "SCD": "SCD",
        "ELOVL6": "ELOVL6",
        "HMGCR": "HMGCR",
    }
    
    def __init__(
        self,
        significance_threshold: float = 0.05,
        fold_change_threshold: float = 1.0,
    ):
        self.significance_threshold = significance_threshold
        self.fold_change_threshold = fold_change_threshold
        self._expression_data: dict[str, GeneExpression] = {}
        self._metabolite_data: dict[str, MetaboliteLevel] = {}
        self._flux_data: dict[str, float] = {}
    
    def load_transcriptomics(
        self,
        expression_data: list[GeneExpression],
    ) -> None:
        """Load transcriptomics data."""
        for expr in expression_data:
            self._expression_data[expr.gene_symbol.upper()] = expr
        logger.info(f"Loaded {len(expression_data)} gene expression values")
    
    def load_metabolomics(
        self,
        metabolite_data: list[MetaboliteLevel],
    ) -> None:
        """Load metabolomics data."""
        for met in metabolite_data:
            self._metabolite_data[met.metabolite_id.upper()] = met
        logger.info(f"Loaded {len(metabolite_data)} metabolite levels")
    
    def load_flux_predictions(
        self,
        flux_data: dict[str, float],
    ) -> None:
        """Load flux prediction data."""
        self._flux_data = {k.upper(): v for k, v in flux_data.items()}
        logger.info(f"Loaded {len(flux_data)} flux predictions")
    
    def _build_pathway_graph(
        self,
        pathway_id: str,
        enzymes: list[str],
        metabolites: list[str],
    ) -> tuple[list[PathwayNode], list[PathwayEdge]]:
        """Build pathway graph from enzyme and metabolite lists."""
        nodes = []
        edges = []
        
        # Add enzyme nodes
        for enzyme in enzymes:
            gene = self.ENZYME_GENE_MAP.get(enzyme, enzyme)
            expr = self._expression_data.get(gene.upper())
            flux = self._flux_data.get(enzyme.upper(), 0.0)
            
            nodes.append(PathwayNode(
                node_id=f"{pathway_id}_{enzyme}",
                node_type="enzyme",
                label=enzyme,
                expression_data=expr,
                flux_change=flux,
            ))
        
        # Add metabolite nodes
        for met in metabolites:
            met_data = self._metabolite_data.get(met.upper())
            nodes.append(PathwayNode(
                node_id=f"{pathway_id}_{met}",
                node_type="metabolite",
                label=met,
                metabolite_data=met_data,
            ))
        
        # Add edges (simplified: enzyme -> metabolite connections)
        for i, enzyme in enumerate(enzymes):
            enzyme_node_id = f"{pathway_id}_{enzyme}"
            if i + 1 < len(metabolites):
                # Enzyme produces next metabolite
                edges.append(PathwayEdge(
                    source_id=enzyme_node_id,
                    target_id=f"{pathway_id}_{metabolites[i+1]}",
                    edge_type="produces",
                    reaction_id=f"{pathway_id}_R{i+1}",
                    flux_change=self._flux_data.get(enzyme.upper(), 0.0),
                ))
            if i > 0:
                # Enzyme consumes previous metabolite
                edges.append(PathwayEdge(
                    source_id=f"{pathway_id}_{metabolites[i-1]}",
                    target_id=enzyme_node_id,
                    edge_type="consumes",
                ))
        
        return nodes, edges
    
    def _compute_impact_score(
        self,
        nodes: list[PathwayNode],
        edges: list[PathwayEdge],
    ) -> tuple[float, PathwayImpactLevel]:
        """Compute pathway impact score."""
        if not nodes:
            return 0.0, PathwayImpactLevel.NEUTRAL
        
        # Count dysregulated nodes
        n_dysregulated = sum(1 for n in nodes if n.is_dysregulated)
        dysregulation_ratio = n_dysregulated / len(nodes)
        
        # Average fold change magnitude
        fold_changes = []
        for node in nodes:
            if node.expression_data:
                fold_changes.append(abs(node.expression_data.log2_fold_change))
            elif node.metabolite_data:
                fold_changes.append(abs(node.metabolite_data.log2_fold_change))
            else:
                fold_changes.append(abs(node.flux_change))
        
        avg_fc = np.mean(fold_changes) if fold_changes else 0.0
        
        # Combined score
        impact = 0.6 * dysregulation_ratio + 0.4 * min(1.0, avg_fc / 3.0)
        
        # Classify impact level
        if impact >= 0.6:
            level = PathwayImpactLevel.HIGH
        elif impact >= 0.35:
            level = PathwayImpactLevel.MEDIUM
        elif impact >= 0.15:
            level = PathwayImpactLevel.LOW
        else:
            level = PathwayImpactLevel.NEUTRAL
        
        return impact, level
    
    def _suggest_targets(
        self,
        pathway_id: str,
        pathway_name: str,
        nodes: list[PathwayNode],
        impact_score: float,
    ) -> list[TargetSuggestion]:
        """Suggest therapeutic targets from pathway analysis."""
        suggestions = []
        
        for node in nodes:
            if node.node_type != "enzyme":
                continue
            
            evidence = []
            confidence = 0.3  # Base confidence
            
            # Check expression dysregulation
            if node.expression_data and node.expression_data.is_significant:
                fc = node.expression_data.log2_fold_change
                evidence.append(f"Gene {node.label} {'upregulated' if fc > 0 else 'downregulated'} (log2FC={fc:.2f})")
                confidence += 0.3
            
            # Check flux change
            if abs(node.flux_change) > 0.5:
                evidence.append(f"Flux change: {node.flux_change:.2f}x")
                confidence += 0.2
            
            # Check if enzyme is in our target list
            if node.label in self.ENZYME_GENE_MAP:
                confidence += 0.1
                evidence.append("Known metabolic target")
            
            # Determine expected effect
            if node.expression_data:
                expected = "inhibit" if node.expression_data.log2_fold_change > 0 else "activate"
            elif node.flux_change > 1.0:
                expected = "inhibit"
            else:
                expected = "activate"
            
            # Only suggest if reasonable confidence
            if confidence >= 0.4:
                suggestions.append(TargetSuggestion(
                    enzyme_id=node.node_id,
                    enzyme_name=node.label,
                    pathway_id=pathway_id,
                    rationale=f"Target {node.label} in {pathway_name} based on multi-omics evidence",
                    confidence=min(1.0, confidence),
                    expected_effect=expected,
                    supporting_evidence=evidence,
                ))
        
        # Sort by confidence
        return sorted(suggestions, key=lambda s: s.confidence, reverse=True)
    
    def _generate_narrative(
        self,
        pathway_name: str,
        impact_level: PathwayImpactLevel,
        dysregulated: list[str],
        targets: list[TargetSuggestion],
    ) -> str:
        """Generate human-readable narrative for pathway analysis."""
        if impact_level == PathwayImpactLevel.NEUTRAL:
            return f"{pathway_name} shows no significant dysregulation."
        
        parts = [
            f"{pathway_name} shows {impact_level.value} impact",
            f"with {len(dysregulated)} dysregulated nodes.",
        ]
        
        if dysregulated:
            parts.append(f"Key dysregulated components: {', '.join(dysregulated[:5])}.")
        
        if targets:
            top_target = targets[0]
            parts.append(
                f"Primary target suggestion: {top_target.enzyme_name} "
                f"({top_target.expected_effect}) with {top_target.confidence:.0%} confidence."
            )
        
        return " ".join(parts)
    
    def analyze_pathway(
        self,
        pathway_id: str,
        enzymes: list[str],
        metabolites: list[str],
    ) -> PathwayAnalysis:
        """
        Analyze a single metabolic pathway.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "map00061")
            enzymes: List of enzyme names in pathway
            metabolites: List of metabolite IDs in pathway
            
        Returns:
            PathwayAnalysis with impact assessment and target suggestions
        """
        pathway_name = self.LIPID_METABOLISM_PATHWAYS.get(
            pathway_id, f"Pathway {pathway_id}"
        )
        
        # Build graph
        nodes, edges = self._build_pathway_graph(pathway_id, enzymes, metabolites)
        
        # Compute impact
        impact_score, impact_level = self._compute_impact_score(nodes, edges)
        
        # Get dysregulated nodes
        dysregulated = [n.label for n in nodes if n.is_dysregulated]
        
        # Get key enzymes (dysregulated enzymes)
        key_enzymes = [
            n.label for n in nodes
            if n.node_type == "enzyme" and n.is_dysregulated
        ]
        
        # Suggest targets
        targets = self._suggest_targets(pathway_id, pathway_name, nodes, impact_score)
        
        # Generate narrative
        narrative = self._generate_narrative(
            pathway_name, impact_level, dysregulated, targets
        )
        
        return PathwayAnalysis(
            pathway_id=pathway_id,
            pathway_name=pathway_name,
            nodes=nodes,
            edges=edges,
            impact_level=impact_level,
            impact_score=impact_score,
            dysregulated_nodes=dysregulated,
            key_enzymes=key_enzymes,
            target_suggestions=targets,
            narrative=narrative,
        )
    
    def analyze_multiple_pathways(
        self,
        pathway_configs: list[tuple[str, list[str], list[str]]],
    ) -> list[PathwayAnalysis]:
        """
        Analyze multiple pathways.
        
        Args:
            pathway_configs: List of (pathway_id, enzymes, metabolites) tuples
            
        Returns:
            List of PathwayAnalysis sorted by impact score
        """
        results = []
        for pathway_id, enzymes, metabolites in pathway_configs:
            analysis = self.analyze_pathway(pathway_id, enzymes, metabolites)
            results.append(analysis)
        
        return sorted(results, key=lambda a: a.impact_score, reverse=True)
    
    def get_cross_pathway_targets(
        self,
        analyses: list[PathwayAnalysis],
        min_confidence: float = 0.5,
    ) -> list[TargetSuggestion]:
        """
        Identify targets appearing across multiple pathways.
        
        Args:
            analyses: List of pathway analyses
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of targets with pathway counts
        """
        enzyme_pathways: dict[str, list[str]] = {}
        enzyme_suggestions: dict[str, TargetSuggestion] = {}
        
        for analysis in analyses:
            for suggestion in analysis.target_suggestions:
                if suggestion.confidence >= min_confidence:
                    enzyme = suggestion.enzyme_name
                    if enzyme not in enzyme_pathways:
                        enzyme_pathways[enzyme] = []
                        enzyme_suggestions[enzyme] = suggestion
                    enzyme_pathways[enzyme].append(analysis.pathway_name)
        
        # Boost confidence for cross-pathway targets
        cross_pathway = []
        for enzyme, pathways in enzyme_pathways.items():
            suggestion = enzyme_suggestions[enzyme]
            if len(pathways) > 1:
                # Boost confidence for multi-pathway targets
                boosted = TargetSuggestion(
                    enzyme_id=suggestion.enzyme_id,
                    enzyme_name=suggestion.enzyme_name,
                    pathway_id=",".join(pathways),
                    rationale=f"{suggestion.rationale} Appears in {len(pathways)} pathways.",
                    confidence=min(1.0, suggestion.confidence + 0.1 * (len(pathways) - 1)),
                    expected_effect=suggestion.expected_effect,
                    supporting_evidence=suggestion.supporting_evidence + [
                        f"Present in {len(pathways)} dysregulated pathways"
                    ],
                )
                cross_pathway.append(boosted)
        
        return sorted(cross_pathway, key=lambda t: t.confidence, reverse=True)


# Example usage
if __name__ == "__main__":
    analyzer = PathwayAnalyzer()
    
    # Load mock transcriptomics data
    expression_data = [
        GeneExpression.from_raw("ENSG000001", "DGAT1", 2.3, 0.001),
        GeneExpression.from_raw("ENSG000002", "YARS2", -1.8, 0.01),
        GeneExpression.from_raw("ENSG000003", "ACACA", 1.5, 0.02),
        GeneExpression.from_raw("ENSG000004", "FASN", 2.8, 0.0005),
        GeneExpression.from_raw("ENSG000005", "CPT1A", -2.1, 0.003),
        GeneExpression.from_raw("ENSG000006", "SCD", 1.2, 0.08),  # Not significant
        GeneExpression.from_raw("ENSG000007", "ELOVL6", 0.5, 0.3),  # Not significant
    ]
    analyzer.load_transcriptomics(expression_data)
    
    # Load mock metabolomics data
    metabolite_data = [
        MetaboliteLevel("M001", "Palmitate", 1.9, 0.005, is_significant=True),
        MetaboliteLevel("M002", "Triglyceride", 2.5, 0.001, is_significant=True),
        MetaboliteLevel("M003", "Acetyl-CoA", -0.8, 0.1),
        MetaboliteLevel("M004", "Malonyl-CoA", 1.4, 0.02, is_significant=True),
    ]
    analyzer.load_metabolomics(metabolite_data)
    
    # Load flux predictions
    flux_data = {
        "DGAT1": 1.8,
        "FASN": 2.2,
        "CPT1": 0.3,
        "ACC1": 1.5,
    }
    analyzer.load_flux_predictions(flux_data)
    
    # Analyze pathways
    pathway_configs = [
        ("map00061", ["ACC1", "FASN", "ELOVL6", "SCD"], ["Acetyl-CoA", "Malonyl-CoA", "Palmitate"]),
        ("map00071", ["CPT1", "CPT2", "ACLY"], ["Palmitate", "Palmitoyl-CoA", "Acetyl-CoA"]),
        ("custom_lipid", ["DGAT1", "ACLY"], ["DAG", "Triglyceride"]),
    ]
    
    analyses = analyzer.analyze_multiple_pathways(pathway_configs)
    
    print("=" * 70)
    print("Multi-Omics Pathway Analysis Results")
    print("=" * 70)
    
    for analysis in analyses:
        print(f"\n{analysis.pathway_name} ({analysis.pathway_id})")
        print(f"  Impact: {analysis.impact_level.value} (score: {analysis.impact_score:.3f})")
        print(f"  Dysregulated: {', '.join(analysis.dysregulated_nodes) or 'None'}")
        print(f"  Key Enzymes: {', '.join(analysis.key_enzymes) or 'None'}")
        print(f"  Narrative: {analysis.narrative}")
        
        if analysis.target_suggestions:
            print("  Target Suggestions:")
            for target in analysis.target_suggestions[:3]:
                print(f"    - {target.enzyme_name}: {target.expected_effect} "
                      f"(conf: {target.confidence:.2f})")
    
    # Cross-pathway analysis
    print("\n" + "=" * 70)
    print("Cross-Pathway Target Analysis")
    print("=" * 70)
    
    cross_targets = analyzer.get_cross_pathway_targets(analyses)
    for target in cross_targets:
        print(f"\n{target.enzyme_name}")
        print(f"  Pathways: {target.pathway_id}")
        print(f"  Confidence: {target.confidence:.2f}")
        print(f"  Evidence: {len(target.supporting_evidence)} pieces")
```

---

## Summary of Improvements

| Module | File | Key Improvement |
|--------|------|-----------------|
| **Multi-Target Scorer** | `multi_target_scorer.py` | Dual DGAT1/YARS2 scoring with Bliss synergy model |
| **Metabolic Flux GNN** | `metabolic_flux_gnn.py` | GNN-based flux prediction on reaction-metabolite graphs |
| **Repurposing Validator** | `repurposing_validator.py` | FDA drug validation with safety/novelty scoring |
| **Docking Coordinator** | `docking_coordinator.py` | Async multi-target docking with resource pooling |
| **Pathway Analyzer** | `pathway_analyzer.py` | Multi-omics integration with cross-pathway target ID |

### Integration Flow
```
PathwayAnalyzer → identifies dysregulated enzymes
        ↓
DockingCoordinator → docks compounds against identified targets
        ↓
MultiTargetScorer → combines docking scores with synergy detection
        ↓
RepurposingValidator → validates FDA-approved hits for repurposing
        ↓
MetabolicFluxPredictor → predicts flux changes upon target inhibition
```