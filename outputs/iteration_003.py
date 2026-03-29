# BrownBioTech Iteration 3→4: Consensus Screening Module

## File 1: `arp_v3/agents/virtual_screen/consensus_scorer.py`

```python
"""
Consensus Virtual Screening Scorer

Combines multiple scoring methods (docking, ML, pharmacophore) into a 
unified consensus score with configurable weighting and outlier detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Available scoring methods for consensus."""
    DOCKING = "docking"
    ML_AFFINITY = "ml_affinity"
    PHARMACOPHORE = "pharmacophore"
    FILTER = "filter"
    CUSTOM = "custom"


class ScoreTransformer(Enum):
    """Methods to normalize scores to comparable scales."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    RANK = "rank"
    SIGMOID = "sigmoid"


@dataclass
class ScoringResult:
    """Individual scoring result from a single method."""
    method: ScoringMethod
    raw_score: float
    normalized_score: float = 0.0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        return self.normalized_score * self.weight


@dataclass
class ConsensusResult:
    """Final consensus result for a single compound."""
    compound_id: str
    consensus_score: float
    individual_scores: dict[ScoringMethod, ScoringResult]
    rank: int = 0
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "consensus_score": round(self.consensus_score, 4),
            "rank": self.rank,
            "confidence": round(self.confidence, 4),
            "flags": self.flags,
            "individual_scores": {
                m.value: {
                    "raw": s.raw_score,
                    "normalized": round(s.normalized_score, 4),
                    "weighted": round(s.weighted_score, 4)
                }
                for m, s in self.individual_scores.items()
            }
        }


class ScorerProtocol(Protocol):
    """Protocol for any scorer that can produce standardized results."""
    
    def score(self, compound_id: str, mol_data: Any) -> float:
        """Return raw score for a compound. Higher = better."""
        ...
    
    @property
    def method(self) -> ScoringMethod:
        """Return the scoring method type."""
        ...


class ConsensusScorer:
    """
    Core consensus scoring engine.
    
    Combines multiple scoring methods using configurable normalization
    and weighting strategies to produce robust compound rankings.
    
    Features:
    - Multiple normalization strategies (min-max, z-score, rank, sigmoid)
    - Outlier detection and flagging
    - Confidence estimation based on scorer agreement
    - Configurable weighting per method
    """
    
    def __init__(
        self,
        scorers: list[ScorerProtocol] | None = None,
        weights: dict[ScoringMethod, float] | None = None,
        transformer: ScoreTransformer = ScoreTransformer.Z_SCORE,
        outlier_threshold: float = 2.0,
        min_scorers: int = 2,
        confidence_method: str = "std_dev"
    ):
        """
        Initialize consensus scorer.
        
        Args:
            scorers: List of scorer instances implementing ScorerProtocol
            weights: Custom weights per method (default: equal weighting)
            transformer: Normalization method for raw scores
            outlier_threshold: Z-score threshold for outlier detection
            min_scorers: Minimum number of scorers required for consensus
            confidence_method: Method for confidence calculation
                - "std_dev": Based on standard deviation of normalized scores
                - "iqr": Based on interquartile range
                - "agreement": Based on rank agreement across methods
        """
        self.scorers: dict[ScoringMethod, ScorerProtocol] = {}
        self.weights = weights or {}
        self.transformer = transformer
        self.outlier_threshold = outlier_threshold
        self.min_scorers = min_scorers
        self.confidence_method = confidence_method
        
        if scorers:
            for scorer in scorers:
                self.add_scorer(scorer)
    
    def add_scorer(self, scorer: ScorerProtocol, weight: float | None = None) -> None:
        """Add a scorer to the consensus ensemble."""
        method = scorer.method
        self.scorers[method] = scorer
        if weight is not None:
            self.weights[method] = weight
        elif method not in self.weights:
            self.weights[method] = 1.0
        
        logger.debug(f"Added {method.value} scorer with weight {self.weights[method]}")
    
    def remove_scorer(self, method: ScoringMethod) -> None:
        """Remove a scorer from the ensemble."""
        if method in self.scorers:
            del self.scorers[method]
            self.weights.pop(method, None)
            logger.debug(f"Removed {method.value} scorer")
    
    def _normalize_scores(
        self,
        raw_scores: dict[ScoringMethod, float]
    ) -> dict[ScoringMethod, float]:
        """Normalize raw scores using the configured transformer."""
        if not raw_scores:
            return {}
        
        values = np.array(list(raw_scores.values()))
        methods = list(raw_scores.keys())
        
        if len(values) < 2:
            return {m: 1.0 for m in methods}
        
        normalized = {}
        
        if self.transformer == ScoreTransformer.MIN_MAX:
            min_val, max_val = values.min(), values.max()
            if max_val - min_val < 1e-10:
                normalized = {m: 0.5 for m in methods}
            else:
                normalized = {
                    m: (v - min_val) / (max_val - min_val)
                    for m, v in raw_scores.items()
                }
        
        elif self.transformer == ScoreTransformer.Z_SCORE:
            mean_val, std_val = values.mean(), values.std()
            if std_val < 1e-10:
                normalized = {m: 0.5 for m in methods}
            else:
                normalized = {
                    m: 0.5 + (v - mean_val) / (2 * std_val)
                    for m, v in raw_scores.items()
                }
                # Clip to [0, 1]
                normalized = {m: max(0.0, min(1.0, n)) for m, n in normalized.items()}
        
        elif self.transformer == ScoreTransformer.RANK:
            ranks = stats.rankdata(values)
            max_rank = len(values)
            normalized = {
                m: r / max_rank
                for m, r in zip(methods, ranks)
            }
        
        elif self.transformer == ScoreTransformer.SIGMOID:
            mean_val, std_val = values.mean(), values.std()
            if std_val < 1e-10:
                normalized = {m: 0.5 for m in methods}
            else:
                normalized = {
                    m: 1.0 / (1.0 + np.exp(-(v - mean_val) / std_val))
                    for m, v in raw_scores.items()
                }
        
        return normalized
    
    def _calculate_confidence(
        self,
        normalized_scores: dict[ScoringMethod, float]
    ) -> float:
        """Calculate confidence score based on scorer agreement."""
        if len(normalized_scores) < 2:
            return 0.5
        
        values = np.array(list(normalized_scores.values()))
        
        if self.confidence_method == "std_dev":
            # Lower std dev = higher confidence
            std_val = values.std()
            confidence = 1.0 - min(std_val / 0.5, 1.0)
        
        elif self.confidence_method == "iqr":
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            confidence = 1.0 - min(iqr / 0.5, 1.0)
        
        elif self.confidence_method == "agreement":
            # Based on how many scores are above/below median
            median = np.median(values)
            above = sum(1 for v in values if v >= median)
            total = len(values)
            # Perfect agreement = all above or all below median
            imbalance = abs(above - total / 2) / (total / 2)
            confidence = imbalance
        
        else:
            confidence = 0.5
        
        return max(0.0, min(1.0, confidence))
    
    def _detect_outliers(
        self,
        normalized_scores: dict[ScoringMethod, float]
    ) -> list[str]:
        """Detect outlier scores that disagree significantly."""
        if len(normalized_scores) < 3:
            return []
        
        values = np.array(list(normalized_scores.values()))
        methods = list(normalized_scores.keys())
        
        mean_val, std_val = values.mean(), values.std()
        if std_val < 1e-10:
            return []
        
        flags = []
        for method, value in zip(methods, values):
            z_score = abs(value - mean_val) / std_val
            if z_score > self.outlier_threshold:
                flags.append(f"outlier_{method.value}")
        
        return flags
    
    def score_compound(
        self,
        compound_id: str,
        mol_data: Any
    ) -> ConsensusResult:
        """Calculate consensus score for a single compound."""
        if len(self.scorers) < self.min_scorers:
            raise ValueError(
                f"Need at least {self.min_scorers} scorers, "
                f"have {len(self.scorers)}"
            )
        
        raw_scores: dict[ScoringMethod, float] = {}
        individual_results: dict[ScoringMethod, ScoringResult] = {}
        
        for method, scorer in self.scorers.items():
            try:
                raw = scorer.score(compound_id, mol_data)
                raw_scores[method] = raw
                individual_results[method] = ScoringResult(
                    method=method,
                    raw_score=raw,
                    weight=self.weights.get(method, 1.0)
                )
            except Exception as e:
                logger.warning(
                    f"Scorer {method.value} failed for {compound_id}: {e}"
                )
                individual_results[method] = ScoringResult(
                    method=method,
                    raw_score=float('nan'),
                    weight=0.0,
                    metadata={"error": str(e)}
                )
        
        # Normalize valid scores
        valid_raw = {m: s for m, s in raw_scores.items() if not np.isnan(s)}
        normalized = self._normalize_scores(valid_raw)
        
        # Update individual results with normalized scores
        for method, norm_score in normalized.items():
            individual_results[method].normalized_score = norm_score
        
        # Calculate weighted consensus
        total_weight = sum(
            self.weights.get(m, 1.0) for m in normalized.keys()
        )
        
        if total_weight < 1e-10:
            consensus = 0.0
        else:
            consensus = sum(
                individual_results[m].weighted_score
                for m in normalized.keys()
            ) / total_weight
        
        # Calculate confidence and detect outliers
        confidence = self._calculate_confidence(normalized)
        flags = self._detect_outliers(normalized)
        
        if len(valid_raw) < len(self.scorers):
            flags.append("incomplete_scoring")
        
        return ConsensusResult(
            compound_id=compound_id,
            consensus_score=consensus,
            individual_scores=individual_results,
            confidence=confidence,
            flags=flags
        )
    
    def score_compounds(
        self,
        compounds: list[tuple[str, Any]],
        parallel: bool = False,
        n_workers: int | None = None
    ) -> list[ConsensusResult]:
        """
        Score multiple compounds and assign ranks.
        
        Args:
            compounds: List of (compound_id, mol_data) tuples
            parallel: Whether to use parallel processing
            n_workers: Number of workers for parallel processing
            
        Returns:
            List of ConsensusResult sorted by consensus_score (descending)
        """
        if parallel:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            n_workers = n_workers or min(len(self.scorers), 4)
            results = []
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.score_compound, cid, mol): cid
                    for cid, mol in compounds
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        cid = futures[future]
                        logger.error(f"Failed to score {cid}: {e}")
        else:
            results = [
                self.score_compound(cid, mol)
                for cid, mol in compounds
            ]
        
        # Sort by consensus score (descending)
        results.sort(key=lambda r: r.consensus_score, reverse=True)
        
        # Assign ranks (handle ties)
        for i, result in enumerate(results):
            if i > 0 and abs(result.consensus_score - results[i-1].consensus_score) < 1e-6:
                result.rank = results[i-1].rank
            else:
                result.rank = i + 1
        
        return results
    
    def get_summary_stats(self, results: list[ConsensusResult]) -> dict[str, Any]:
        """Get summary statistics for a set of consensus results."""
        if not results:
            return {}
        
        scores = [r.consensus_score for r in results]
        confidences = [r.confidence for r in results]
        
        return {
            "n_compounds": len(results),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "confidence_mean": float(np.mean(confidences)),
            "n_flagged": sum(1 for r in results if r.flags),
            "flag_distribution": self._count_flags(results)
        }
    
    def _count_flags(self, results: list[ConsensusResult]) -> dict[str, int]:
        """Count occurrences of each flag type."""
        flag_counts: dict[str, int] = {}
        for result in results:
            for flag in result.flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        return flag_counts


# Example usage and testing
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockMolData:
        smiles: str
        features: list[float]
    
    class MockDockingScorer:
        @property
        def method(self) -> ScoringMethod:
            return ScoringMethod.DOCKING
        
        def score(self, compound_id: str, mol_data: Any) -> float:
            # Simulate docking score (negative = better binding)
            base = -8.0
            noise = np.random.normal(0, 1.5)
            return -(base + noise)  # Convert to positive (higher = better)
    
    class MockMLScorer:
        @property
        def method(self) -> ScoringMethod:
            return ScoringMethod.ML_AFFINITY
        
        def score(self, compound_id: str, mol_data: Any) -> float:
            # Simulate ML predicted affinity
            return np.random.uniform(0.2, 0.9)
    
    class MockPharmacophoreScorer:
        @property
        def method(self) -> ScoringMethod:
            return ScoringMethod.PHARMACOPHORE
        
        def score(self, compound_id: str, mol_data: Any) -> float:
            # Simulate pharmacophore fit score
            return np.random.uniform(0.1, 1.0)
    
    # Setup consensus scorer
    scorer = ConsensusScorer(
        transformer=ScoreTransformer.Z_SCORE,
        weights={
            ScoringMethod.DOCKING: 1.5,
            ScoringMethod.ML_AFFINITY: 1.0,
            ScoringMethod.PHARMACOPHORE: 0.8
        }
    )
    
    scorer.add_scorer(MockDockingScorer())
    scorer.add_scorer(MockMLScorer())
    scorer.add_scorer(MockPharmacophoreScorer())
    
    # Generate mock compounds
    np.random.seed(42)
    compounds = [
        (f"CMP_{i:04d}", MockMolData(
            smiles=f"C{i}CC(N)C(=O)O",
            features=np.random.rand(128).tolist()
        ))
        for i in range(20)
    ]
    
    # Run consensus scoring
    results = scorer.score_compounds(compounds)
    
    # Display top 5
    print("=" * 70)
    print("TOP 5 COMPOUNDS BY CONSENSUS SCORE")
    print("=" * 70)
    for r in results[:5]:
        print(f"\nRank {r.rank}: {r.compound_id}")
        print(f"  Consensus: {r.consensus_score:.4f} | Confidence: {r.confidence:.4f}")
        print(f"  Flags: {r.flags if r.flags else 'None'}")
        for method, score in r.individual_scores.items():
            print(f"    {method.value:15s}: raw={score.raw_score:.4f}, "
                  f"norm={score.normalized_score:.4f}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    stats_summary = scorer.get_summary_stats(results)
    for k, v in stats_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
```

---

## File 2: `arp_v3/agents/virtual_screen/docking_engine.py`

```python
"""
Standardized Docking Engine

Provides a unified interface for molecular docking with standardized
output format compatible with the consensus scoring system.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class DockingSoftware(Enum):
    """Supported docking software backends."""
    AUTO_DOCK_VINA = "vina"
    GLIDE = "glide"
    GOLD = "gold"
    PLANTS = "plants"
    MOCK = "mock"  # For testing


class DockingResultStatus(Enum):
    """Status of a docking result."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class DockingPose:
    """Single docking pose result."""
    pose_id: int
    score: float  # Binding affinity (kcal/mol, negative = better)
    rmsd_lb: float = 0.0  # RMSD lower bound
    rmsd_ub: float = 0.0  # RMSD upper bound
    coordinates: np.ndarray | None = None
    interactions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StandardizedDockingResult:
    """
    Standardized docking output format.
    
    All docking engines must produce results in this format for
    compatibility with the consensus scorer.
    """
    compound_id: str
    status: DockingResultStatus
    best_score: float = 0.0
    mean_score: float = 0.0
    best_pose: Optional[DockingPose] = None
    all_poses: list[DockingPose] = field(default_factory=list)
    n_poses: int = 0
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def normalized_score(self) -> float:
        """Convert to positive score (higher = better) for consensus."""
        return -self.best_score
    
    def to_consensus_format(self) -> float:
        """Return score in format expected by consensus scorer."""
        return self.normalized_score
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "compound_id": self.compound_id,
            "status": self.status.value,
            "best_score": self.best_score,
            "mean_score": round(self.mean_score, 4),
            "normalized_score": round(self.normalized_score, 4),
            "n_poses": self.n_poses,
            "execution_time": round(self.execution_time, 3),
            "metadata": self.metadata
        }


@dataclass
class DockingConfig:
    """Configuration for docking run."""
    software: DockingSoftware = DockingSoftware.AUTO_DOCK_VINA
    receptor_path: str | Path = ""
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    n_poses: int = 9
    exhaustiveness: int = 8
    max_runtime: float = 300.0  # seconds
    seed: int | None = None
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.software != DockingSoftware.MOCK and not self.receptor_path:
            issues.append("Receptor path required for non-mock docking")
        
        if any(s <= 0 for s in [self.size_x, self.size_y, self.size_z]):
            issues.append("Box sizes must be positive")
        
        if self.n_poses < 1:
            issues.append("Must request at least 1 pose")
        
        return issues


class DockingBackend(Protocol):
    """Protocol for docking software backends."""
    
    def dock_single(
        self,
        ligand_data: Any,
        config: DockingConfig
    ) -> StandardizedDockingResult:
        """Dock a single ligand."""
        ...
    
    def dock_batch(
        self,
        ligands: list[tuple[str, Any]],
        config: DockingConfig
    ) -> list[StandardizedDockingResult]:
        """Dock multiple ligands."""
        ...


class MockDockingBackend:
    """Mock docking backend for testing and development."""
    
    def __init__(self, base_score: float = -8.0, score_std: float = 1.5):
        self.base_score = base_score
        self.score_std = score_std
    
    def dock_single(
        self,
        ligand_data: Any,
        config: DockingConfig
    ) -> StandardizedDockingResult:
        """Generate mock docking result."""
        rng = np.random.default_rng(config.seed)
        
        n_poses = config.n_poses
        poses = []
        
        for i in range(n_poses):
            # First pose is best, others get progressively worse
            score = self.base_score + rng.normal(0, self.score_std) + i * 0.5
            poses.append(DockingPose(
                pose_id=i + 1,
                score=score,
                rmsd_lb=rng.uniform(0, 2),
                rmsd_ub=rng.uniform(2, 5),
                coordinates=rng.uniform(-10, 10, size=(20, 3)),
                interactions=self._mock_interactions(rng)
            ))
        
        # Sort by score (most negative = best)
        poses.sort(key=lambda p: p.score)
        
        best = poses[0]
        mean = np.mean([p.score for p in poses])
        
        return StandardizedDockingResult(
            compound_id=getattr(ligand_data, 'compound_id', 'unknown'),
            status=DockingResultStatus.SUCCESS,
            best_score=best.score,
            mean_score=mean,
            best_pose=best,
            all_poses=poses,
            n_poses=n_poses,
            execution_time=rng.uniform(0.5, 2.0),
            metadata={"backend": "mock"}
        )
    
    def dock_batch(
        self,
        ligands: list[tuple[str, Any]],
        config: DockingConfig
    ) -> list[StandardizedDockingResult]:
        """Dock multiple ligands with mock backend."""
        results = []
        for compound_id, ligand_data in ligands:
            if ligand_data is None:
                results.append(StandardizedDockingResult(
                    compound_id=compound_id,
                    status=DockingResultStatus.FAILED,
                    metadata={"error": "No ligand data"}
                ))
                continue
            
            # Add compound_id to ligand data if needed
            if hasattr(ligand_data, 'compound_id'):
                ligand_data.compound_id = compound_id
            
            result = self.dock_single(ligand_data, config)
            result.compound_id = compound_id
            results.append(result)
        
        return results
    
    def _mock_interactions(self, rng: np.random.Generator) -> list[dict]:
        """Generate mock interaction data."""
        interaction_types = ["hydrogen_bond", "pi_stack", "hydrophobic", "ionic"]
        n_interactions = rng.integers(1, 5)
        
        return [
            {
                "type": interaction_types[rng.integers(0, len(interaction_types))],
                "residue": f"RES{rng.integers(1, 300)}",
                "distance": round(rng.uniform(2.0, 5.0), 2)
            }
            for _ in range(n_interactions)
        ]


class DockingEngine:
    """
    Unified docking engine with standardized output.
    
    Provides a consistent interface regardless of the underlying
    docking software, ensuring compatibility with consensus scoring.
    """
    
    def __init__(
        self,
        backend: DockingBackend | None = None,
        default_config: DockingConfig | None = None
    ):
        """
        Initialize docking engine.
        
        Args:
            backend: Docking backend (defaults to MockDockingBackend)
            default_config: Default docking configuration
        """
        self.backend = backend or MockDockingBackend()
        self.default_config = default_config or DockingConfig()
    
    def dock(
        self,
        compound_id: str,
        ligand_data: Any,
        config: DockingConfig | None = None
    ) -> StandardizedDockingResult:
        """
        Dock a single compound.
        
        Args:
            compound_id: Unique identifier for the compound
            ligand_data: Ligand data (RDKit mol, SMILES, file path, etc.)
            config: Docking configuration (uses default if None)
            
        Returns:
            StandardizedDockingResult with consistent format
        """
        config = config or self.default_config
        
        # Validate configuration
        issues = config.validate()
        if issues:
            logger.warning(f"Config issues: {issues}")
        
        try:
            result = self.backend.dock_single(ligand_data, config)
            result.compound_id = compound_id
            logger.debug(
                f"Docking {compound_id}: score={result.best_score:.2f}, "
                f"status={result.status.value}"
            )
            return result
            
        except Exception as e:
            logger.error(f"Docking failed for {compound_id}: {e}")
            return StandardizedDockingResult(
                compound_id=compound_id,
                status=DockingResultStatus.FAILED,
                metadata={"error": str(e)}
            )
    
    def dock_batch(
        self,
        compounds: list[tuple[str, Any]],
        config: DockingConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> list[StandardizedDockingResult]:
        """
        Dock multiple compounds.
        
        Args:
            compounds: List of (compound_id, ligand_data) tuples
            config: Docking configuration
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of StandardizedDockingResult
        """
        config = config or self.default_config
        total = len(compounds)
        
        logger.info(f"Starting batch docking of {total} compounds")
        
        results = []
        for i, (compound_id, ligand_data) in enumerate(compounds):
            result = self.dock(compound_id, ligand_data, config)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        # Summary
        n_success = sum(1 for r in results if r.status == DockingResultStatus.SUCCESS)
        n_failed = sum(1 for r in results if r.status == DockingResultStatus.FAILED)
        
        logger.info(
            f"Batch docking complete: {n_success} success, {n_failed} failed"
        )
        
        return results
    
    def filter_by_score(
        self,
        results: list[StandardizedDockingResult],
        min_score: float | None = None,
        max_score: float | None = None,
        top_n: int | None = None
    ) -> list[StandardizedDockingResult]:
        """
        Filter docking results by score criteria.
        
        Args:
            results: List of docking results
            min_score: Minimum binding affinity (kcal/mol, negative)
            max_score: Maximum binding affinity (kcal/mol, negative)
            top_n: Keep only top N results
            
        Returns:
            Filtered list of results
        """
        filtered = [
            r for r in results
            if r.status == DockingResultStatus.SUCCESS
        ]
        
        if min_score is not None:
            filtered = [r for r in filtered if r.best_score <= min_score]
        
        if max_score is not None:
            filtered = [r for r in filtered if r.best_score >= max_score]
        
        # Sort by score (most negative = best)
        filtered.sort(key=lambda r: r.best_score)
        
        if top_n is not None:
            filtered = filtered[:top_n]
        
        return filtered


# Integration with consensus scorer
from consensus_scorer import ScoringMethod, ScorerProtocol


class DockingConsensusScorer(ScorerProtocol):
    """
    Adapter to use DockingEngine as a consensus scorer.
    
    Implements ScorerProtocol for seamless integration with
    ConsensusScorer.
    """
    
    def __init__(
        self,
        docking_engine: DockingEngine,
        config: DockingConfig | None = None
    ):
        self.engine = docking_engine
        self.config = config
        self._cache: dict[str, float] = {}
    
    @property
    def method(self) -> ScoringMethod:
        return ScoringMethod.DOCKING
    
    def score(self, compound_id: str, mol_data: Any) -> float:
        """Score compound using docking (returns normalized positive score)."""
        if compound_id in self._cache:
            return self._cache[compound_id]
        
        result = self.engine.dock(compound_id, mol_data, self.config)
        
        if result.status != DockingResultStatus.SUCCESS:
            raise RuntimeError(f"Docking failed: {result.metadata.get('error', 'unknown')}")
        
        score = result.to_consensus_format()
        self._cache[compound_id] = score
        return score
    
    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._cache.clear()


if __name__ == "__main__":
    # Test docking engine
    @dataclass
    class MockLigand:
        compound_id: str
        smiles: str
    
    engine = DockingEngine(
        backend=MockDockingBackend(base_score=-9.0, score_std=1.0),
        default_config=DockingConfig(n_poses=5, seed=42)
    )
    
    # Single docking
    ligand = MockLigand("TEST_001", "CC(C)CC(N)C(=O)O")
    result = engine.dock("TEST_001", ligand)
    
    print("=" * 60)
    print("SINGLE DOCKING RESULT")
    print("=" * 60)
    print(result.to_dict())
    
    # Batch docking
    compounds = [
        (f"CMP_{i:04d}", MockLigand(f"CMP_{i:04d}", f"C{i}CC(N)C(=O)O"))
        for i in range(10)
    ]
    
    def progress(current, total):
        print(f"\rProgress: {current}/{total}", end="", flush=True)
    
    results = engine.dock_batch(compounds, progress_callback=progress)
    print("\n")
    
    # Filter results
    filtered = engine.filter_by_score(results, min_score=-10.0, top_n=3)
    
    print("=" * 60)
    print("TOP 3 FILTERED RESULTS")
    print("=" * 60)
    for r in filtered:
        print(f"{r.compound_id}: {r.best_score:.2f} kcal/mol "
              f"(normalized: {r.normalized_score:.2f})")
    
    # Test consensus adapter
    print("\n" + "=" * 60)
    print("CONSENSUS SCORER ADAPTER TEST")
    print("=" * 60)
    
    adapter = DockingConsensusScorer(engine)
    score = adapter.score("TEST_002", MockLigand("TEST_002", "CCC(N)C(=O)O"))
    print(f"Consensus-compatible score: {score:.4f}")
```

---

## File 3: `arp_v3/agents/virtual_screen/ml_affinity.py`

```python
"""
ML-Based Affinity Prediction

Machine learning models for predicting binding affinity with
probability estimates for consensus scoring integration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported ML model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class FeatureType(Enum):
    """Supported molecular feature types."""
    MORGAN_FINGERPRINT = "morgan_fp"
    MACCS_KEYS = "maccs_keys"
    RDKIT_DESCRIPTORS = "rdkit_descriptors"
    GRAPH_CONV = "graph_conv"
    CUSTOM = "custom"


@dataclass
class PredictionResult:
    """Standardized prediction result."""
    compound_id: str
    predicted_affinity: float  # pIC50 or similar
    predicted_proba: float  # Probability of being active (0-1)
    uncertainty: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def normalized_score(self) -> float:
        """Return score normalized for consensus (higher = better)."""
        # Assuming pIC50 range [4, 10] -> [0, 1]
        return max(0.0, min(1.0, (self.predicted_affinity - 4.0) / 6.0))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "predicted_affinity": round(self.predicted_affinity, 4),
            "predicted_proba": round(self.predicted_proba, 4),
            "uncertainty": round(self.uncertainty, 4),
            "normalized_score": round(self.normalized_score, 4),
            "metadata": self.metadata
        }


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction."""
    
    @property
    @abstractmethod
    def feature_type(self) -> FeatureType:
        ...
    
    @abstractmethod
    def extract(self, mol_data: Any) -> np.ndarray:
        """Extract features from molecular data."""
        ...
    
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        ...


class MockFeatureExtractor(FeatureExtractor):
    """Mock feature extractor for testing."""
    
    def __init__(self, n_features: int = 128):
        self.n_features = n_features
        self._names = [f"feature_{i}" for i in range(n_features)]
    
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.CUSTOM
    
    def extract(self, mol_data: Any) -> np.ndarray:
        if isinstance(mol_data, np.ndarray):
            return mol_data[:self.n_features]
        return np.random.rand(self.n_features)
    
    @property
    def feature_names(self) -> list[str]:
        return self._names


class MLPredictor(ABC):
    """Abstract base class for ML predictors."""
    
    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        ...
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict affinity values."""
        ...
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability of activity."""
        ...
    
    @abstractmethod
    def predict_uncertainty(self, features: np.ndarray) -> np.ndarray:
        """Predict uncertainty estimates."""
        ...


class MockMLPredictor(MLPredictor):
    """Mock ML predictor for testing."""
    
    def __init__(
        self,
        base_affinity: float = 7.0,
        affinity_std: float = 1.0,
        seed: int = 42
    ):
        self.base_affinity = base_affinity
        self.affinity_std = affinity_std
        self.rng = np.random.default_rng(seed)
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.RANDOM_FOREST
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n_samples = features.shape[0]
        return self.rng.normal(self.base_affinity, self.affinity_std, n_samples)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability using sigmoid transformation."""
        affinities = self.predict(features)
        # Convert pIC50 to probability (pIC50 7+ = likely active)
        proba = 1.0 / (1.0 + np.exp(-(affinities - 6.5) * 2))
        return proba
    
    def predict_uncertainty(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n_samples = features.shape[0]
        return self.rng.uniform(0.3, 1.5, n_samples)


class AffinityPredictor:
    """
    ML-based affinity prediction with probability estimates.
    
    Provides standardized interface for predicting binding affinity
    with uncertainty quantification and probability outputs for
    consensus scoring integration.
    """
    
    def __init__(
        self,
        model: MLPredictor,
        feature_extractor: FeatureExtractor,
        activity_threshold: float = 6.5,
        cache_predictions: bool = True
    ):
        """
        Initialize affinity predictor.
        
        Args:
            model: ML prediction model
            feature_extractor: Feature extraction pipeline
            activity_threshold: pIC50 threshold for activity classification
            cache_predictions: Whether to cache predictions
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.activity_threshold = activity_threshold
        self._cache: dict[str, PredictionResult] = {} if cache_predictions else None
    
    def predict(
        self,
        compound_id: str,
        mol_data: Any,
        include_uncertainty: bool = True
    ) -> PredictionResult:
        """
        Predict affinity for a single compound.
        
        Args:
            compound_id: Unique compound identifier
            mol_data: Molecular data (RDKit mol, SMILES, etc.)
            include_uncertainty: Whether to calculate uncertainty
            
        Returns:
            PredictionResult with affinity, probability, and uncertainty
        """
        # Check cache
        if self._cache is not None and compound_id in self._cache:
            return self._cache[compound_id]
        
        try:
            # Extract features
            features = self.feature_extractor.extract(mol_data)
            
            # Make predictions
            affinity = float(self.model.predict(features)[0])
            proba = float(self.model.predict_proba(features)[0])
            uncertainty = (
                float(self.model.predict_uncertainty(features)[0])
                if include_uncertainty else 0.0
            )
            
            result = PredictionResult(
                compound_id=compound_id,
                predicted_affinity=affinity,
                predicted_proba=proba,
                uncertainty=uncertainty,
                metadata={
                    "model_type": self.model.model_type.value,
                    "feature_type": self.feature_extractor.feature_type.value
                }
            )
            
            # Cache result
            if self._cache is not None:
                self._cache[compound_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {compound_id}: {e}")
            return PredictionResult(
                compound_id=compound_id,
                predicted_affinity=float('nan'),
                predicted_proba=0.0,
                uncertainty=float('inf'),
                metadata={"error": str(e)}
            )
    
    def predict_batch(
        self,
        compounds: list[tuple[str, Any]],
        include_uncertainty: bool = True
    ) -> list[PredictionResult]:
        """
        Predict affinity for multiple compounds.
        
        Args:
            compounds: List of (compound_id, mol_data) tuples
            include_uncertainty: Whether to calculate uncertainty
            
        Returns:
            List of PredictionResult
        """
        results = []
        for compound_id, mol_data in compounds:
            result = self.predict(compound_id, mol_data, include_uncertainty)
            results.append(result)
        
        logger.info(
            f"Batch prediction complete: {len(results)} compounds, "
            f"model={self.model.model_type.value}"
        )
        
        return results
    
    def filter_by_probability(
        self,
        results: list[PredictionResult],
        min_proba: float = 0.5,
        min_affinity: float | None = None
    ) -> list[PredictionResult]:
        """
        Filter predictions by activity probability.
        
        Args:
            results: List of prediction results
            min_proba: Minimum probability threshold
            min_affinity: Optional minimum affinity threshold
            
        Returns:
            Filtered list sorted by probability (descending)
        """
        filtered = [
            r for r in results
            if r.predicted_proba >= min_proba
            and not np.isnan(r.predicted_affinity)
        ]
        
        if min_affinity is not None:
            filtered = [r for r in filtered if r.predicted_affinity >= min_affinity]
        
        filtered.sort(key=lambda r: r.predicted_proba, reverse=True)
        return filtered
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        if self._cache is not None:
            self._cache.clear()


# Integration with consensus scorer
from consensus_scorer import ScoringMethod, ScorerProtocol


class MLConsensusScorer(ScorerProtocol):
    """
    Adapter to use AffinityPredictor as a consensus scorer.
    
    Implements ScorerProtocol for seamless integration with
    ConsensusScorer. Uses probability output for scoring.
    """
    
    def __init__(
        self,
        predictor: AffinityPredictor,
        use_probability: bool = True
    ):
        """
        Initialize ML consensus scorer.
        
        Args:
            predictor: AffinityPredictor instance
            use_probability: If True, use probability; else use normalized affinity
        """
        self.predictor = predictor
        self.use_probability = use_probability
        self._cache: dict[str, float] = {}
    
    @property
    def method(self) -> ScoringMethod:
        return ScoringMethod.ML_AFFINITY
    
    def score(self, compound_id: str, mol_data: Any) -> float:
        """Score compound using ML prediction."""
        if compound_id in self._cache:
            return self._cache[compound_id]
        
        result = self.predictor.predict(compound_id, mol_data)
        
        if np.isnan(result.predicted_affinity):
            raise RuntimeError(f"ML prediction failed for {compound_id}")
        
        if self.use_probability:
            score = result.predicted_proba
        else:
            score = result.normalized_score
        
        self._cache[compound_id] = score
        return score
    
    def predict_proba(self, compound_id: str, mol_data: Any) -> tuple[float, float]:
        """
        Get probability and uncertainty for a compound.
        
        Returns:
            Tuple of (probability, uncertainty)
        """
        result = self.predictor.predict(compound_id, mol_data)
        return result.predicted_proba, result.uncertainty
    
    def clear_cache(self) -> None:
        """Clear score cache."""
        self._cache.clear()


if __name__ == "__main__":
    # Test ML affinity predictor
    predictor = AffinityPredictor(
        model=MockMLPredictor(base_affinity=7.5, affinity_std=0.8, seed=42),
        feature_extractor=MockFeatureExtractor(n_features=128)
    )
    
    # Single prediction
    print("=" * 60)
    print("SINGLE PREDICTION")
    print("=" * 60)
    result = predictor.predict("TEST_001", np.random.rand(128))
    print(result.to_dict())
    
    # Batch prediction
    compounds = [
        (f"CMP_{i:04d}", np.random.rand(128))
        for i in range(20)
    ]
    
    results = predictor.predict_batch(compounds)
    
    # Filter by probability
    filtered = predictor.filter_by_probability(results, min_proba=0.6)
    
    print("\n" + "=" * 60)
    print("TOP 5 BY PROBABILITY (proba >= 0.6)")
    print("=" * 60)
    for r in filtered[:5]:
        print(f"{r.compound_id}: pIC50={r.predicted_affinity:.2f}, "
              f"proba={r.predicted_proba:.4f}, "
              f"uncertainty={r.uncertainty:.2f}")
    
    # Test consensus adapter
    print("\n" + "=" * 60)
    print("CONSENSUS SCORER ADAPTER TEST")
    print("=" * 60)
    
    adapter = MLConsensusScorer(predictor, use_probability=True)
    score = adapter.score("TEST_002", np.random.rand(128))
    print(f"Consensus-compatible score (probability): {score:.4f}")
    
    proba, uncertainty = adapter.predict_proba("TEST_003", np.random.rand(128))
    print(f"Probability: {proba:.4f}, Uncertainty: {uncertainty:.4f}")
```

---

## File 4: `arp_v3/agents/virtual_screen/pharmacophore.py`

```python
"""
Pharmacophore Screening Module

3D pharmacophore-based screening for virtual compound filtering
with standardized output for consensus scoring integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PharmacophoreFeature(Enum):
    """Pharmacophore feature types."""
    HYDROGEN_BOND_DONOR = "HBD"
    HYDROGEN_BOND_ACCEPTOR = "HBA"
    HYDROPHOBIC = "HYD"
    AROMATIC = "ARO"
    POSITIVE_IONIZABLE = "POS"
    NEGATIVE_IONIZABLE = "NEG"
    METAL_BINDER = "MET"
    HALOGEN_BOND = "HAL"


@dataclass
class PharmacophorePoint:
    """Single pharmacophore feature point."""
    feature_type: PharmacophoreFeature
    x: float
    y: float
    z: float
    radius: float = 1.0  # Tolerance radius in Angstroms
    required: bool = True  # Whether this feature is required
    
    def distance_to(self, other_point: PharmacophorePoint) -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt(
            (self.x - other_point.x) ** 2 +
            (self.y - other_point.y) ** 2 +
            (self.z - other_point.z) ** 2
        )


@dataclass
class PharmacophoreModel:
    """
    3D pharmacophore model definition.
    
    Defines the spatial arrangement of pharmacophore features
    that a compound should match for activity.
    """
    name: str
    features: list[PharmacophorePoint]
    min_required_matches: int = 0  # 0 = all required features
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.min_required_matches == 0:
            self.min_required_matches = sum(
                1 for f in self.features if f.required
            )
    
    @property
    def feature_counts(self) -> dict[PharmacophoreFeature, int]:
        """Count features by type."""
        counts: dict[PharmacophoreFeature, int] = {}
        for f in self.features:
            counts[f.feature_type] = counts.get(f.feature_type, 0) + 1
        return counts
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_features": len(self.features),
            "min_required_matches": self.min_required_matches,
            "feature_counts": {k.value: v for k, v in self.feature_counts.items()},
            "metadata": self.metadata
        }


@dataclass
class PharmacophoreMatch:
    """Result of matching a compound against a pharmacophore."""
    compound_id: str
    model_name: str
    n_matched: int
    n_total_required: int
    matched_features: list[PharmacophoreFeature]
    unmatched_features: list[PharmacophoreFeature]
    rmsd: float = 0.0
    fit_score: float = 0.0
    passed: bool = False
    
    @property
    def match_ratio(self) -> float:
        """Ratio of matched to required features."""
        if self.n_total_required == 0:
            return 0.0
        return self.n_matched / self.n_total_required
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "model_name": self.model_name,
            "n_matched": self.n_matched,
            "n_total_required": self.n_total_required,
            "match_ratio": round(self.match_ratio, 4),
            "fit_score": round(self.fit_score, 4),
            "passed": self.passed,
            "matched_features": [f.value for f in self.matched_features],
            "unmatched_features": [f.value for f in self.unmatched_features]
        }


class PharmacophoreScreener:
    """
    Pharmacophore-based compound screening.
    
    Screens compounds against 3D pharmacophore models to identify
    those with matching feature arrangements.
    """
    
    def __init__(
        self,
        models: list[PharmacophoreModel] | None = None,
        rmsd_threshold: float = 2.0,
        strict_mode: bool = False
    ):
        """
        Initialize pharmacophore screener.
        
        Args:
            models: List of pharmacophore models to screen against
            rmsd_threshold: Maximum RMSD for a valid match (Angstroms)
            strict_mode: If True, all required features must match exactly
        """
        self.models: list[PharmacophoreModel] = models or []
        self.rmsd_threshold = rmsd_threshold
        self.strict_mode = strict_mode
    
    def add_model(self, model: PharmacophoreModel) -> None:
        """Add a pharmacophore model."""
        self.models.append(model)
        logger.debug(f"Added pharmacophore model: {model.name}")
    
    def remove_model(self, name: str) -> None:
        """Remove a pharmacophore model by name."""
        self.models = [m for m in self.models if m.name != name]
    
    def _extract_pharmacophore_features(
        self,
        mol_data: Any
    ) -> list[PharmacophorePoint]:
        """
        Extract pharmacophore features from molecular data.
        
        In production, this would use RDKit or similar to identify
        HBD, HBA, hydrophobic, aromatic, etc. features with 3D coords.
        """
        # Mock implementation for testing
        if hasattr(mol_data, 'pharmacophore_points'):
            return mol_data.pharmacophore_points
        
        # Generate random features for mock data
        rng = np.random.default_rng(42)
        n_features = rng.integers(3, 8)
        
        feature_types = list(PharmacophoreFeature)
        points = []
        
        for _ in range(n_features):
            ftype = feature_types[rng.integers(0, len(feature_types))]
            points.append(PharmacophorePoint(
                feature_type=ftype,
                x=rng.uniform(-10, 10),
                y=rng.uniform(-10, 10),
                z=rng.uniform(-10, 10),
                radius=rng.uniform(0.8, 1.5)
            ))
        
        return points
    
    def _match_features(
        self,
        compound_features: list[PharmacophorePoint],
        model: PharmacophoreModel
    ) -> PharmacophoreMatch:
        """
        Match compound features against pharmacophore model.
        
        Uses greedy matching algorithm with distance tolerance.
        """
        required_model_features = [f for f in model.features if f.required]
        matched: list[PharmacophoreFeature] = []
        unmatched: list[PharmacophoreFeature] = []
        used_compound_features: set[int] = set()
        
        for model_feat in required_model_features:
            best_match_idx = -1
            best_distance = float('inf')
            
            for i, comp_feat in enumerate(compound_features):
                if i in used_compound_features:
                    continue
                
                # Check feature type compatibility
                if comp_feat.feature_type != model_feat.feature_type:
                    # Allow some cross-compatibility
                    compatible = self._check_feature_compatibility(
                        comp_feat.feature_type, model_feat.feature_type
                    )
                    if not compatible:
                        continue
                
                # Calculate distance with tolerance
                distance = comp_feat.distance_to(model_feat)
                effective_radius = comp_feat.radius + model_feat.radius
                
                if distance < effective_radius and distance < best_distance:
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx >= 0:
                matched.append(model_feat.feature_type)
                used_compound_features.add(best_match_idx)
            else:
                unmatched.append(model_feat.feature_type)
        
        # Calculate fit score based on match quality
        fit_score = self._calculate_fit_score(
            len(matched),
            len(required_model_features),
            best_distance if matched else float('inf')
        )
        
        # Calculate RMSD of matched features
        rmsd = self._calculate_rmsd(
            compound_features,
            required_model_features,
            matched
        )
        
        # Determine if match passed
        passed = (
            len(matched) >= model.min_required_matches and
            (not self.strict_mode or len(unmatched) == 0) and
            rmsd <= self.rmsd_threshold
        )
        
        return PharmacophoreMatch(
            compound_id="",
            model_name=model.name,
            n_matched=len(matched),
            n_total_required=len(required_model_features),
            matched_features=matched,
            unmatched_features=unmatched,
            rmsd=rmsd,
            fit_score=fit_score,
            passed=passed
        )
    
    def _check_feature_compatibility(
        self,
        feat1: PharmacophoreFeature,
        feat2: PharmacophoreFeature
    ) -> bool:
        """Check if two feature types are compatible for matching."""
        compatible_pairs = {
            (PharmacophoreFeature.HYDROPHOBIC, PharmacophoreFeature.AROMATIC),
            (PharmacophoreFeature.AROMATIC, PharmacophoreFeature.HYDROPHOBIC),
            (PharmacophoreFeature.POSITIVE_IONIZABLE, PharmacophoreFeature.HYDROGEN_BOND_DONOR),
            (PharmacophoreFeature.HYDROGEN_BOND_DONOR, PharmacophoreFeature.POSITIVE_IONIZABLE),
            (PharmacophoreFeature.NEGATIVE_IONIZABLE, PharmacophoreFeature.HYDROGEN_BOND_ACCEPTOR),
            (PharmacophoreFeature.HYDROGEN_BOND_ACCEPTOR, PharmacophoreFeature.NEGATIVE_IONIZABLE),
        }
        return (feat1, feat2) in compatible_pairs
    
    def _calculate_fit_score(
        self,
        n_matched: int,
        n_required: int,
        best_distance: float
    ) -> float:
        """Calculate overall fit score (0-1, higher = better)."""
        if n_required == 0:
            return 0.0
        
        match_component = n_matched / n_required
        
        if best_distance == float('inf'):
            distance_component = 0.0
        else:
            distance_component = max(0.0, 1.0 - best_distance / 5.0)
        
        return 0.7 * match_component + 0.3 * distance_component
    
    def _calculate_rmsd(
        self,
        compound_features: list[PharmacophorePoint],
        model_features: list[PharmacophorePoint],
        matched: list[PharmacophoreFeature]
    ) -> float:
        """Calculate RMSD of matched feature positions."""
        if not matched:
            return float('inf')
        
        squared_distances = []
        
        for model_feat in model_features:
            if model_feat.feature_type not in matched:
                continue
            
            # Find closest matching compound feature
            min_dist = float('inf')
            for comp_feat in compound_features:
                if comp_feat.feature_type == model_feat.feature_type:
                    dist = comp_feat.distance_to(model_feat)
                    min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                squared_distances.append(min_dist ** 2)
        
        if not squared_distances:
            return float('inf')
        
        return np.sqrt(np.mean(squared_distances))
    
    def screen(
        self,
        compound_id: str,
        mol_data: Any,
        model: PharmacophoreModel | None = None
    ) -> PharmacophoreMatch:
        """
        Screen a single compound against pharmacophore model(s).
        
        Args:
            compound_id: Compound identifier
            mol_data: Molecular data with 3D coordinates
            model: Specific model to use (uses best match if None)
            
        Returns:
            Best PharmacophoreMatch across all models
        """
        compound_features = self._extract_pharmacophore_features(mol_data)
        
        models_to_use = [model] if model else self.models
        
        if not models_to_use:
            raise ValueError("No pharmacophore models available")
        
        best_match: Optional[PharmacophoreMatch] = None
        
        for pharm_model in models_to_use:
            match = self._match_features(compound_features, pharm_model)
            match.compound_id = compound_id
            
            if best_match is None or match.fit_score > best_match.fit_score:
                best_match = match
        
        return best_match
    
    def screen_batch(
        self,
        compounds: list[tuple[str, Any]],
        model: PharmacophoreModel | None = None
    ) -> list[PharmacophoreMatch]:
        """
        Screen multiple compounds.
        
        Args:
            compounds: List of (compound_id, mol_data) tuples
            model: Specific model to use
            
        Returns:
            List of PharmacophoreMatch sorted by fit_score (descending)
        """
        results = []
        
        for compound_id, mol_data in compounds:
            try:
                match = self.screen(compound_id, mol_data, model)
                results.append(match)
            except Exception as e:
                logger.warning(f"Screening failed for {compound_id}: {e}")
                results.append(PharmacophoreMatch(
                    compound_id=compound_id,
                    model_name=model.name if model else "unknown",
                    n_matched=0,
                    n_total_required=0,
                    matched_features=[],
                    unmatched_features=[],
                    rmsd=float('inf'),
                    fit_score=0.0,
                    passed=False
                ))
        
        results.sort(key=lambda m: m.fit_score, reverse=True)
        
        n_passed = sum(1 for m in results if m.passed)
        logger.info(
            f"Pharmacophore screening complete: {n_passed}/{len(results)} passed"
        )
        
        return results
    
    def filter_passed(
        self,
        matches: list[PharmacophoreMatch]
    ) -> list[PharmacophoreMatch]:
        """Filter to only matches that passed the threshold."""
        return [m for m in matches if m.passed]


# Integration with consensus scorer
from consensus_scorer import ScoringMethod, ScorerProtocol


class PharmacophoreConsensusScorer(ScorerProtocol):
    """
    Adapter to use PharmacophoreScreener as a consensus scorer.
    """
    
    def __init__(
        self,
        screener: PharmacophoreScreener,
        model: PharmacophoreModel | None = None
    ):
        self.screener = screener
        self.model = model
        self._cache: dict[str, float] = {}
    
    @property
    def method(self) -> ScoringMethod:
        return ScoringMethod.PHARMACOPHORE
    
    def score(self, compound_id: str, mol_data: Any) -> float:
        """Score compound using pharmacophore fit."""
        if compound_id in self._cache:
            return self._cache[compound_id]
        
        match = self.screener.screen(compound_id, mol_data, self.model)
        
        if not match.passed and match.fit_score < 0.1:
            raise RuntimeError(f"Pharmacophore match failed for {compound_id}")
        
        # Use fit_score as the score (already 0-1)
        score = match.fit_score
        self._cache[compound_id] = score
        return score
    
    def clear_cache(self) -> None:
        """Clear score cache."""
        self._cache.clear()


def create_example_pharmacophore() -> PharmacophoreModel:
    """Create an example kinase inhibitor pharmacophore."""
    return PharmacophoreModel(
        name="kinase_inhibitor_v1",
        features=[
            PharmacophorePoint(
                feature_type=PharmacophoreFeature.HYDROGEN_BOND_ACCEPTOR,
                x=0.0, y=0.0, z=0.0,
                radius=1.0, required=True
            ),
            PharmacophorePoint(
                feature_type=PharmacophoreFeature.HYDROGEN_BOND_DONOR,
                x=3.5, y=1.0, z=0.5,
                radius=1.0, required=True
            ),
            PharmacophorePoint(
                feature_type=PharmacophoreFeature.HYDROPHOBIC,
                x=-2.0, y=3.0, z=1.0,
                radius=1.5, required=True
            ),
            PharmacophorePoint(
                feature_type=PharmacophoreFeature.AROMATIC,
                x=1.0, y=-3.0, z=0.0,
                radius=1.2, required=False
            ),
        ],
        min_required_matches=3,
        metadata={"target": "CDK2", "source": "crystal_structure_1HCK"}
    )


if __name__ == "__main__":
    # Create pharmacophore model
    model = create_example_pharmacophore()
    print("=" * 60)
    print("PHARMACOPHORE MODEL")
    print("=" * 60)
    print(model.to_dict())
    
    # Create screener
    screener = PharmacophoreScreener(
        models=[model],
        rmsd_threshold=2.0
    )
    
    # Screen mock compounds
    np.random.seed(42)
    compounds = [
        (f"CMP_{i:04d}", {"mock": True, "seed": i})
        for i in range(15)
    ]
    
    results = screener.screen_batch(compounds)
    
    print("\n" + "=" * 60)
    print("SCREENING RESULTS (TOP 5)")
    print("=" * 60)
    for r in results[:5]:
        print(f"{r.compound_id}: fit={r.fit_score:.4f}, "
              f"matched={r.n_matched}/{r.n_total_required}, "
              f"passed={r.passed}")
    
    # Test consensus adapter
    print("\n" + "=" * 60)
    print("CONSENSUS SCORER ADAPTER TEST")
    print("=" * 60)
    
    adapter = PharmacophoreConsensusScorer(screener, model)
    score = adapter.score("TEST_001", {"mock": True, "seed": 5})
    print(f"Consensus-compatible score: {score:.4f}")
```

---

## File 5: `arp_v3/pipelines/screening_pipeline.py`

```python
"""
Integrated Screening Pipeline

Orchestrates the complete virtual screening workflow combining
docking, ML prediction, and pharmacophore screening with
consensus scoring.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from agents.virtual_screen.consensus_scorer import (
    ConsensusScorer,
    ConsensusResult,
    ScoreTransformer,
    ScoringMethod
)
from agents.virtual_screen.docking_engine import (
    DockingEngine,
    DockingConfig,
    DockingResultStatus,
    StandardizedDockingResult,
    DockingConsensusScorer
)
from agents.virtual_screen.ml_affinity import (
    AffinityPredictor,
    MLPredictor,
    FeatureExtractor,
    PredictionResult,
    MLConsensusScorer
)
from agents.virtual_screen.pharmacophore import (
    PharmacophoreScreener,
    PharmacophoreModel,
    PharmacophoreMatch,
    PharmacophoreConsensusScorer
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the screening pipeline."""
    # Docking settings
    enable_docking: bool = True
    docking_weight: float = 1.5
    docking_min_score: float = -10.0  # kcal/mol
    docking_n_poses: int = 9
    
    # ML settings
    enable_ml: bool = True
    ml_weight: float = 1.0
    ml_min_proba: float = 0.5
    
    # Pharmacophore settings
    enable_pharmacophore: bool = True
    pharmacophore_weight: float = 0.8
    pharmacophore_min_fit: float = 0.3
    
    # Consensus settings
    consensus_transformer: str = "z_score"
    consensus_min_scorers: int = 2
    top_n: int = 100
    
    # Performance settings
    parallel: bool = False
    n_workers: int = 4
    save_intermediate: bool = True
    output_dir: str = "./screening_results"


@dataclass
class ScreeningStageResult:
    """Results from a single screening stage."""
    stage_name: str
    n_input: int
    n_output: int
    n_failed: int
    execution_time: float
    results: list[Any] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_failed": self.n_failed,
            "execution_time": round(self.execution_time, 3),
            "pass_rate": round(self.n_output / max(self.n_input, 1), 4)
        }


@dataclass
class PipelineResult:
    """Complete pipeline results."""
    pipeline_id: str
    timestamp: str
    config: PipelineConfig
    stage_results: dict[str, ScreeningStageResult]
    final_results: list[ConsensusResult]
    summary: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "timestamp": self.timestamp,
            "config": {
                "enable_docking": self.config.enable_docking,
                "enable_ml": self.config.enable_ml,
                "enable_pharmacophore": self.config.enable_pharmacophore,
                "top_n": self.config.top_n,
                "parallel": self.config.parallel
            },
            "stages": {
                k: v.to_dict() for k, v in self.stage_results.items()
            },
            "summary": self.summary,
            "top_compounds": [
                r.to_dict() for r in self.final_results[:10]
            ]
        }


class ScreeningPipeline:
    """
    Integrated virtual screening pipeline.
    
    Orchestrates docking, ML prediction, and pharmacophore screening
    with consensus scoring to produce robust compound rankings.
    
    Features:
    - Configurable stage enabling/disabling
    - Intermediate filtering between stages
    - Progress tracking and timing
    - Result persistence
    - Comprehensive summary statistics
    """
    
    def __init__(
        self,
        docking_engine: Optional[DockingEngine] = None,
        ml_predictor: Optional[AffinityPredictor] = None,
        pharmacophore_screener: Optional[PharmacophoreScreener] = None,
        config: PipelineConfig | None = None
    ):
        """
        Initialize screening pipeline.
        
        Args:
            docking_engine: Docking engine instance
            ml_predictor: ML affinity predictor instance
            pharmacophore_screener: Pharmacophore screener instance
            config: Pipeline configuration
        """
        self.docking_engine = docking_engine
        self.ml_predictor = ml_predictor
        self.pharmacophore_screener = pharmacophore_screener
        self.config = config or PipelineConfig()
        
        self._progress_callback: Optional[Callable[[str, int, int], None]] = None
        self._stage_results: dict[str, ScreeningStageResult] = {}
    
    def set_progress_callback(
        self,
        callback: Callable[[str, int, int], None]
    ) -> None:
        """Set callback for progress updates (stage, current, total)."""
        self._progress_callback = callback
    
    def _report_progress(self, stage: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(stage, current, total)
    
    def _run_docking_stage(
        self,
        compounds: list[tuple[str, Any]]
    ) -> tuple[list[tuple[str, Any]], ScreeningStageResult]:
        """Run docking stage and filter results."""
        if not self.config.enable_docking or not self.docking_engine:
            return compounds, ScreeningStageResult(
                stage_name="docking",
                n_input=len(compounds),
                n_output=len(compounds),
                n_failed=0,
                execution_time=0.0,
                results=[]
            )
        
        logger.info(f"Running docking stage on {len(compounds)} compounds")
        start_time = time.time()
        
        docking_config = DockingConfig(
            n_poses=self.config.docking_n_poses
        )
        
        results = self.docking_engine.dock_batch(
            compounds,
            config=docking_config,
            progress_callback=lambda c, t: self._report_progress("docking", c, t)
        )
        
        # Filter by score
        filtered_results = self.docking_engine.filter_by_score(
            results,
            min_score=self.config.docking_min_score
        )
        
        passed_ids = {r.compound_id for r in filtered_results}
        filtered_compounds = [
            (cid, mol) for cid, mol in compounds
            if cid in passed_ids
        ]
        
        exec_time = time.time() - start_time
        n_failed = sum(1 for r in results if r.status != DockingResultStatus.SUCCESS)
        
        stage_result = ScreeningStageResult(
            stage_name="docking",
            n_input=len(compounds),
            n_output=len(filtered_compounds),
            n_failed=n_failed,
            execution_time=exec_time,
            results=filtered_results
        )
        
        logger.info(
            f"Docking complete: {len(filtered_compounds)} passed, "
            f"{n_failed} failed, {exec_time:.2f}s"
        )
        
        return filtered_compounds, stage_result
    
    def _run_ml_stage(
        self,
        compounds: list[tuple[str, Any]]
    ) -> tuple[list[tuple[str, Any]], ScreeningStageResult]:
        """Run ML prediction stage and filter results."""
        if not self.config.enable_ml or not self.ml_predictor:
            return compounds, ScreeningStageResult(
                stage_name="ml_prediction",
                n_input=len(compounds),
                n_output=len(compounds),
                n_failed=0,
                execution_time=0.0,
                results=[]
            )
        
        logger.info(f"Running ML prediction stage on {len(compounds)} compounds")
        start_time = time.time()
        
        results = self.ml_predictor.predict_batch(compounds)
        
        # Filter by probability
        filtered_results = self.ml_predictor.filter_by_probability(
            results,
            min_proba=self.config.ml_min_proba
        )
        
        passed_ids = {r.compound_id for r in filtered_results}
        filtered_compounds = [
            (cid, mol) for cid, mol in compounds
            if cid in passed_ids
        ]
        
        exec_time = time.time() - start_time
        n_failed = sum(1 for r in results if np.isnan(r.predicted_affinity))
        
        stage_result = ScreeningStageResult(
            stage_name="ml_prediction",
            n_input=len(compounds),
            n_output=len(filtered_compounds),
            n_failed=n_failed,
            execution_time=exec_time,
            results=filtered_results
        )
        
        logger.info(
            f"ML prediction complete: {len(filtered_compounds)} passed, "
            f"{n_failed} failed, {exec_time:.2f}s"
        )
        
        return filtered_compounds, stage_result
    
    def _run_pharmacophore_stage(
        self,
        compounds: list[tuple[str, Any]]
    ) -> tuple[list[tuple[str, Any]], ScreeningStageResult]:
        """Run pharmacophore screening stage and filter results."""
        if not self.config.enable_pharmacophore or not self.pharmacophore_screener:
            return compounds, ScreeningStageResult(
                stage_name="pharmacophore",
                n_input=len(compounds),
                n_output=len(compounds),
                n_failed=0,
                execution_time=0.0,
                results=[]
            )
        
        logger.info(
            f"Running pharmacophore stage on {len(compounds)} compounds"
        )
        start_time = time.time()
        
        results = self.pharmacophore_screener.screen_batch(compounds)
        
        # Filter by fit score
        filtered_results = [
            r for r in results
            if r.fit_score >= self.config.pharmacophore_min_fit
        ]
        
        passed_ids = {r.compound_id for r in filtered_results}
        filtered_compounds = [
            (cid, mol) for cid, mol in compounds
            if cid in passed_ids
        ]
        
        exec_time = time.time() - start_time
        n_failed = len(compounds) - len(filtered_compounds)
        
        stage_result = ScreeningStageResult(
            stage_name="pharmacophore",
            n_input=len(compounds),
            n_output=len(filtered_compounds),
            n_failed=n_failed,
            execution_time=exec_time,
            results=filtered_results
        )
        
        logger.info(
            f"Pharmacophore complete: {len(filtered_compounds)} passed, "
            f"{n_failed} failed, {exec_time:.2f}s"
        )
        
        return filtered_compounds, stage_result
    
    def _run_consensus_stage(
        self,
        compounds: list[tuple[str, Any]]
    ) -> tuple[list[ConsensusResult], ScreeningStageResult]:
        """Run consensus scoring on remaining compounds."""
        logger.info(f"Running consensus scoring on {len(compounds)} compounds")
        start_time = time.time()
        
        # Build consensus scorer with available methods
        consensus = ConsensusScorer(
            transformer=ScoreTransformer(self.config.consensus_transformer),
            min_scorers=self.config.consensus_min_scorers,
            weights={}
        )
        
        if self.config.enable_docking and self.docking_engine:
            consensus.add_scorer(
                DockingConsensusScorer(self.docking_engine),
                weight=self.config.docking_weight
            )
        
        if self.config.enable_ml and self.ml_predictor:
            consensus.add_scorer(
                MLConsensusScorer(self.ml_predictor),
                weight=self.config.ml_weight
            )
        
        if self.config.enable_pharmacophore and self.pharmacophore_screener:
            consensus.add_scorer(
                PharmacophoreConsensusScorer(self.pharmacophore_screener),
                weight=self.config.pharmacophore_weight
            )
        
        # Run consensus scoring
        results = consensus.score_compounds(
            compounds,
            parallel=self.config.parallel,
            n_workers=self.config.n_workers
        )
        
        # Limit to top N
        final_results = results[:self.config.top_n]
        
        exec_time = time.time() - start_time
        
        stage_result = ScreeningStageResult(
            stage_name="consensus",
            n_input=len(compounds),
            n_output=len(final_results),
            n_failed=0,
            execution_time=exec_time,
            results=final_results
        )
        
        logger.info(
            f"Consensus scoring complete: {len(final_results)} top compounds, "
            f"{exec_time:.2f}s"
        )
        
        return final_results, stage_result
    
    def run(
        self,
        compounds: list[tuple[str, Any]],
        pipeline_id: str | None = None
    ) -> PipelineResult:
        """
        Run the complete screening pipeline.
        
        Args:
            compounds: List of (compound_id, mol_data) tuples
            pipeline_id: Optional identifier for this run
            
        Returns:
            PipelineResult with all stage results and final rankings
        """
        pipeline_id = pipeline_id or f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting pipeline {pipeline_id} with {len(compounds)} compounds")
        
        self._stage_results = {}
        
        # Stage 1: Docking (optional filter)
        compounds_after_docking, docking_result = self._run_docking_stage(compounds)
        self._stage_results["docking"] = docking_result
        
        # Stage 2: ML Prediction (optional filter)
        compounds_after_ml, ml_result = self._run_ml_stage(compounds_after_docking)
        self._stage_results["ml_prediction"] = ml_result
        
        # Stage 3: Pharmacophore (optional filter)
        compounds_after_pharm, pharm_result = self._run_pharmacophore_stage(compounds_after_ml)
        self._stage_results["pharmacophore"] = pharm_result
        
        # Stage 4: Consensus Scoring
        final_results, consensus_result = self._run_consensus_stage(compounds_after_pharm)
        self._stage_results["consensus"] = consensus_result
        
        # Build summary
        summary = {
            "total_input": len(compounds),
            "total_output": len(final_results),
            "overall_pass_rate": len(final_results) / max(len(compounds), 1),
            "total_execution_time": sum(
                sr.execution_time for sr in self._stage_results.values()
            ),
            "stage_pass_rates": {
                k: v.to_dict()["pass_rate"]
                for k, v in self._stage_results.items()
            }
        }
        
        if final_results:
            scores = [r.consensus_score for r in final_results]
            summary["score_range"] = {
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "mean": round(np.mean(scores), 4)
            }
            summary["avg_confidence"] = round(
                np.mean([r.confidence for r in final_results]), 4
            )
            summary["n_flagged"] = sum(1 for r in final_results if r.flags)
        
        result = PipelineResult(
            pipeline_id=pipeline_id,
            timestamp=datetime.now().isoformat(),
            config=self.config,
            stage_results=self._stage_results,
            final_results=final_results,
            summary=summary
        )
        
        logger.info(
            f"Pipeline {pipeline_id} complete: "
            f"{len(compounds)} -> {len(final_results)} compounds"
        )
        
        return result
    
    def save_results(
        self,
        result: PipelineResult,
        output_dir: str | None = None
    ) -> Path:
        """
        Save pipeline results to JSON file.
        
        Args:
            result: PipelineResult to save
            output_dir: Output directory (uses config default if None)
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{result.pipeline_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        return output_file


def create_test_pipeline() -> ScreeningPipeline:
    """Create a pipeline with mock components for testing."""
    from agents.virtual_screen.docking_engine import MockDockingBackend
    from agents.virtual_screen.ml_affinity import MockMLPredictor, MockFeatureExtractor
    from agents.virtual_screen.pharmacophore import create_example_pharmacophore
    
    # Create mock components
    docking_engine = DockingEngine(
        backend=MockDockingBackend(base_score=-8.5, score_std=1.2),
        default_config=DockingConfig(n_poses=5, seed=42)
    )
    
    ml_predictor = AffinityPredictor(
        model=MockMLPredictor(base_affinity=7.0, affinity_std=1.0, seed=42),
        feature_extractor=MockFeatureExtractor(n_features=128)
    )
    
    pharm_model = create_example_pharmacophore()
    pharm_screener = PharmacophoreScreener(
        models=[pharm_model],
        rmsd_threshold=2.5
    )
    
    # Create pipeline with relaxed thresholds for testing
    config = PipelineConfig(
        docking_min_score=-12.0,  # Relaxed
        ml_min_proba=0.3,  # Relaxed
        pharmacophore_min_fit=0.1,  # Relaxed
        top_n=10
    )
    
    return ScreeningPipeline(
        docking_engine=docking_engine,
        ml_predictor=ml_predictor,
        pharmacophore_screener=pharm_screener,
        config=config
    )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test pipeline
    pipeline = create_test_pipeline()
    
    # Progress callback
    def on_progress(stage: str, current: int, total: int):
        print(f"\r  [{stage}] {current}/{total}", end="", flush=True)
    
    pipeline.set_progress_callback(on_progress)
    
    # Generate test compounds
    np.random.seed(42)
    compounds = [
        (f"CMP_{i:04d}", np.random.rand(128))
        for i in range(50)
    ]
    
    print("=" * 70)
    print("RUNNING SCREENING PIPELINE")
    print("=" * 70)
    print(f"Input: {len(compounds)} compounds\n")
    
    # Run pipeline
    result = pipeline.run(compounds, pipeline_id="test_run_001")
    print("\n")
    
    # Display results
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Pipeline ID: {result.pipeline_id}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Total input: {result.summary['total_input']}")
    print(f"Total output: {result.summary['total_output']}")
    print(f"Overall pass rate: {result.summary['overall_pass_rate']:.2%}")
    print(f"Total time: {result.summary['total_execution_time']:.2f}s")
    
    print("\n" + "-" * 70)
    print("STAGE RESULTS")
    print("-" * 70)
    for stage_name, stage_result in result.stage_results.items():
        print(f"  {stage_name:20s}: "
              f"{stage_result.n_input} -> {stage_result.n_output} "
              f"({stage_result.to_dict()['pass_rate']:.1%}) "
              f"[{stage_result.execution_time:.2f}s]")
    
    print("\n" + "-" * 70)
    print("TOP 5 COMPOUNDS")
    print("-" * 70)
    for r in result.final_results[:5]:
        print(f"  Rank {r.rank}: {r.compound_id}")
        print(f"    Consensus: {r.consensus_score:.4f} | "
              f"Confidence: {r.confidence:.4f}")
        if r.flags:
            print(f"    Flags: {r.flags}")
    
    # Save results
    output_path = pipeline.save_results(result, output_dir="./test_output")
    print(f"\nResults saved to: {output_path}")
```

---

## Summary of Improvements

| Module | Improvement | Key Features |
|--------|-------------|--------------|
| `consensus_scorer.py` | **NEW** | Multi-method normalization (min-max, z-score, rank, sigmoid), outlier detection, confidence estimation, parallel processing |
| `docking_engine.py` | **MODIFIED** | Standardized `StandardizedDockingResult` output, `to_consensus_format()` method, `DockingConsensusScorer` adapter |
| `ml_affinity.py` | **MODIFIED** | Added `predict_proba()` interface, `MLConsensusScorer` adapter, uncertainty quantification |
| `pharmacophore.py` | **NEW** | 3D pharmacophore matching, feature compatibility rules, fit scoring, `PharmacophoreConsensusScorer` adapter |
| `screening_pipeline.py` | **MODIFIED** | Orchestrates all stages with intermediate filtering, progress tracking, result persistence, comprehensive summary |