# BrownBioTech Platform — Iteration 6/100
## Metabolism-First Targeting Enhancement

Below are the complete modules for this iteration, organized by the planned file structure.

---

## 1. Shallow Surface Binding Engine

### `brownbiotech/arp_v3/agents/virtual_screen/shallow_surface_detector.py`

```python
"""
Shallow Surface Detector — Identifies shallow, groove-like binding sites
on protein surfaces that are typical of metabolism-first targets
(e.g., lipid-transfer proteins, membrane-proximal enzymes).

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SurfaceTopology(Enum):
    """Classification of surface binding site topology."""
    SHALLOW_GROOVE = "shallow_groove"
    DEEP_POCKET = "deep_pocket"
    FLAT_SURFACE = "flat_surface"
    LIPID_INTERFACE = "lipid_interface"
    ALLOSTERIC_CLEFT = "allosteric_cleft"


@dataclass
class SurfacePatch:
    """Represents a detected surface patch with geometric properties."""
    patch_id: int
    residue_indices: list[int]
    centroid: np.ndarray
    normal_vector: np.ndarray
    area_angstrom2: float
    depth_angstrom: float
    hydrophobicity_score: float
    topology: SurfaceTopology
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class ShallowSurfaceDetector:
    """
    Detects shallow surface binding sites suitable for metabolism-first
    peptide targeting.

    Uses geometric analysis of solvent-excluded surfaces combined with
    physicochemical property mapping to identify druggable shallow grooves
    that are often missed by deep-pocket-focused algorithms.

    Parameters
    ----------
    min_patch_area : float
        Minimum surface patch area in Å² to consider (default: 80).
    max_depth : float
        Maximum depth in Å to classify as shallow (default: 6.0).
    hydrophobicity_threshold : float
        Minimum hydrophobicity score for lipid-interface patches (default: 0.4).
    probe_radius : float
        Rolling probe radius for surface generation in Å (default: 1.4).
    """

    def __init__(
        self,
        min_patch_area: float = 80.0,
        max_depth: float = 6.0,
        hydrophobicity_threshold: float = 0.4,
        probe_radius: float = 1.4,
    ) -> None:
        self.min_patch_area = min_patch_area
        self.max_depth = max_depth
        self.hydrophobicity_threshold = hydrophobicity_threshold
        self.probe_radius = probe_radius

        # Kyte-Doolittle hydrophobicity scale (simplified)
        self._hydro_scale: dict[str, float] = {
            "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
            "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
            "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
            "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
        }

    def detect_shallow_sites(
        self,
        coords: np.ndarray,
        residue_names: list[str],
        residue_indices: list[int],
        sasa_values: Optional[np.ndarray] = None,
    ) -> list[SurfacePatch]:
        """
        Detect shallow surface binding sites from atomic coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Atomic coordinates, shape (N, 3), in Ångströms.
        residue_names : list[str]
            One-letter residue codes for each atom.
        residue_indices : list[int]
            Residue index for each atom.
        sasa_values : np.ndarray, optional
            Pre-computed SASA values per atom. If None, estimated.

        Returns
        -------
        list[SurfacePatch]
            Detected shallow surface patches ranked by druggability.
        """
        if len(coords) == 0:
            logger.warning("Empty coordinate array provided.")
            return []

        if len(coords) != len(residue_names):
            raise ValueError(
                f"coords length ({len(coords)}) != residue_names length "
                f"({len(residue_names)})"
            )

        coords = np.asarray(coords, dtype=np.float64)

        # Estimate SASA if not provided
        if sasa_values is None:
            sasa_values = self._estimate_sasa(coords)
        else:
            sasa_values = np.asarray(sasa_values, dtype=np.float64)

        # Identify surface-exposed atoms
        surface_mask = sasa_values > 0.1
        surface_coords = coords[surface_mask]
        surface_residues = [r for r, m in zip(residue_names, surface_mask) if m]
        surface_res_idx = [r for r, m in zip(residue_indices, surface_mask) if m]
        surface_sasa = sasa_values[surface_mask]

        if len(surface_coords) < 10:
            logger.info("Too few surface atoms for patch detection.")
            return []

        # Cluster surface atoms into patches
        patches = self._cluster_surface_atoms(
            surface_coords, surface_residues, surface_res_idx, surface_sasa
        )

        # Classify topology and filter shallow sites
        shallow_patches = []
        for patch in patches:
            patch.topology = self._classify_topology(patch)
            if self._is_shallow_target(patch):
                shallow_patches.append(patch)

        # Rank by combined score
        shallow_patches.sort(
            key=lambda p: self._druggability_score(p), reverse=True
        )

        logger.info(
            f"Detected {len(shallow_patches)} shallow surface sites "
            f"from {len(patches)} total patches."
        )
        return shallow_patches

    def _estimate_sasa(self, coords: np.ndarray) -> np.ndarray:
        """Rough SASA estimation using neighbor counting."""
        n = len(coords)
        sasa = np.zeros(n, dtype=np.float64)
        cutoff = 6.0  # Å

        for i in range(n):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            neighbors = np.sum((dists > 0.1) & (dists < cutoff))
            # Fewer neighbors → more exposed
            sasa[i] = max(0.0, 1.0 - neighbors / 20.0)

        return sasa

    def _cluster_surface_atoms(
        self,
        coords: np.ndarray,
        residues: list[str],
        res_indices: list[int],
        sasa: np.ndarray,
    ) -> list[SurfacePatch]:
        """Cluster surface atoms into contiguous patches."""
        from collections import defaultdict

        cluster_dist = 4.5  # Å — atoms within this distance are same patch
        n = len(coords)
        visited = [False] * n
        patches: list[SurfacePatch] = []
        patch_id = 0

        for start in range(n):
            if visited[start]:
                continue

            # BFS clustering
            queue = [start]
            visited[start] = True
            members: list[int] = []

            while queue:
                current = queue.pop(0)
                members.append(current)

                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(coords[current] - coords[j])
                        if dist < cluster_dist:
                            visited[j] = True
                            queue.append(j)

            if len(members) < 5:
                continue

            member_coords = coords[members]
            member_res = [residues[m] for m in members]
            member_res_idx = list({res_indices[m] for m in members})

            centroid = np.mean(member_coords, axis=0)
            area = len(members) * 5.0  # Rough estimate: ~5 Å² per surface atom

            # Estimate depth from centroid variance
            deviations = np.linalg.norm(member_coords - centroid, axis=1)
            depth = float(np.percentile(deviations, 90) * 2)

            # Hydrophobicity score
            hydro_scores = [
                self._hydro_scale.get(r, 0.0) for r in member_res
            ]
            hydro_score = float(np.mean([
                (s + 4.5) / 9.0 for s in hydro_scores  # Normalize to [0, 1]
            ]))

            # Normal vector via SVD
            centered = member_coords - centroid
            if len(centered) >= 3:
                _, _, vh = np.linalg.svd(centered)
                normal = vh[-1]
            else:
                normal = np.array([0.0, 0.0, 1.0])

            patch = SurfacePatch(
                patch_id=patch_id,
                residue_indices=sorted(member_res_idx),
                centroid=centroid,
                normal_vector=normal,
                area_angstrom2=area,
                depth_angstrom=depth,
                hydrophobicity_score=hydro_score,
                topology=SurfaceTopology.FLAT_SURFACE,
                metadata={"n_atoms": len(members)},
            )
            patches.append(patch)
            patch_id += 1

        return patches

    def _classify_topology(self, patch: SurfacePatch) -> SurfaceTopology:
        """Classify patch topology based on geometric features."""
        depth = patch.depth_angstrom
        area = patch.area_angstrom2
        hydro = patch.hydrophobicity_score

        if hydro > self.hydrophobicity_threshold and depth < 4.0:
            return SurfaceTopology.LIPID_INTERFACE
        if depth < 3.0 and area > 120:
            return SurfaceTopology.FLAT_SURFACE
        if depth < self.max_depth and area > self.min_patch_area:
            return SurfaceTopology.SHALLOW_GROOVE
        if depth >= self.max_depth:
            return SurfaceTopology.DEEP_POCKET
        return SurfaceTopology.ALLOSTERIC_CLEFT

    def _is_shallow_target(self, patch: SurfacePatch) -> bool:
        """Filter: only keep shallow, metabolism-relevant sites."""
        if patch.topology in (
            SurfaceTopology.SHALLOW_GROOVE,
            SurfaceTopology.LIPID_INTERFACE,
            SurfaceTopology.FLAT_SURFACE,
        ):
            return patch.area_angstrom2 >= self.min_patch_area
        return False

    def _druggability_score(self, patch: SurfacePatch) -> float:
        """
        Compute druggability score for shallow surface sites.
        Higher = more likely to bind a stabilizing peptide.
        """
        # Shallow grooves with moderate hydrophobicity score best
        depth_score = 1.0 - abs(patch.depth_angstrom - 4.0) / 6.0
        depth_score = max(0.0, depth_score)

        # Sweet spot for hydrophobicity (not too hydrophobic, not too polar)
        hydro_score = 1.0 - abs(patch.hydrophobicity_score - 0.55) / 0.55
        hydro_score = max(0.0, hydro_score)

        # Area contribution (larger is better up to a point)
        area_score = min(1.0, patch.area_angstrom2 / 300.0)

        # Topology bonus
        topo_bonus = {
            SurfaceTopology.SHALLOW_GROOVE: 0.3,
            SurfaceTopology.LIPID_INTERFACE: 0.2,
            SurfaceTopology.FLAT_SURFACE: 0.1,
            SurfaceTopology.ALLOSTERIC_CLEFT: 0.15,
            SurfaceTopology.DEEP_POCKET: 0.0,
        }.get(patch.topology, 0.0)

        combined = (
            0.35 * depth_score
            + 0.30 * hydro_score
            + 0.20 * area_score
            + topo_bonus
        )
        return float(np.clip(combined, 0.0, 1.0))
```

---

### `brownbiotech/arp_v3/agents/virtual_screen/lipid_groove_featurizer.py`

```python
"""
Lipid Groove Featurizer — Extracts features from lipid-binding grooves
and membrane-proximal sites for virtual screening of metabolism-first targets.

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LipidGrooveFeatures:
    """Feature vector for a lipid groove binding site."""
    groove_id: int
    # Geometric features
    length_angstrom: float
    width_angstrom: float
    depth_angstrom: float
    curvature: float
    # Physicochemical features
    hydrophobic_fraction: float
    aromatic_fraction: float
    charged_fraction: float
    polarity_score: float
    # Lipid-specific features
    lipid_headgroup_compatibility: float
    acyl_chain_depth: float
    membrane_proximity: float
    # Feature vector for ML models
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(11))
    metadata: dict = field(default_factory=dict)


class LipidGrooveFeaturizer:
    """
    Featurizes lipid-binding grooves for metabolism-first virtual screening.

    Translates geometric and physicochemical properties of lipid-interaction
    sites into fixed-length feature vectors suitable for ML scoring models.

    Parameters
    ----------
    n_bins : int
        Number of radial bins for curvature calculation (default: 8).
    membrane_thickness : float
        Assumed membrane thickness in Å for proximity scoring (default: 30.0).
    """

    AROMATIC_RESIDUES = {"F", "Y", "W", "H"}
    CHARGED_RESIDUES = {"K", "R", "D", "E"}
    POLAR_RESIDUES = {"S", "T", "N", "Q", "C"}
    HYDROPHOBIC_RESIDUES = {"I", "V", "L", "F", "C", "M", "A", "G", "W", "P"}

    def __init__(
        self,
        n_bins: int = 8,
        membrane_thickness: float = 30.0,
    ) -> None:
        self.n_bins = n_bins
        self.membrane_thickness = membrane_thickness

    def featurize(
        self,
        groove_coords: np.ndarray,
        residue_names: list[str],
        groove_id: int = 0,
        membrane_normal: Optional[np.ndarray] = None,
    ) -> LipidGrooveFeatures:
        """
        Extract features from a lipid groove.

        Parameters
        ----------
        groove_coords : np.ndarray
            Coordinates of groove atoms, shape (N, 3).
        residue_names : list[str]
            One-letter residue codes for each atom.
        groove_id : int
            Identifier for this groove.
        membrane_normal : np.ndarray, optional
            Unit vector normal to the membrane plane. If None,
            estimated from groove geometry.

        Returns
        -------
        LipidGrooveFeatures
            Computed feature set for the groove.
        """
        if len(groove_coords) < 3:
            raise ValueError(
                f"Need at least 3 atoms, got {len(groove_coords)}"
            )

        coords = np.asarray(groove_coords, dtype=np.float64)
        centroid = np.mean(coords, axis=0)

        # Geometric features
        length, width, depth = self._compute_groove_dimensions(coords)
        curvature = self._compute_curvature(coords, centroid)

        # Physicochemical features
        hydro_frac, arom_frac, charged_frac, polarity = (
            self._compute_residue_fractions(residue_names)
        )

        # Lipid-specific features
        headgroup_compat = self._compute_headgroup_compatibility(
            residue_names, coords, centroid
        )
        acyl_depth = self._estimate_acyl_chain_depth(coords, centroid)
        membrane_prox = self._compute_membrane_proximity(
            coords, centroid, membrane_normal
        )

        features = LipidGrooveFeatures(
            groove_id=groove_id,
            length_angstrom=length,
            width_angstrom=width,
            depth_angstrom=depth,
            curvature=curvature,
            hydrophobic_fraction=hydro_frac,
            aromatic_fraction=arom_frac,
            charged_fraction=charged_frac,
            polarity_score=polarity,
            lipid_headgroup_compatibility=headgroup_compat,
            acyl_chain_depth=acyl_depth,
            membrane_proximity=membrane_prox,
        )

        # Build fixed-length vector
        features.feature_vector = np.array([
            length,
            width,
            depth,
            curvature,
            hydro_frac,
            arom_frac,
            charged_frac,
            polarity,
            headgroup_compat,
            acyl_depth,
            membrane_prox,
        ], dtype=np.float64)

        return features

    def featurize_batch(
        self,
        groove_data: list[tuple[np.ndarray, list[str]]],
    ) -> list[LipidGrooveFeatures]:
        """Featurize multiple grooves at once."""
        results = []
        for idx, (coords, residues) in enumerate(groove_data):
            try:
                feat = self.featurize(coords, residues, groove_id=idx)
                results.append(feat)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(
                    f"Failed to featurize groove {idx}: {e}"
                )
        return results

    def _compute_groove_dimensions(
        self, coords: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute length, width, and depth of the groove via PCA."""
        centered = coords - np.mean(coords, axis=0)
        _, s, vh = np.linalg.svd(centered, full_matrices=False)

        # Principal axes define length, width, depth
        lengths = s  # Singular values ≈ spread along each axis
        if len(lengths) < 3:
            lengths = np.pad(lengths, (0, 3 - len(lengths)))

        # Sort descending
        lengths = np.sort(lengths)[::-1]
        return float(lengths[0]), float(lengths[1]), float(lengths[2])

    def _compute_curvature(
        self, coords: np.ndarray, centroid: np.ndarray
    ) -> float:
        """
        Compute curvature as normalized variance of radial distances.
        Higher = more curved groove.
        """
        radial_dists = np.linalg.norm(coords - centroid, axis=1)
        if len(radial_dists) < 2:
            return 0.0

        mean_r = np.mean(radial_dists)
        if mean_r < 1e-6:
            return 0.0

        # Coefficient of variation of radial distances
        cv = np.std(radial_dists) / mean_r
        return float(np.clip(cv, 0.0, 2.0))

    def _compute_residue_fractions(
        self, residue_names: list[str]
    ) -> tuple[float, float, float, float]:
        """Compute physicochemical composition fractions."""
        n = len(residue_names)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0

        unique_residues = set(residue_names)

        hydro_count = sum(
            1 for r in residue_names if r in self.HYDROPHOBIC_RESIDUES
        )
        arom_count = sum(
            1 for r in residue_names if r in self.AROMATIC_RESIDUES
        )
        charged_count = sum(
            1 for r in residue_names if r in self.CHARGED_RESIDUES
        )
        polar_count = sum(
            1 for r in residue_names if r in self.POLAR_RESIDUES
        )

        return (
            hydro_count / n,
            arom_count / n,
            charged_count / n,
            polar_count / n,
        )

    def _compute_headgroup_compatibility(
        self,
        residue_names: list[str],
        coords: np.ndarray,
        centroid: np.ndarray,
    ) -> float:
        """
        Score how compatible the groove is with lipid headgroup binding.
        Headgroups prefer positive/polar residues at the groove entrance.
        """
        dists = np.linalg.norm(coords - centroid, axis=1)
        median_dist = np.median(dists) if len(dists) > 0 else 0.0

        # Atoms near the entrance (farther from centroid)
        entrance_mask = dists > median_dist
        entrance_residues = [
            r for r, m in zip(residue_names, entrance_mask) if m
        ]

        if not entrance_residues:
            return 0.5  # Neutral

        positive = sum(1 for r in entrance_residues if r in {"K", "R", "H"})
        polar = sum(
            1 for r in entrance_residues
            if r in self.POLAR_RESIDUES or r in {"K", "R", "H"}
        )
        score = (positive * 1.0 + polar * 0.5) / len(entrance_residues)
        return float(np.clip(score, 0.0, 1.0))

    def _estimate_acyl_chain_depth(
        self, coords: np.ndarray, centroid: np.ndarray
    ) -> float:
        """
        Estimate how deep acyl chains can penetrate into the groove.
        Based on the hydrophobic core depth.
        """
        dists = np.linalg.norm(coords - centroid, axis=1)
        # Depth is approximated by the range of distances
        depth = float(np.max(dists) - np.min(dists)) if len(dists) > 1 else 0.0
        return min(depth, 20.0)  # Cap at 20 Å

    def _compute_membrane_proximity(
        self,
        coords: np.ndarray,
        centroid: np.ndarray,
        membrane_normal: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score proximity to a membrane plane (0=far, 1=at membrane).
        """
        if membrane_normal is None:
            # Estimate from groove orientation (assume longest axis is parallel)
            centered = coords - centroid
            if len(centered) >= 3:
                _, _, vh = np.linalg.svd(centered)
                membrane_normal = vh[0]  # Longest axis
            else:
                membrane_normal = np.array([0.0, 0.0, 1.0])

        # Project centroid onto normal
        projection = abs(np.dot(centroid, membrane_normal))

        # Normalize: closer to 0 = more centered (higher proximity)
        proximity = 1.0 - min(projection / self.membrane_thickness, 1.0)
        return float(np.clip(proximity, 0.0, 1.0))
```

---

### `brownbiotech/arp_v3/agents/virtual_screen/canonical_vs_noncanonical.py`

```python
"""
Canonical vs Noncanonical Binding Mode Classifier.

Distinguishes between canonical (deep-pocket, lock-and-key) and
noncanonical (shallow-surface, induced-fit, metabolism-first) binding modes
to route virtual screening results appropriately.

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BindingMode(Enum):
    """Classification of peptide-protein binding mode."""
    CANONICAL_POCKET = "canonical_pocket"
    SHALLOW_SURFACE = "shallow_surface"
    LIPID_GROOVE = "lipid_groove"
    INDUCED_FIT = "induced_fit"
    ALLOSTERIC_SURFACE = "allosteric_surface"
    MEMBRANE_PROXIMAL = "membrane_proximal"


@dataclass
class BindingModePrediction:
    """Result of binding mode classification."""
    mode: BindingMode
    confidence: float
    is_noncanonical: bool
    scores: dict[str, float]
    recommendation: str


class CanonicalNoncanonicalClassifier:
    """
    Classifies binding interactions as canonical or noncanonical.

    Uses a rule-based scoring system with configurable thresholds to
    determine whether a peptide-target interaction follows canonical
    deep-pocket binding or a noncanonical shallow-surface mechanism.

    Parameters
    ----------
    depth_threshold : float
        Pocket depth threshold in Å (default: 6.0).
    buried_sasa_threshold : float
        Minimum buried SASA fraction for canonical (default: 0.6).
    noncanonical_score_threshold : float
        Score above which interaction is classified noncanonical (default: 0.55).
    """

    def __init__(
        self,
        depth_threshold: float = 6.0,
        buried_sasa_threshold: float = 0.6,
        noncanonical_score_threshold: float = 0.55,
    ) -> None:
        self.depth_threshold = depth_threshold
        self.buried_sasa_threshold = buried_sasa_threshold
        self.nc_threshold = noncanonical_score_threshold

    def classify(
        self,
        pocket_depth: float,
        buried_sasa_fraction: float,
        interface_hydrophobicity: float,
        n_hydrogen_bonds: int,
        n_pi_interactions: int,
        conformational_rmsd: float = 0.0,
        membrane_proximity: float = 0.0,
        groove_curvature: float = 0.0,
    ) -> BindingModePrediction:
        """
        Classify a binding mode from structural descriptors.

        Parameters
        ----------
        pocket_depth : float
            Depth of the binding pocket in Å.
        buried_sasa_fraction : float
            Fraction of peptide SASA buried upon binding [0, 1].
        interface_hydrophobicity : float
            Hydrophobicity of the interface [0, 1].
        n_hydrogen_bonds : int
            Number of hydrogen bonds at the interface.
        n_pi_interactions : int
            Number of pi-stacking/cation-pi interactions.
        conformational_rmsd : float
            RMSD of peptide conformational change upon binding (Å).
        membrane_proximity : float
            Membrane proximity score [0, 1].
        groove_curvature : float
            Curvature of the binding groove [0, 2].

        Returns
        -------
        BindingModePrediction
            Classification result with confidence and recommendation.
        """
        scores = self._compute_scores(
            pocket_depth=pocket_depth,
            buried_sasa_fraction=buried_sasa_fraction,
            interface_hydrophobicity=interface_hydrophobicity,
            n_hydrogen_bonds=n_hydrogen_bonds,
            n_pi_interactions=n_pi_interactions,
            conformational_rmsd=conformational_rmsd,
            membrane_proximity=membrane_proximity,
            groove_curvature=groove_curvature,
        )

        # Determine mode
        mode, confidence = self._determine_mode(scores)

        is_noncanonical = mode in (
            BindingMode.SHALLOW_SURFACE,
            BindingMode.LIPID_GROOVE,
            BindingMode.INDUCED_FIT,
            BindingMode.ALLOSTERIC_SURFACE,
            BindingMode.MEMBRANE_PROXIMAL,
        )

        recommendation = self._generate_recommendation(mode, confidence, scores)

        return BindingModePrediction(
            mode=mode,
            confidence=confidence,
            is_noncanonical=is_noncanonical,
            scores=scores,
            recommendation=recommendation,
        )

    def _compute_scores(
        self,
        pocket_depth: float,
        buried_sasa_fraction: float,
        interface_hydrophobicity: float,
        n_hydrogen_bonds: int,
        n_pi_interactions: int,
        conformational_rmsd: float,
        membrane_proximity: float,
        groove_curvature: float,
    ) -> dict[str, float]:
        """Compute individual mode scores."""
        # Shallow surface score
        shallow_score = (
            (1.0 - min(pocket_depth / self.depth_threshold, 1.0)) * 0.5
            + (1.0 - min(buried_sasa_fraction / self.buried_sasa_threshold, 1.0)) * 0.3
            + min(groove_curvature / 1.5, 1.0) * 0.2
        )

        # Lipid groove score
        lipid_score = (
            min(interface_hydrophobicity, 1.0) * 0.4
            + min(membrane_proximity, 1.0) * 0.35
            + (1.0 - min(pocket_depth / 8.0, 1.0)) * 0.25
        )

        # Induced fit score
        induced_score = (
            min(conformational_rmsd / 3.0, 1.0) * 0.5
            + min(n_hydrogen_bonds / 10.0, 1.0) * 0.3
            + min(n_pi_interactions / 5.0, 1.0) * 0.2
        )

        # Canonical pocket score
        canonical_score = (
            min(pocket_depth / 10.0, 1.0) * 0.4
            + min(buried_sasa_fraction, 1.0) * 0.35
            + min(n_hydrogen_bonds / 8.0, 1.0) * 0.25
        )

        # Allosteric surface score
        allosteric_score = (
            (1.0 - min(buried_sasa_fraction, 1.0)) * 0.4
            + min(groove_curvature / 1.0, 1.0) * 0.3
            + (1.0 - min(pocket_depth / 8.0, 1.0)) * 0.3
        )

        # Membrane proximal score
        membrane_score = (
            min(membrane_proximity, 1.0) * 0.5
            + min(interface_hydrophobicity, 1.0) * 0.3
            + (1.0 - min(pocket_depth / 8.0, 1.0)) * 0.2
        )

        return {
            "shallow_surface": shallow_score,
            "lipid_groove": lipid_score,
            "induced_fit": induced_score,
            "canonical_pocket": canonical_score,
            "allosteric_surface": allosteric_score,
            "membrane_proximal": membrane_score,
        }

    def _determine_mode(
        self, scores: dict[str, float]
    ) -> tuple[BindingMode, float]:
        """Select the highest-scoring mode."""
        best_mode_name = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_mode_name]

        # Confidence: margin over second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.5 + margin * 2.0, 1.0)
        else:
            confidence = 0.5

        mode_map = {
            "shallow_surface": BindingMode.SHALLOW_SURFACE,
            "lipid_groove": BindingMode.LIPID_GROOVE,
            "induced_fit": BindingMode.INDUCED_FIT,
            "canonical_pocket": BindingMode.CANONICAL_POCKET,
            "allosteric_surface": BindingMode.ALLOSTERIC_SURFACE,
            "membrane_proximal": BindingMode.MEMBRANE_PROXIMAL,
        }

        return mode_map[best_mode_name], float(confidence)

    def _generate_recommendation(
        self,
        mode: BindingMode,
        confidence: float,
        scores: dict[str, float],
    ) -> str:
        """Generate actionable recommendation based on classification."""
        if mode == BindingMode.CANONICAL_POCKET:
            return (
                "Canonical pocket binding detected. Standard scoring pipeline "
                "is appropriate. Consider noncanonical design only if metabolic "
                f"stability is a concern (confidence: {confidence:.2f})."
            )

        if mode == BindingMode.SHALLOW_SURFACE:
            return (
                "Shallow surface binding detected. Route to noncanonical "
                "peptide design with enhanced surface complementarity. "
                "Consider macrocyclization for entropic gain. "
                f"(confidence: {confidence:.2f})"
            )

        if mode == BindingMode.LIPID_GROOVE:
            return (
                "Lipid groove binding detected. Prioritize hydrophobic "
                "noncanonical amino acids (e.g., Nle, Cha, Tle). "
                "Evaluate membrane-mimetic assay conditions. "
                f"(confidence: {confidence:.2f})"
            )

        if mode == BindingMode.INDUCED_FIT:
            return (
                "Induced fit mechanism detected. Consider flexible docking "
                "and conformational ensemble screening. Noncanonical residues "
                "with restricted conformations may improve binding kinetics. "
                f"(confidence: {confidence:.2f})"
            )

        if mode == BindingMode.MEMBRANE_PROXIMAL:
            return (
                "Membrane-proximal binding detected. Design peptides with "
                "amphipathic character. Consider lipidation for membrane "
                "anchoring. Use membrane-aware scoring functions. "
                f"(confidence: {confidence:.2f})"
            )

        if mode == BindingMode.ALLOSTERIC_SURFACE:
            return (
                "Allosteric surface binding detected. Validate allosteric "
                "mechanism experimentally. Noncanonical design may improve "
                "selectivity over orthosteric sites. "
                f"(confidence: {confidence:.2f})"
            )

        return "Unable to generate recommendation."
```

---

## 2. Enhanced Design Agent

### `brownbiotech/arp_v3/agents/design/noncanonical_generator.py`

```python
"""
Noncanonical Amino Acid Peptide Generator.

Generates peptide sequences incorporating noncanonical amino acids (ncAAs)
optimized for shallow-surface and metabolism-first targeting.

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NcAACategory(Enum):
    """Categories of noncanonical amino acids."""
    HYDROPHOBIC = "hydrophobic"
    AROMATIC = "aromatic"
    CATIONIC = "cationic"
    ANIONIC = "anionic"
    CONSTRAINED = "constrained"
    FLUORINATED = "fluorinated"
    METABOLIC_STABLE = "metabolic_stable"


@dataclass
class NoncanonicalAA:
    """Representation of a noncanonical amino acid."""
    code: str  # 3-letter code
    name: str
    category: NcAACategory
    hydrophobicity: float  # Normalized [0, 1]
    volume_angstrom3: float
    metabolic_stability: float  # [0, 1], 1 = fully stable
    is_proteogenic: bool = False
    smarts: str = ""
    notes: str = ""


@dataclass
class GeneratedPeptide:
    """A generated peptide sequence with metadata."""
    sequence: str  # One-letter codes, ncAAs use lowercase
    full_sequence: str  # With ncAA 3-letter codes
    length: int
    n_canonical: int
    n_noncanonical: int
    ncAA_fraction: float
    hydrophobicity: float
    metabolic_stability_score: float
    surface_complementarity_score: float
    total_score: float
    mutations: list[tuple[int, str, str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Noncanonical amino acid library
NC_AA_LIBRARY: dict[str, NoncanonicalAA] = {
    "nle": NoncanonicalAA("Nle", "Norleucine", NcAACategory.HYDROPHOBIC, 0.82, 124.0, 0.85),
    "cha": NoncanonicalAA("Cha", "Cyclohexylalanine", NcAACategory.HYDROPHOBIC, 0.90, 153.0, 0.92),
    "tle": NoncanonicalAA("Tle", "Tert-leucine", NcAACategory.HYDROPHOBIC, 0.88, 140.0, 0.95),
    "phe(4f)": NoncanonicalAA("Phe(4-F)", "4-Fluorophenylalanine", NcAACategory.FLUORINATED, 0.72, 135.0, 0.88),
    "phe(3f)": NoncanonicalAA("Phe(3-F)", "3-Fluorophenylalanine", NcAACategory.FLUORINATED, 0.70, 135.0, 0.87),
    "trp(5f)": NoncanonicalAA("Trp(5-F)", "5-Fluorotryptophan", NcAACategory.FLUORINATED, 0.65, 163.0, 0.86),
    "dap": NoncanonicalAA("Dap", "2,3-Diaminopropionic acid", NcAACategory.CATIONIC, 0.10, 75.0, 0.70),
    "dab": NoncanonicalAA("Dab", "2,4-Diaminobutyric acid", NcAACategory.CATIONIC, 0.15, 91.0, 0.75),
    "orn": NoncanonicalAA("Orn", "Ornithine", NcAACategory.CATIONIC, 0.12, 105.0, 0.72),
    "pen": NoncanonicalAA("Pen", "Penicillamine", NcAACategory.HYDROPHOBIC, 0.75, 117.0, 0.80),
    "acm": NoncanonicalAA("AcM", "N-Acetyl-methyl", NcAACategory.METABOLIC_STABLE, 0.50, 100.0, 0.90),
    "mea": NoncanonicalAA("MeA", "N-Methyl-alanine", NcAACategory.METABOLIC_STABLE, 0.45, 92.0, 0.93),
    "mep": NoncanonicalAA("MeP", "N-Methyl-proline", NcAACategory.CONSTRAINED, 0.40, 109.0, 0.94),
    "pip": NoncanonicalAA("Pip", "Pipecolic acid", NcAACategory.CONSTRAINED, 0.42, 108.0, 0.91),
    "tic": NoncanonicalAA("Tic", "1,2,3,4-Tetrahydroisoquinoline-3-carboxylic acid", NcAACategory.CONSTRAINED, 0.68, 142.0, 0.89),
    "aba": NoncanonicalAA("Abu", "Alpha-aminobutyric acid", NcAACategory.HYDROPHOBIC, 0.60, 92.0, 0.78),
    "hph": NoncanonicalAA("Hph", "Homophenylalanine", NcAACategory.HYDROPHOBIC, 0.78, 140.0, 0.82),
}

# Mapping from canonical to recommended ncAA substitutions
CANONICAL_TO_NC: dict[str, list[str]] = {
    "L": ["nle", "tle", "cha"],
    "I": ["nle", "tle", "cha"],
    "V": ["tle", "aba"],
    "F": ["phe(4f)", "phe(3f)", "hph", "cha"],
    "W": ["trp(5f)"],
    "A": ["mea", "aba", "tle"],
    "P": ["pip", "mep"],
    "K": ["orn", "dab", "dap"],
    "R": ["dab", "orn"],
    "C": ["pen"],
}


class NoncanonicalGenerator:
    """
    Generates peptide variants with noncanonical amino acids.

    Supports targeted substitution (replacing specific residues) and
    diversity-driven generation for virtual screening libraries.

    Parameters
    ----------
    max_ncAA_fraction : float
        Maximum fraction of noncanonical residues (default: 0.5).
    min_ncAA_fraction : float
        Minimum fraction for metabolism-first designs (default: 0.2).
    metabolic_stability_weight : float
        Weight for metabolic stability in scoring (default: 0.4).
    surface_complementarity_weight : float
        Weight for surface complementarity (default: 0.35).
    hydrophobicity_weight : float
        Weight for hydrophobicity matching (default: 0.25).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_ncAA_fraction: float = 0.5,
        min_ncAA_fraction: float = 0.2,
        metabolic_stability_weight: float = 0.4,
        surface_complementarity_weight: float = 0.35,
        hydrophobicity_weight: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        self.max_ncAA = max_ncAA_fraction
        self.min_ncAA = min_ncAA_fraction
        self.w_metab = metabolic_stability_weight
        self.w_surf = surface_complementarity_weight
        self.w_hydro = hydrophobicity_weight

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_variants(
        self,
        parent_sequence: str,
        n_variants: int = 10,
        target_hydrophobicity: Optional[float] = None,
        allowed_categories: Optional[set[NcAACategory]] = None,
        protected_positions: Optional[set[int]] = None,
    ) -> list[GeneratedPeptide]:
        """
        Generate noncanonical variants of a parent peptide.

        Parameters
        ----------
        parent_sequence : str
            Parent peptide sequence (one-letter codes, uppercase).
        n_variants : int
            Number of variants to generate.
        target_hydrophobicity : float, optional
            Target hydrophobicity [0, 1]. If None, match parent.
        allowed_categories : set[NcAACategory], optional
            Allowed ncAA categories. If None, all allowed.
        protected_positions : set[int], optional
            Positions that cannot be mutated.

        Returns
        -------
        list[GeneratedPeptide]
            Generated variants sorted by total score.
        """
        if not parent_sequence:
            raise ValueError("Parent sequence cannot be empty")

        protected = protected_positions or set()
        allowed = allowed_categories or set(NcAACategory)

        # Filter ncAA library by allowed categories
        eligible_ncAAs = {
            code: ncAA for code, ncAA in NC_AA_LIBRARY.items()
            if ncAA.category in allowed
        }

        if not eligible_ncAAs:
            logger.warning("No eligible ncAAs for given categories.")
            return []

        parent_hydro = self._sequence_hydrophobicity(parent_sequence)
        if target_hydrophobicity is None:
            target_hydrophobicity = parent_hydro

        # Identify mutable positions
        mutable_positions = [
            i for i, aa in enumerate(parent_sequence)
            if i not in protected and aa in CANONICAL_TO_NC
        ]

        if not mutable_positions:
            logger.info("No mutable positions found.")
            return []

        variants: list[GeneratedPeptide] = []

        for _ in range(n_variants):
            peptide = self._generate_single_variant(
                parent_sequence=parent_sequence,
                mutable_positions=mutable_positions,
                eligible_ncAAs=eligible_ncAAs,
                target_hydrophobicity=target_hydrophobicity,
            )
            if peptide is not None:
                variants.append(peptide)

        variants.sort(key=lambda p: p.total_score, reverse=True)
        logger.info(
            f"Generated {len(variants)} ncAA variants from "
            f"'{parent_sequence}' (parent hydro: {parent_hydro:.2f})"
        )
        return variants

    def _generate_single_variant(
        self,
        parent_sequence: str,
        mutable_positions: list[int],
        eligible_ncAAs: dict[str, NoncanonicalAA],
        target_hydrophobicity: float,
    ) -> Optional[GeneratedPeptide]:
        """Generate a single variant."""
        seq_len = len(parent_sequence)
        max_mutations = int(seq_len * self.max_ncAA)
        min_mutations = max(1, int(seq_len * self.min_ncAA))

        n_mutations = random.randint(min_mutations, max_mutations)
        positions_to_mutate = random.sample(
            mutable_positions, min(n_mutations, len(mutable_positions))
        )

        sequence_list = list(parent_sequence)
        mutations: list[tuple[int, str, str]] = []
        ncAA_codes: list[str] = []

        for pos in positions_to_mutate:
            original_aa = sequence_list[pos]
            candidates = CANONICAL_TO_NC.get(original_aa, [])

            # Filter to eligible
            candidates = [c for c in candidates if c in eligible_ncAAs]
            if not candidates:
                continue

            # Select with bias toward hydrophobicity target
            current_hydro = self._sequence_hydrophobicity(
                "".join(sequence_list)
            )
            hydro_diff = target_hydrophobicity - current_hydro

            if hydro_diff > 0:
                # Need more hydrophobic
                candidates.sort(
                    key=lambda c: eligible_ncAAs[c].hydrophobicity, reverse=True
                )
            else:
                # Need less hydrophobic
                candidates.sort(
                    key=lambda c: eligible_ncAAs[c].hydrophobicity
                )

            # Add some randomness (pick from top 3)
            top_n = min(3, len(candidates))
            chosen_code = random.choice(candidates[:top_n])
            chosen_ncAA = eligible_ncAAs[chosen_code]

            # Use lowercase letter for ncAA in sequence
            # Map first letter of code as placeholder
            ncAA_placeholder = chosen_code[0].lower()
            sequence_list[pos] = ncAA_placeholder
            ncAA_codes.append(chosen_code)
            mutations.append((pos, original_aa, chosen_code))

        if not mutations:
            return None

        final_seq = "".join(sequence_list)
        full_seq = self._build_full_sequence(
            parent_sequence, mutations
        )

        # Compute scores
        n_canonical = seq_len - len(mutations)
        n_noncanonical = len(mutations)
        nc_frac = n_noncanonical / seq_len
        hydro = self._sequence_hydrophobicity(final_seq)
        metab_score = self._compute_metabolic_stability(mutations)
        surf_score = self._compute_surface_complementarity(mutations)
        total = (
            self.w_metab * metab_score
            + self.w_surf * surf_score
            + self.w_hydro * (1.0 - abs(hydro - target_hydrophobicity))
        )

        return GeneratedPeptide(
            sequence=final_seq,
            full_sequence=full_seq,
            length=seq_len,
            n_canonical=n_canonical,
            n_noncanonical=n_noncanonical,
            ncAA_fraction=nc_frac,
            hydrophobicity=hydro,
            metabolic_stability_score=metab_score,
            surface_complementarity_score=surf_score,
            total_score=total,
            mutations=mutations,
        )

    def _build_full_sequence(
        self,
        parent: str,
        mutations: list[tuple[int, str, str]],
    ) -> str:
        """Build full sequence string with ncAA 3-letter codes."""
        parts = []
        for i, aa in enumerate(parent):
            # Check if this position was mutated
            mut = next((m for m in mutations if m[0] == i), None)
            if mut:
                parts.append(f"[{mut[2]}]")
            else:
                parts.append(aa)
        return "".join(parts)

    def _sequence_hydrophobicity(self, sequence: str) -> float:
        """Estimate sequence hydrophobicity [0, 1]."""
        if not sequence:
            return 0.5

        hydro_scale = {
            "I": 0.90, "V": 0.82, "L": 0.85, "F": 0.72, "C": 0.65,
            "M": 0.71, "A": 0.60, "G": 0.45, "T": 0.42, "S": 0.40,
            "W": 0.55, "Y": 0.48, "P": 0.38, "H": 0.30, "E": 0.22,
            "Q": 0.25, "D": 0.20, "N": 0.28, "K": 0.18, "R": 0.15,
        }

        total = 0.0
        for aa in sequence:
            key = aa.upper()
            if key in hydro_scale:
                total += hydro_scale[key]
            elif aa.islower():
                # ncAA placeholder — look up by code
                for code, ncAA in NC_AA_LIBRARY.items():
                    if code[0].lower() == aa:
                        total += ncAA.hydrophobicity
                        break
                else:
                    total += 0.5
            else:
                total += 0.5

        return total / len(sequence)

    def _compute_metabolic_stability(
        self, mutations: list[tuple[int, str, str]]
    ) -> float:
        """Compute metabolic stability improvement score."""
        if not mutations:
            return 0.5

        stabilities = []
        for _, _, nc_code in mutations:
            ncAA = NC_AA_LIBRARY.get(nc_code)
            if ncAA:
                stabilities.append(ncAA.metabolic_stability)

        return float(np.mean(stabilities)) if stabilities else 0.5

    def _compute_surface_complementarity(
        self, mutations: list[tuple[int, str, str]]
    ) -> float:
        """
        Estimate surface complementarity contribution.
        Constrained and hydrophobic ncAAs improve shallow-surface binding.
        """
        if not mutations:
            return 0.5

        scores = []
        for _, _, nc_code in mutations:
            ncAA = NC_AA_LIBRARY.get(nc_code)
            if ncAA:
                # Constrained residues score higher for surface binding
                if ncAA.category in (
                    NcAACategory.CONSTRAINED,
                    NcAACategory.HYDROPHOBIC,
                ):
                    scores.append(0.7 + 0.3 * ncAA.hydrophobicity)
                elif ncAA.category == NcAACategory.FLUORINATED:
                    scores.append(0.75)
                else:
                    scores.append(0.5)

        return float(np.mean(scores)) if scores else 0.5
```

---

### `brownbiotech/arp_v3/agents/design/metabolic_constraint_solver.py`

```python
"""
Metabolic Constraint Solver — Ensures designed peptides meet metabolic
stability requirements for metabolism-first targeting.

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProteaseType(Enum):
    """Known protease cleavage specificities."""
    TRYPSIN = "trypsin"        # Cuts after K, R
    CHYMOTRYPSIN = "chymotrypsin"  # Cuts after F, Y, W, L
    ELASTASE = "elastase"      # Cuts after A, V, G, S
    CARBOXYPEPTIDASE = "carboxypeptidase"  # C-terminal exopeptidase
    AMINOPEPTIDASE = "aminopeptidase"      # N-terminal exopeptidase
    THERMOLYSIN = "thermolysin"  # Cuts before hydrophobic residues


@dataclass
class CleavageSite:
    """Identified protease cleavage vulnerability."""
    position: int
    protease: ProteaseType
    residue: str
    severity: float  # 0-1, higher = more likely to be cleaved
    context: str  # Flanking sequence context


@dataclass
class MetabolicProfile:
    """Complete metabolic stability profile of a peptide."""
    sequence: str
    n_cleavage_sites: int
    cleavage_sites: list[CleavageSite]
    overall_stability_score: float  # [0, 1]
    half_life_estimate_hours: float
    n_terminal_stability: float
    c_terminal_stability: float
    internal_stability: float
    recommendations: list[str]


class MetabolicConstraintSolver:
    """
    Analyzes and improves metabolic stability of peptide designs.

    Identifies protease cleavage sites and suggests noncanonical amino acid
    substitutions to improve stability while maintaining binding properties.

    Parameters
    ----------
    min_stability_score : float
        Minimum acceptable stability score (default: 0.6).
    target_half_life_hours : float
        Target half-life in hours for in vivo stability (default: 4.0).
    protease_weights : dict[str, float], optional
        Custom weights for different proteases.
    """

    # Protease recognition patterns
    PROTEASE_SPECIFICITY: dict[ProteaseType, set[str]] = {
        ProteaseType.TRYPSIN: {"K", "R"},
        ProteaseType.CHYMOTRYPSIN: {"F", "Y", "W", "L"},
        ProteaseType.ELASTASE: {"A", "V", "G", "S"},
        ProteaseType.THERMOLYSIN: {"I", "V", "L", "A", "F", "M"},
    }

    # Context-dependent severity modifiers
    # (preceding residue -> severity multiplier)
    CONTEXT_MODIFIERS: dict[str, float] = {
        "P": 0.1,   # Proline before cleavage site strongly reduces cleavage
        "G": 1.3,   # Glycine before increases flexibility/cleavage
        "D": 0.7,   # Negative charge can reduce some proteases
        "E": 0.7,
    }

    def __init__(
        self,
        min_stability_score: float = 0.6,
        target_half_life_hours: float = 4.0,
        protease_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.min_stability = min_stability_score
        self.target_half_life = target_half_life_hours
        self.protease_weights = protease_weights or {
            ProteaseType.TRYPSIN.value: 1.0,
            ProteaseType.CHYMOTRYPSIN.value: 0.9,
            ProteaseType.ELASTASE.value: 0.7,
            ProteaseType.AMINOPEPTIDASE.value: 0.8,
            ProteaseType.CARBOXYPEPTIDASE.value: 0.8,
            ProteaseType.THERMOLYSIN.value: 0.6,
        }

    def analyze(self, sequence: str) -> MetabolicProfile:
        """
        Analyze metabolic stability of a peptide sequence.

        Parameters
        ----------
        sequence : str
            Peptide sequence (one-letter codes).

        Returns
        -------
        MetabolicProfile
            Complete metabolic stability analysis.
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        cleavage_sites = self._find_cleavage_sites(sequence)

        n_term_stab = self._assess_n_terminal(sequence)
        c_term_stab = self._assess_c_terminal(sequence)
        internal_stab = self._assess_internal_stability(
            sequence, cleavage_sites
        )

        # Overall stability: weighted combination
        overall = (
            0.20 * n_term_stab
            + 0.20 * c_term_stab
            + 0.60 * internal_stab
        )

        # Estimate half-life (empirical model)
        half_life = self._estimate_half_life(overall, len(sequence))

        recommendations = self._generate_recommendations(
            sequence, cleavage_sites, overall
        )

        return MetabolicProfile(
            sequence=sequence,
            n_cleavage_sites=len(cleavage_sites),
            cleavage_sites=cleavage_sites,
            overall_stability_score=overall,
            half_life_estimate_hours=half_life,
            n_terminal_stability=n_term_stab,
            c_terminal_stability=c_term_stab,
            internal_stability=internal_stab,
            recommendations=recommendations,
        )

    def suggest_stabilizing_mutations(
        self,
        profile: MetabolicProfile,
        max_mutations: int = 3,
        preserve_positions: Optional[set[int]] = None,
    ) -> list[dict[str, object]]:
        """
        Suggest mutations to improve metabolic stability.

        Parameters
        ----------
        profile : MetabolicProfile
            Analysis result from analyze().
        max_mutations : int
            Maximum number of mutations to suggest.
        preserve_positions : set[int], optional
            Positions that should not be mutated.

        Returns
        -------
        list[dict]
            Suggested mutations with rationale.
        """
        preserve = preserve_positions or set()
        suggestions: list[dict[str, object]] = []

        # Sort cleavage sites by severity
        sorted_sites = sorted(
            profile.cleavage_sites, key=lambda s: s.severity, reverse=True
        )

        # Stabilizing substitutions
        stabilizing_subs: dict[str, str] = {
            "K": "R",   # Arg is slightly more stable than Lys
            "R": "K",   # Sometimes reverse is better
            "F": "Y",   # Tyr slightly less chymotrypsin-susceptible
            "W": "F",   # Trp → Phe reduces chymotrypsin
            "L": "I",   # Ile slightly less susceptible
            "A": "Tle", # Tert-leucine replaces Ala
            "G": "A",   # Remove flexibility hotspot
        }

        for site in sorted_sites:
            if len(suggestions) >= max_mutations:
                break

            pos = site.position
            if pos in preserve:
                continue

            original = site.residue
            if original not in stabilizing_subs:
                continue

            sub = stabilizing_subs[original]

            # Check if substitution introduces new cleavage
            new_sites = self._find_cleavage_sites_at(
                profile.sequence, pos, sub
            )
            new_severity = sum(s.severity for s in new_sites)

            if new_severity < site.severity:
                suggestions.append({
                    "position": pos,
                    "original": original,
                    "substitution": sub,
                    "protease": site.protease.value,
                    "severity_reduction": site.severity - new_severity,
                    "rationale": (
                        f"Replace {original}{pos+1} with {sub} to reduce "
                        f"{site.protease.value} cleavage (severity: "
                        f"{site.severity:.2f} → {new_severity:.2f})"
                    ),
                })

        return suggestions

    def _find_cleavage_sites(self, sequence: str) -> list[CleavageSite]:
        """Find all protease cleavage sites in sequence."""
        sites: list[CleavageSite] = []

        for protease, target_residues in self.PROTEASE_SPECIFICITY.items():
            weight = self.protease_weights.get(protease.value, 0.5)

            for i, aa in enumerate(sequence):
                if aa.upper() in target_residues:
                    severity = weight

                    # Context modifier
                    if i > 0:
                        prev_aa = sequence[i - 1].upper()
                        modifier = self.CONTEXT_MODIFIERS.get(prev_aa, 1.0)
                        severity *= modifier

                    # N-terminal residues are more exposed to aminopeptidases
                    if i < 3 and protease == ProteaseType.TRYPSIN:
                        severity *= 1.3

                    # C-terminal residues more exposed to carboxypeptidases
                    if i >= len(sequence) - 3 and protease in (
                        ProteaseType.CHYMOTRYPSIN,
                        ProteaseType.TRYPSIN,
                    ):
                        severity *= 1.2

                    # Proline after cleavage site blocks many proteases
                    if i < len(sequence) - 1 and sequence[i + 1].upper() == "P":
                        severity *= 0.05

                    if severity > 0.1:
                        context_start = max(0, i - 2)
                        context_end = min(len(sequence), i + 3)
                        context = sequence[context_start:context_end]

                        sites.append(CleavageSite(
                            position=i,
                            protease=protease,
                            residue=aa.upper(),
                            severity=float(np.clip(severity, 0.0, 1.0)),
                            context=context,
                        ))

        return sites

    def _find_cleavage_sites_at(
        self, sequence: str, position: int, new_residue: str
    ) -> list[CleavageSite]:
        """Find cleavage sites after a hypothetical mutation."""
        mutated = list(sequence)
        mutated[position] = new_residue[0] if len(new_residue) == 1 else new_residue[0].upper()
        return self._find_cleavage_sites("".join(mutated))

    def _assess_n_terminal(self, sequence: str) -> float:
        """Assess N-terminal stability against aminopeptidases."""
        if not sequence:
            return 0.5

        first_three = sequence[:3].upper()
        score = 1.0

        # Basic residues at N-term are rapidly cleaved
        for aa in first_three:
            if aa in {"K", "R", "H"}:
                score -= 0.25
            elif aa in {"A", "G", "S"}:
                score -= 0.15
            elif aa == "P":
                score += 0.1  # Proline blocks aminopeptidases

        return float(np.clip(score, 0.0, 1.0))

    def _assess_c_terminal(self, sequence: str) -> float:
        """Assess C-terminal stability against carboxypeptidases."""
        if not sequence:
            return 0.5

        last_three = sequence[-3:].upper()
        score = 1.0

        for aa in last_three:
            if aa in {"K", "R"}:
                score -= 0.25
            elif aa in {"A", "G"}:
                score -= 0.15

        # C-terminal amidation would help (flag for recommendation)
        return float(np.clip(score, 0.0, 1.0))

    def _assess_internal_stability(
        self,
        sequence: str,
        cleavage_sites: list[CleavageSite],
    ) -> float:
        """Assess internal stability based on cleavage site density."""
        if not cleavage_sites:
            return 1.0

        total_severity = sum(s.severity for s in cleavage_sites)
        max_possible = len(sequence) * 0.8  # Rough upper bound
        stability = 1.0 - (total_severity / max_possible)

        return float(np.clip(stability, 0.0, 1.0))

    def _estimate_half_life(
        self, stability_score: float, length: int
    ) -> float:
        """
        Empirical half-life estimation in hours.
        Based on typical peptide pharmacokinetics.
        """
        # Base half-life depends on length (longer = slower renal clearance)
        base_half_life = 0.5 + length * 0.3  # hours

        # Stability modulates half-life exponentially
        half_life = base_half_life * (stability_score ** 0.5) * 3.0

        return float(np.clip(half_life, 0.1, 48.0))

    def _generate_recommendations(
        self,
        sequence: str,
        cleavage_sites: list[CleavageSite],
        stability: float,
    ) -> list[str]:
        """Generate actionable stability improvement recommendations."""
        recs: list[str] = []

        if stability < self.min_stability:
            recs.append(
                f"Overall stability ({stability:.2f}) is below target "
                f"({self.min_stability:.2f}). Consider ncAA substitutions."
            )

        # N-terminal recommendations
        if sequence and sequence[0].upper() in {"K", "R"}:
            recs.append(
                "N-terminal basic residue detected. Consider acetylation "
                "or substitution with D-amino acid."
            )

        # C-terminal recommendations
        if sequence and sequence[-1].upper() in {"K", "R", "A"}:
            recs.append(
                "C-terminal residue susceptible to carboxypeptidases. "
                "Consider C-terminal amidation."
            )

        # High-severity sites
        high_severity = [s for s in cleavage_sites if s.severity > 0.7]
        if high_severity:
            proteases = set(s.protease.value for s in high_severity)
            recs.append(
                f"{len(high_severity)} high-severity cleavage site(s) for "
                f"protease(s): {', '.join(proteases)}. "
                "Priority for ncAA substitution."
            )

        # Proline insertion suggestion
        if len(cleavage_sites) > 2:
            recs.append(
                "Multiple cleavage sites detected. Consider strategic "
                "proline insertions to block protease access."
            )

        # Macrocyclization suggestion for longer peptides
        if len(sequence) >= 8 and stability < 0.7:
            recs.append(
                "Peptide length ≥8 with moderate stability. Consider "
                "head-to-tail cyclization or side-chain stapling."
            )

        if not recs:
            recs.append("Metabolic stability profile is acceptable.")

        return recs
```

---

### `brownbiotech/arp_v3/agents/design/generator.py` (Modified)

```python
"""
Enhanced Peptide Design Generator — Integration point for metabolism-first
targeting with noncanonical amino acid support and metabolic constraints.

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from brownbiotech.arp_v3.agents.design.metabolic_constraint_solver import (
    MetabolicConstraintSolver,
    MetabolicProfile,
)
from brownbiotech.arp_v3.agents.design.noncanonical_generator import (
    NoncanonicalGenerator,
    GeneratedPeptide,
    NcAACategory,
)
from brownbiotech.arp_v3.agents.virtual_screen.canonical_vs_noncanonical import (
    BindingMode,
    CanonicalNoncanonicalClassifier,
    BindingModePrediction,
)

logger = logging.getLogger(__name__)


@dataclass
class DesignResult:
    """Complete result from the enhanced design pipeline."""
    parent_sequence: str
    binding_mode: BindingModePrediction
    metabolic_profile: MetabolicProfile
    generated_variants: list[GeneratedPeptide]
    stabilizing_mutations: list[dict]
    top_variant: Optional[GeneratedPeptide] = None
    passes_metabolic_filter: bool = False
    metadata: dict = field(default_factory=dict)


class EnhancedDesignGenerator:
    """
    Enhanced peptide design generator with metabolism-first targeting.

    Integrates binding mode classification, noncanonical amino acid
    generation, and metabolic constraint solving into a unified pipeline.

    Parameters
    ----------
    min_metabolic_stability : float
        Minimum metabolic stability score for variants (default: 0.6).
    n_variants : int
        Number of ncAA variants to generate (default: 20).
    ncAA_max_fraction : float
        Maximum ncAA fraction in generated variants (default: 0.4).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        min_metabolic_stability: float = 0.6,
        n_variants: int = 20,
        ncAA_max_fraction: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        self.min_metabolic = min_metabolic_stability
        self.n_variants = n_variants

        # Initialize sub-modules
        self.classifier = CanonicalNoncanonicalClassifier()
        self.nc_generator = NoncanonicalGenerator(
            max_ncAA_fraction=ncAA_max_fraction,
            seed=seed,
        )
        self.metabolic_solver = MetabolicConstraintSolver(
            min_stability_score=min_metabolic_stability,
        )

    def design(
        self,
        parent_sequence: str,
        binding_descriptors: Optional[dict] = None,
        protected_positions: Optional[set[int]] = None,
    ) -> DesignResult:
        """
        Run the full enhanced design pipeline.

        Parameters
        ----------
        parent_sequence : str
            Parent peptide sequence to improve.
        binding_descriptors : dict, optional
            Structural descriptors for binding mode classification.
            Keys: pocket_depth, buried_sasa_fraction, interface_hydrophobicity,
            n_hydrogen_bonds, n_pi_interactions, conformational_rmsd,
            membrane_proximity, groove_curvature.
        protected_positions : set[int], optional
            Positions that must not be mutated.

        Returns
        -------
        DesignResult
            Complete design result with variants and recommendations.
        """
        logger.info(
            f"Starting enhanced design for '{parent_sequence}' "
            f"(length: {len(parent_sequence)})"
        )

        # Step 1: Classify binding mode
        binding_mode = self._classify_binding_mode(binding_descriptors)
        logger.info(
            f"Binding mode: {binding_mode.mode.value} "
            f"(confidence: {binding_mode.confidence:.2f}, "
            f"noncanonical: {binding_mode.is_noncanonical})"
        )

        # Step 2: Analyze metabolic stability
        metabolic_profile = self.metabolic_solver.analyze(parent_sequence)
        logger.info(
            f"Metabolic stability: {metabolic_profile.overall_stability_score:.2f}, "
            f"est. half-life: {metabolic_profile.half_life_estimate_hours:.1f}h, "
            f"cleavage sites: {metabolic_profile.n_cleavage_sites}"
        )

        # Step 3: Determine allowed ncAA categories based on binding mode
        allowed_categories = self._select_ncAA_categories(binding_mode.mode)

        # Step 4: Determine target hydrophobicity
        target_hydro = self._select_target_hydrophobicity(binding_mode.mode)

        # Step 5: Generate ncAA variants
        variants = self.nc_generator.generate_variants(
            parent_sequence=parent_sequence,
            n_variants=self.n_variants,
            target_hydrophobicity=target_hydro,
            allowed_categories=allowed_categories,
            protected_positions=protected_positions,
        )

        # Step 6: Filter by metabolic stability
        filtered_variants = self._filter_by_metabolism(variants)

        # Step 7: Get stabilizing mutation suggestions
        stabilizing_mutations = self.metabolic_solver.suggest_stabilizing_mutations(
            metabolic_profile,
            max_mutations=3,
            preserve_positions=protected_positions,
        )

        # Step 8: Select top variant
        top_variant = filtered_variants[0] if filtered_variants else None
        passes_filter = len(filtered_variants) > 0

        result = DesignResult(
            parent_sequence=parent_sequence,
            binding_mode=binding_mode,
            metabolic_profile=metabolic_profile,
            generated_variants=filtered_variants,
            stabilizing_mutations=stabilizing_mutations,
            top_variant=top_variant,
            passes_metabolic_filter=passes_filter,
            metadata={
                "n_total_variants": len(variants),
                "n_filtered_variants": len(filtered_variants),
                "allowed_categories": [c.value for c in allowed_categories],
                "target_hydrophobicity": target_hydro,
            },
        )

        logger.info(
            f"Design complete: {len(filtered_variants)}/{len(variants)} "
            f"variants pass metabolic filter. "
            f"Top score: {top_variant.total_score:.3f}" if top_variant
            else "Design complete: no variants pass metabolic filter."
        )

        return result

    def _classify_binding_mode(
        self, descriptors: Optional[dict]
    ) -> BindingModePrediction:
        """Classify binding mode from descriptors or use defaults."""
        if descriptors is None:
            # Default: assume shallow surface for metabolism-first
            descriptors = {
                "pocket_depth": 4.0,
                "buried_sasa_fraction": 0.35,
                "interface_hydrophobicity": 0.5,
                "n_hydrogen_bonds": 5,
                "n_pi_interactions": 1,
                "conformational_rmsd": 1.5,
                "membrane_proximity": 0.3,
                "groove_curvature": 0.8,
            }

        return self.classifier.classify(**descriptors)

    def _select_ncAA_categories(
        self, mode: BindingMode
    ) -> set[NcAACategory]:
        """Select appropriate ncAA categories based on binding mode."""
        base_categories = {
            NcAACategory.METABOLIC_STABLE,
            NcAACategory.CONSTRAINED,
        }

        mode_categories = {
            BindingMode.CANONICAL_POCKET: {
                NcAACategory.HYDROPHOBIC,
                NcAACategory.FLUORINATED,
            },
            BindingMode.SHALLOW_SURFACE: {
                NcAACategory.HYDROPHOBIC,
                NcAACategory.CONSTRAINED,
                NcAACategory.AROMATIC,
            },
            BindingMode.LIPID_GROOVE: {
                NcAACategory.HYDROPHOBIC,
                NcAACategory.FLUORINATED,
            },
            BindingMode.INDUCED_FIT: {
                NcAACategory.CONSTRAINED,
                NcAACategory.METABOLIC_STABLE,
            },
            BindingMode.ALLOSTERIC_SURFACE: {
                NcAACategory.CONSTRAINED,
                NcAACategory.AROMATIC,
            },
            BindingMode.MEMBRANE_PROXIMAL: {
                NcAACategory.HYDROPHOBIC,
                NcAACategory.FLUORINATED,
                NcAACategory.CATIONIC,
            },
        }

        selected = base_categories | mode_categories.get(mode, set())
        return selected

    def _select_target_hydrophobicity(self, mode: BindingMode) -> float:
        """Select target hydrophobicity based on binding mode."""
        targets = {
            BindingMode.CANONICAL_POCKET: 0.50,
            BindingMode.SHALLOW_SURFACE: 0.55,
            BindingMode.LIPID_GROOVE: 0.70,
            BindingMode.INDUCED_FIT: 0.50,
            BindingMode.ALLOSTERIC_SURFACE: 0.55,
            BindingMode.MEMBRANE_PROXIMAL: 0.65,
        }
        return targets.get(mode, 0.50)

    def _filter_by_metabolism(
        self, variants: list[GeneratedPeptide]
    ) -> list[GeneratedPeptide]:
        """Filter variants by metabolic stability threshold."""
        filtered = [
            v for v in variants
            if v.metabolic_stability_score >= self.min_metabolic
        ]
        return filtered
```

---

## 3. Validation Pipeline Integration

### `brownbiotech/arp_v3/validation/metabolism_first_validator.py`

```python
"""
Metabolism-First Validation Pipeline — Validates that designed peptides
meet all metabolism-first targeting criteria before advancing to synthesis.

BrownBioTech Iteration 6/100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from brownbiotech.arp_v3.agents.design.generator import DesignResult
from brownbiotech.arp_v3.agents.design.metabolic_constraint_solver import (
    MetabolicProfile,
)
from brownbiotech.arp_v3.agents.virtual_screen.canonical_vs_noncanonical import (
    BindingMode,
)

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation outcome status."""
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"
    FAIL_CRITICAL = "fail_critical"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    check_name: str
    status: ValidationStatus
    value: float
    threshold: float
    message: str


@dataclass
class ValidationReport:
    """Complete validation report for a design."""
    sequence: str
    overall_status: ValidationStatus
    checks: list[ValidationCheck]
    warnings: list[str]
    critical_failures: list[str]
    score: float  # Aggregate validation score [0, 1]
    recommendation: str


class MetabolismFirstValidator:
    """
    Validates peptide designs against metabolism-first criteria.

    Checks include metabolic stability, binding mode appropriateness,
    ncAA incorporation quality, and synthesis feasibility.

    Parameters
    ----------
    min_metabolic_score : float
        Minimum metabolic stability score (default: 0.6).
    min_ncAA_fraction : float
        Minimum ncAA fraction for noncanonical designs (default: 0.15).
    max_ncAA_fraction : float
        Maximum ncAA fraction for synthesis feasibility (default: 0.5).
    min_half_life_hours : float
        Minimum estimated half-life (default: 2.0).
    max_peptide_length : int
        Maximum peptide length for synthesis (default: 25).
    """

    def __init__(
        self,
        min_metabolic_score: float = 0.6,
        min_ncAA_fraction: float = 0.15,
        max_ncAA_fraction: float = 0.5,
        min_half_life_hours: float = 2.0,
        max_peptide_length: int = 25,
    ) -> None:
        self.min_metabolic = min_metabolic_score
        self.min_ncAA = min_ncAA_fraction
        self.max_ncAA = max_ncAA_fraction
        self.min_half_life = min_half_life_hours
        self.max_length = max_peptide_length

    def validate(self, design_result: DesignResult) -> ValidationReport:
        """
        Validate a complete design result.

        Parameters
        ----------
        design_result : DesignResult
            Output from EnhancedDesignGenerator.design().

        Returns
        -------
        ValidationReport
            Complete validation report with pass/fail status.
        """
        checks: list[ValidationCheck] = []
        warnings: list[str] = []
        critical_failures: list[str] = []

        variant = design_result.top_variant
        if variant is None:
            return ValidationReport(
                sequence=design_result.parent_sequence,
                overall_status=ValidationStatus.FAIL_CRITICAL,
                checks=[],
                warnings=["No variants generated."],
                critical_failures=["No viable variants produced by design pipeline."],
                score=0.0,
                recommendation="Re-run design with relaxed constraints.",
            )

        # Check 1: Metabolic stability
        metab_check = self._check_metabolic_stability(
            variant.metabolic_stability_score
        )
        checks.append(metab_check)
        if metab_check.status == ValidationStatus.FAIL_CRITICAL:
            critical_failures.append(metab_check.message)
        elif metab_check.status == ValidationStatus.FAIL:
            warnings.append(metab_check.message)

        # Check 2: Half-life
        half_life_check = self._check_half_life(
            design_result.metabolic_profile.half_life_estimate_hours
        )
        checks.append(half_life_check)
        if half_life_check.status in (
            ValidationStatus.FAIL, ValidationStatus.FAIL_CRITICAL
        ):
            warnings.append(half_life_check.message)

        # Check 3: ncAA fraction
        ncAA_check = self._check_ncAA_fraction(
            variant.ncAA_fraction,
            design_result.binding_mode.is_noncanonical,
        )
        checks.append(ncAA_check)
        if ncAA_check.status == ValidationStatus.FAIL:
            warnings.append(ncAA_check.message)

        # Check 4: Peptide length
        length_check = self._check_length(variant.length)
        checks.append(length_check)
        if length_check.status == ValidationStatus.FAIL_CRITICAL:
            critical_failures.append(length_check.message)

        # Check 5: Binding mode consistency
        mode_check = self._check_binding_mode_consistency(design_result)
        checks.append(mode_check)
        if mode_check.status == ValidationStatus.FAIL:
            warnings.append(mode_check.message)

        # Check 6: Hydrophobicity balance
        hydro_check = self._check_hydrophobicity(variant.hydrophobicity)
        checks.append(hydro_check)
        if hydro_check.status == ValidationStatus.PASS_WITH_WARNINGS:
            warnings.append(hydro_check.message)

        # Check 7: Cleavage site reduction
        cleavage_check = self._check_cleavage_reduction(design_result)
        checks.append(cleavage_check)
        if cleavage_check.status == ValidationStatus.FAIL:
            warnings.append(cleavage_check.message)

        # Aggregate
        score = np.mean([c.value for c in checks])
        overall = self._determine_overall_status(checks, critical_failures)
        recommendation = self._generate_final_recommendation(
            overall, critical_failures, warnings
        )

        return ValidationReport(
            sequence=variant.full_sequence,
            overall_status=overall,
            checks=checks,
            warnings=warnings,
            critical_failures=critical_failures,
            score=float(score),
            recommendation=recommendation,
        )

    def _check_metabolic_stability(
        self, score: float
    ) -> ValidationCheck:
        if score >= self.min_metabolic:
            return ValidationCheck(
                check_name="metabolic_stability",
                status=ValidationStatus.PASS,
                value=score,
                threshold=self.min_metabolic,
                message=f"Metabolic stability {score:.2f} ≥ {self.min_metabolic:.2f}.",
            )
        if score >= self.min_metabolic * 0.8:
            return ValidationCheck(
                check_name="metabolic_stability",
                status=ValidationStatus.FAIL,
                value=score,
                threshold=self.min_metabolic,
                message=(
                    f"Metabolic stability {score:.2f} below threshold "
                    f"{self.min_metabolic:.2f} but within 80%."
                ),
            )
        return ValidationCheck(
            check_name="metabolic_stability",
            status=ValidationStatus.FAIL_CRITICAL,
            value=score,
            threshold=self.min_metabolic,
            message=(
                f"Metabolic stability {score:.2f} critically below "
                f"threshold {self.min_metabolic:.2f}."
            ),
        )

    def _check_half_life(self, half_life: float) -> ValidationCheck:
        status = (
            ValidationStatus.PASS
            if half_life >= self.min_half_life
            else ValidationStatus.FAIL
        )
        return ValidationCheck(
            check_name="half_life",
            status=status,
            value=half_life,
            threshold=self.min_half_life,
            message=(
                f"Estimated half-life {half_life:.1f}h "
                f"{'≥' if status == ValidationStatus.PASS else '<'} "
                f"target {self.min_half_life:.1f}h."
            ),
        )

    def _check_ncAA_fraction(
        self, fraction: float, is_noncanonical: bool
    ) -> ValidationCheck:
        if is_noncanonical and fraction < self.min_ncAA:
            return ValidationCheck(
                check_name="ncAA_fraction",
                status=ValidationStatus.FAIL,
                value=fraction,
                threshold=self.min_ncAA,
                message=(
                    f"Noncanonical design requires ≥{self.min_ncAA:.0%} ncAA "
                    f"but got {fraction:.0%}."
                ),
            )
        if fraction > self.max_ncAA:
            return ValidationCheck(
                check_name="ncAA_fraction",
                status=ValidationStatus.FAIL,
                value=fraction,
                threshold=self.max_ncAA,
                message=(
                    f"ncAA fraction {fraction:.0%} exceeds synthesis "
                    f"limit {self.max_ncAA:.0%}."
                ),
            )
        return ValidationCheck(
            check_name="ncAA_fraction",
            status=ValidationStatus.PASS,
            value=fraction,
            threshold=self.max_ncAA,
            message=f"ncAA fraction {fraction:.0%} within acceptable range.",
        )

    def _check_length(self, length: int) -> ValidationCheck:
        status = (
            ValidationStatus.PASS
            if length <= self.max_length
            else ValidationStatus.FAIL_CRITICAL
        )
        return ValidationCheck(
            check_name="peptide_length",
            status=status,
            value=float(length),
            threshold=float(self.max_length),
            message=(
                f"Length {length} "
                f"{'≤' if status == ValidationStatus.PASS else '>'} "
                f"maximum {self.max_length}."
            ),
        )

    def _check_binding_mode_consistency(
        self, design: DesignResult
    ) -> ValidationCheck:
        mode = design.binding_mode.mode
        variant = design.top_variant
        if variant is None:
            return ValidationCheck(
                check_name="binding_mode_consistency",
                status=ValidationStatus.FAIL,
                value=0.0,
                threshold=1.0,
                message="No variant to check.",
            )

        # Check if ncAA categories match binding mode
        is_consistent = True
        if mode == BindingMode.LIPID_GROOVE:
            if variant.hydrophobicity < 0.5:
                is_consistent = False
        elif mode == BindingMode.MEMBRANE_PROXIMAL:
            if variant.hydrophobicity < 0.45:
                is_consistent = False

        status = (
            ValidationStatus.PASS if is_consistent
            else ValidationStatus.FAIL
        )
        return ValidationCheck(
            check_name="binding_mode_consistency",
            status=status,
            value=1.0 if is_consistent else 0.0,
            threshold=1.0,
            message=(
                f"Binding mode {mode.value} "
                f"{'consistent' if is_consistent else 'inconsistent'} "
                f"with variant properties."
            ),
        )

    def _check_hydrophobicity(self, hydro: float) -> ValidationCheck:
        if 0.3 <= hydro <= 0.75:
            return ValidationCheck(
                check_name="hydrophobicity_balance",
                status=ValidationStatus.PASS,
                value=hydro,
                threshold=0.75,
                message=f"Hydrophobicity {hydro:.2f} in optimal range.",
            )
        return ValidationCheck(
            check_name="hydrophobicity_balance",
            status=ValidationStatus.PASS_WITH_WARNINGS,
            value=hydro,
            threshold=0.75,
            message=(
                f"Hydrophobicity {hydro:.2f} outside optimal range "
                f"[0.3, 0.75]. May affect solubility or binding."
            ),
        )

    def _check_cleavage_reduction(
        self, design: DesignResult
    ) -> ValidationCheck:
        parent_sites = design.metabolic_profile.n_cleavage_sites
        variant = design.top_variant

        if variant is None:
            return ValidationCheck(
                check_name="cleavage_reduction",
                status=ValidationStatus.FAIL,
                value=0.0,
                threshold=0.0,
                message="No variant to check.",
            )

        # Re-analyze variant for cleavage sites (approximate)
        # In production, this would re-run the metabolic solver
        # Here we estimate from mutations
        n_mutations = variant.n_noncanonical
        estimated_reduction = min(n_mutations * 0.5, parent_sites)
        remaining = max(0, parent_sites - estimated_reduction)

        if remaining < parent_sites * 0.5:
            return ValidationCheck(
                check_name="cleavage_reduction",
                status=ValidationStatus.PASS,
                value=float(remaining),
                threshold=float(parent_sites * 0.5),
                message=(
                    f"Estimated cleavage sites reduced from {parent_sites} "
                    f"to ~{remaining:.0f}."
                ),
            )
        return ValidationCheck(
            check_name="cleavage_reduction",
            status=ValidationStatus.FAIL,
            value=float(remaining),
            threshold=float(parent_sites * 0.5),
            message=(
                f"Cleavage site reduction insufficient: "
                f"{parent_sites} → ~{remaining:.0f}."
            ),
        )

    def _determine_overall_status(
        self,
        checks: list[ValidationCheck],
        critical_failures: list[str],
    ) -> ValidationStatus:
        if critical_failures:
            return ValidationStatus.FAIL_CRITICAL
        if any(c.status == ValidationStatus.FAIL for c in checks):
            return ValidationStatus.FAIL
        if any(
            c.status == ValidationStatus.PASS_WITH_WARNINGS for c in checks
        ):
            return ValidationStatus.PASS_WITH_WARNINGS
        return ValidationStatus.PASS

    def _generate_final_recommendation(
        self,
        status: ValidationStatus,
        critical: list[str],
        warnings: list[str],
    ) -> str:
        if status == ValidationStatus.PASS:
            return (
                "Design passes all validation checks. "
                "Ready for synthesis nomination."
            )
        if status == ValidationStatus.PASS_WITH_WARNINGS:
            return (
                "Design passes with warnings. Review warnings before "
                "advancing: " + "; ".join(warnings[:3])
            )
        if status == ValidationStatus.FAIL:
            return (
                "Design fails validation. Address failures: "
                + "; ".join(warnings[:3])
            )
        return (
            "Design has critical failures that must be resolved: "
            + "; ".join(critical[:3])
        )
```

---

## 4. Integration Test / Demo Script

### `brownbiotech/arp_v3/demo_iteration_6.py`

```python
"""
Demo script for Iteration 6/100 — Metabolism-First Targeting Enhancement.
Run to verify all modules work together end-to-end.
"""

from __future__ import annotations

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

from brownbiotech.arp_v3.agents.virtual_screen.shallow_surface_detector import (
    ShallowSurfaceDetector,
)
from brownbiotech.arp_v3.agents.virtual_screen.lipid_groove_featurizer import (
    LipidGrooveFeaturizer,
)
from brownbiotech.arp_v3.agents.virtual_screen.canonical_vs_noncanonical import (
    CanonicalNoncanonicalClassifier,
)
from brownbiotech.arp_v3.agents.design.generator import EnhancedDesignGenerator
from brownbiotech.arp_v3.validation.metabolism_first_validator import (
    MetabolismFirstValidator,
)


def demo_surface_detection():
    """Demo shallow surface detection on synthetic data."""
    print("\n" + "=" * 60)
    print("DEMO: Shallow Surface Detection")
    print("=" * 60)

    # Create a synthetic shallow groove (elongated, shallow cluster)
    np.random.seed(42)
    n_atoms = 60

    # Elongated groove along x-axis, shallow along z
    x = np.random.uniform(-15, 15, n_atoms)
    y = np.random.normal(0, 2.0, n_atoms)
    z = np.random.normal(0, 1.5, n_atoms)  # Shallow in z
    coords = np.column_stack([x, y, z])

    # Hydrophobic residues for lipid groove
    residues = list(np.random.choice(
        list("LIVFAWMC"), size=n_atoms
    ))
    res_indices = list(range(0, n_atoms, 3)) * 2 + list(range(1, n_atoms, 3))

    detector = ShallowSurfaceDetector(
        min_patch_area=50.0,
        max_depth=6.0,
    )

    patches = detector.detect_shallow_sites(
        coords=coords,
        residue_names=residues,
        residue_indices=res_indices,
    )

    print(f"  Detected {len(patches)} shallow surface patches")
    for p in patches[:3]:
        print(
            f"    Patch {p.patch_id}: {p.topology.value}, "
            f"area={p.area_angstrom2:.0f}Å², "
            f"depth={p.depth_angstrom:.1f}Å, "
            f"hydro={p.hydrophobicity_score:.2f}, "
            f"druggability={detector._druggability_score(p):.3f}"
        )


def demo_lipid_featurization():
    """Demo lipid groove featurization."""
    print("\n" + "=" * 60)
    print("DEMO: Lipid Groove Featurization")
    print("=" * 60)

    np.random.seed(42)
    # Elongated lipid-binding groove
    n = 40
    x = np.random.uniform(-12, 12, n)
    y = np.random.normal(0, 3.0, n)
    z = np.random.normal(0, 2.0, n)
    coords = np.column_stack([x, y, z])

    residues = list(np.random.choice(
        list("LIVFWYK"), size=n, p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15]
    ))

    featurizer = LipidGrooveFeaturizer()
    features = featurizer.featurize(coords, residues, groove_id=0)

    print(f"  Groove dimensions: {features.length_angstrom:.1f} × "
          f"{features.width_angstrom:.1f} × {features.depth_angstrom:.1f} Å")
    print(f"  Hydrophobic fraction: {features.hydrophobicity_score:.2f}")
    print(f"  Headgroup compatibility: {features.lipid_headgroup_compatibility:.2f}")
    print(f"  Acyl chain depth: {features.acyl_chain_depth:.1f} Å")
    print(f"  Membrane proximity: {features.membrane_proximity:.2f}")
    print(f"  Feature vector shape: {features.feature_vector.shape}")


def demo_binding_mode_classification():
    """Demo canonical vs noncanonical classification."""
    print("\n" + "=" * 60)
    print("DEMO: Binding Mode Classification")
    print("=" * 60)

    classifier = CanonicalNoncanonicalClassifier()

    test_cases = [
        {
            "name": "Deep pocket (canonical)",
            "params": dict(
                pocket_depth=10.0, buried_sasa_fraction=0.75,
                interface_hydrophobicity=0.5, n_hydrogen_bonds=8,
                n_pi_interactions=2,
            ),
        },
        {
            "name": "Shallow groove (noncanonical)",
            "params": dict(
                pocket_depth=3.5, buried_sasa_fraction=0.30,
                interface_hydrophobicity=0.55, n_hydrogen_bonds=4,
                n_pi_interactions=1, groove_curvature=0.9,
            ),
        },
        {
            "name": "Lipid groove (noncanonical)",
            "params": dict(
                pocket_depth=4.0, buried_sasa_fraction=0.35,
                interface_hydrophobicity=0.75, n_hydrogen_bonds=3,
                n_pi_interactions=0, membrane_proximity=0.8,
            ),
        },
    ]

    for case in test_cases:
        pred = classifier.classify(**case["params"])
        print(f"  {case['name']}:")
        print(f"    Mode: {pred.mode.value} "
              f"(noncanonical: {pred.is_noncanonical})")
        print(f"    Confidence: {pred.confidence:.2f}")
        print(f"    Top scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted(pred.scores.items(), key=lambda x: x[1], reverse=True)[:3])}")
        print(f"    → {pred.recommendation[:80]}...")


def demo_full_pipeline():
    """Demo the full enhanced design + validation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO: Full Design + Validation Pipeline")
    print("=" * 60)

    # Parent peptide (typical initial hit from screening)
    parent = "KFLVAGWRSMP"

    # Binding descriptors for a shallow lipid groove target
    descriptors = {
        "pocket_depth": 4.2,
        "buried_sasa_fraction": 0.32,
        "interface_hydrophobicity": 0.65,
        "n_hydrogen_bonds": 4,
        "n_pi_interactions": 1,
        "conformational_rmsd": 1.8,
        "membrane_proximity": 0.7,
        "groove_curvature": 0.85,
    }

    # Run design pipeline
    designer = EnhancedDesignGenerator(
        min_metabolic_stability=0.55,
        n_variants=15,
        ncAA_max_fraction=0.45,
        seed=42,
    )

    result = designer.design(
        parent_sequence=parent,
        binding_descriptors=descriptors,
        protected_positions={0, 9},  # Protect termini
    )

    print(f"  Parent: {parent}")
    print(f"  Binding mode: {result.binding_mode.mode.value}")
    print(f"  Metabolic stability (parent): "
          f"{result.metabolic_profile.overall_stability_score:.2f}")
    print(f"  Cleavage sites (parent): {result.metabolic_profile.n_cleavage_sites}")
    print(f"  Variants generated: {result.metadata['n_total_variants']}")
    print(f"  Variants passing filter: {result.metadata['n_filtered_variants']}")

    if result.top_variant:
        v = result.top_variant
        print(f"\n  Top variant: {v.full_sequence}")
        print(f"    ncAA fraction: {v.ncAA_fraction:.0%}")
        print(f"    Metabolic stability: {v.metabolic_stability_score:.2f}")
        print(f"    Surface complementarity: {v.surface_complementarity_score:.2f}")
        print(f"    Total score: {v.total_score:.3f}")
        print(f"    Mutations: {[(f'{m[1]}{m[0]+1}→{m[2]}') for m in v.mutations]}")

    # Stabilizing mutation suggestions
    if result.stabilizing_mutations:
        print(f"\n  Stabilizing mutations suggested:")
        for mut in result.stabilizing_mutations:
            print(f"    {mut['rationale']}")

    # Run validation
    print("\n  --- Validation ---")
    validator = MetabolismFirstValidator(
        min_metabolic_score=0.55,
        min_half_life_hours=1.5,
    )
    report = validator.validate(result)

    print(f"  Overall status: {report.overall_status.value}")
    print(f"  Validation score: {report.score:.3f}")
    for check in report.checks:
        icon = {"pass": "✓", "pass_with_warnings": "⚠",
                "fail": "✗", "fail_critical": "✗✗"}
        print(f"    {icon.get(check.status.value, '?')} {check.check_name}: "
              f"{check.value:.2f} (threshold: {check.threshold:.2f})")
    if report.warnings:
        print(f"  Warnings: {len(report.warnings)}")
    print(f"  → {report.recommendation}")


if __name__ == "__main__":
    demo_surface_detection()
    demo_lipid_featurization()
    demo_binding_mode_classification()
    demo_full_pipeline()
    print("\n" + "=" * 60)
    print("Iteration 6/100 demo complete.")
    print("=" * 60)
```

---

## Summary of Improvements

| Module | File | Purpose |
|--------|------|---------|
| **ShallowSurfaceDetector** | `virtual_screen/shallow_surface_detector.py` | Identifies shallow, groove-like binding sites on protein surfaces using geometric clustering and physicochemical scoring |
| **LipidGrooveFeaturizer** | `virtual_screen/lipid_groove_featurizer.py` | Extracts 11-dimensional feature vectors from lipid-binding grooves for ML scoring models |
| **CanonicalNoncanonicalClassifier** | `virtual_screen/canonical_vs_noncanonical.py` | Routes interactions to canonical vs noncanonical pipelines based on structural descriptors |
| **NoncanonicalGenerator** | `design/noncanonical_generator.py` | Generates peptide variants with 17 ncAAs from a curated library, targeting hydrophobicity and stability |
| **MetabolicConstraintSolver** | `design/metabolic_constraint_solver.py` | Identifies protease cleavage sites (6 protease types) and suggests stabilizing substitutions |
| **EnhancedDesignGenerator** | `design/generator.py` | Unified pipeline integrating binding mode classification → ncAA selection → metabolic filtering |
| **MetabolismFirstValidator** | `validation/metabolism_first_validator.py` | 7-check validation gate (metabolic stability, half-life, ncAA fraction, length, mode consistency, hydrophobicity, cleavage reduction) |