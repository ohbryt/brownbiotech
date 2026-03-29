# BrownBioTech Iteration 9/100: Network Propagation & Stratification Module

## File: `brownbiotech/multiomics/network_propagation.py`

```python
"""
Network Propagation & Stratification Module for MultiOmics Agent v3.0

This module implements graph-based signal propagation for biological networks
and patient stratification based on propagated multi-omics features.

Key capabilities:
- Diffusion-based network propagation (random walk with restart)
- Integration of DGAT1/YARS2 mechanistic findings
- Resistance loop detection via propagation patterns
- Patient stratification using propagated feature embeddings
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

logger = logging.getLogger(__name__)


class PropagationMethod(Enum):
    """Supported network propagation algorithms."""
    RANDOM_WALK_RESTART = "rwr"
    HEAT_DIFFUSION = "heat"
    PERSONALIZED_PAGERANK = "ppr"


class StratificationMethod(Enum):
    """Supported patient stratification approaches."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    GMM = "gmm"


@dataclass
class PropagationResult:
    """Container for network propagation results."""
    propagated_scores: np.ndarray
    convergence_history: List[float]
    converged: bool
    iterations: int
    method: PropagationMethod
    node_order: List[str]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert propagated scores to DataFrame."""
        return pd.DataFrame({
            'node': self.node_order,
            'propagated_score': self.propagated_scores
        }).sort_values('propagated_score', ascending=False)


@dataclass
class StratificationResult:
    """Container for patient stratification results."""
    patient_ids: List[str]
    cluster_labels: np.ndarray
    cluster_probabilities: Optional[np.ndarray]
    method: StratificationMethod
    n_clusters: int
    silhouette_score: Optional[float]
    feature_importance: Optional[Dict[str, float]]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert stratification results to DataFrame."""
        result = pd.DataFrame({
            'patient_id': self.patient_ids,
            'cluster': self.cluster_labels
        })
        if self.cluster_probabilities is not None:
            for i in range(self.n_clusters):
                result[f'cluster_{i}_prob'] = self.cluster_probabilities[:, i]
        return result


@dataclass
class ResistanceLoopSignature:
    """Signature for detected resistance loops in propagation patterns."""
    loop_id: str
    source_genes: List[str]
    propagated_targets: List[str]
    loop_strength: float
    is_dgat1_mediated: bool
    is_yars2_mediated: bool
    patient_prevalence: float


class NetworkPropagator:
    """
    Graph-based network propagation engine.
    
    Implements diffusion-based signal propagation through biological networks
    to identify functional modules and disease-associated pathways.
    
    Parameters
    ----------
    adjacency_matrix : sparse matrix
        Sparse adjacency matrix representing the biological network
    node_names : List[str]
        Ordered list of node identifiers matching matrix indices
    method : PropagationMethod
        Propagation algorithm to use
    """
    
    def __init__(
        self,
        adjacency_matrix: sparse.spmatrix,
        node_names: List[str],
        method: PropagationMethod = PropagationMethod.RANDOM_WALK_RESTART
    ) -> None:
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if len(node_names) != adjacency_matrix.shape[0]:
            raise ValueError(
                f"Node names length ({len(node_names)}) must match "
                f"matrix dimension ({adjacency_matrix.shape[0]})"
            )
        
        self.adjacency = adjacency_matrix
        self.node_names = node_names
        self.node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        self.method = method
        self._transition_matrix: Optional[sparse.spmatrix] = None
        self._degree_matrix: Optional[sparse.spmatrix] = None
        
        logger.info(
            f"Initialized NetworkPropagator with {len(node_names)} nodes "
            f"using {method.value} method"
        )
    
    def _compute_transition_matrix(self) -> sparse.spmatrix:
        """Compute column-normalized transition matrix for random walk."""
        if self._transition_matrix is not None:
            return self._transition_matrix
        
        # Add self-loops for stability
        adj = self.adjacency + sparse.eye(self.adjacency.shape[0])
        
        # Column-normalize for right-stochastic matrix
        col_sums = np.array(adj.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        
        # D^{-1} @ A
        diag_inv = sparse.diags(1.0 / col_sums)
        self._transition_matrix = diag_inv @ adj
        
        return self._transition_matrix
    
    def _compute_degree_matrix(self) -> sparse.spmatrix:
        """Compute degree matrix for heat diffusion."""
        if self._degree_matrix is not None:
            return self._degree_matrix
        
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()
        self._degree_matrix = sparse.diags(degrees)
        return self._degree_matrix
    
    def propagate(
        self,
        seed_vector: Union[np.ndarray, Dict[str, float]],
        restart_prob: float = 0.7,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        diffusion_coefficient: float = 0.1
    ) -> PropagationResult:
        """
        Run network propagation from seed nodes.
        
        Parameters
        ----------
        seed_vector : np.ndarray or Dict[str, float]
            Initial signal. Either array matching node order or 
            dict mapping node names to initial scores
        restart_prob : float
            Restart probability for RWR (higher = more local)
        max_iterations : int
            Maximum propagation iterations
        tolerance : float
            Convergence threshold
        diffusion_coefficient : float
            Time parameter for heat diffusion
            
        Returns
        -------
        PropagationResult
            Container with propagated scores and metadata
        """
        # Convert dict to vector if needed
        if isinstance(seed_vector, dict):
            vec = np.zeros(len(self.node_names))
            for node, score in seed_vector.items():
                if node in self.node_to_idx:
                    vec[self.node_to_idx[node]] = score
                else:
                    warnings.warn(f"Seed node '{node}' not found in network")
            seed_vector = vec
        
        if len(seed_vector) != len(self.node_names):
            raise ValueError(
                f"Seed vector length ({len(seed_vector)}) must match "
                f"number of nodes ({len(self.node_names)})"
            )
        
        # Normalize seed vector
        seed_norm = seed_vector / (np.linalg.norm(seed_vector) + 1e-10)
        
        if self.method == PropagationMethod.RANDOM_WALK_RESTART:
            result = self._rwr_propagate(
                seed_norm, restart_prob, max_iterations, tolerance
            )
        elif self.method == PropagationMethod.HEAT_DIFFUSION:
            result = self._heat_propagate(
                seed_norm, diffusion_coefficient, max_iterations, tolerance
            )
        elif self.method == PropagationMethod.PERSONALIZED_PAGERANK:
            result = self._ppr_propagate(
                seed_norm, restart_prob, max_iterations, tolerance
            )
        else:
            raise ValueError(f"Unknown propagation method: {self.method}")
        
        return result
    
    def _rwr_propagate(
        self,
        seed: np.ndarray,
        restart_prob: float,
        max_iterations: int,
        tolerance: float
    ) -> PropagationResult:
        """Random Walk with Restart propagation."""
        W = self._compute_transition_matrix()
        
        current = seed.copy()
        history = []
        
        for i in range(max_iterations):
            new = (1 - restart_prob) * (W @ current) + restart_prob * seed
            diff = np.linalg.norm(new - current)
            history.append(diff)
            current = new
            
            if diff < tolerance:
                logger.info(f"RWR converged after {i+1} iterations")
                return PropagationResult(
                    propagated_scores=current,
                    convergence_history=history,
                    converged=True,
                    iterations=i + 1,
                    method=self.method,
                    node_order=self.node_names
                )
        
        logger.warning(f"RWR did not converge after {max_iterations} iterations")
        return PropagationResult(
            propagated_scores=current,
            convergence_history=history,
            converged=False,
            iterations=max_iterations,
            method=self.method,
            node_order=self.node_names
        )
    
    def _heat_propagate(
        self,
        seed: np.ndarray,
        t: float,
        max_iterations: int,
        tolerance: float
    ) -> PropagationResult:
        """Heat diffusion propagation."""
        D = self._compute_degree_matrix()
        L = D - self.adjacency  # Graph Laplacian
        
        # Approximate exp(-tL) @ seed via power series
        current = seed.copy()
        history = []
        term = seed.copy()
        factorial = 1.0
        
        for i in range(max_iterations):
            factorial *= (i + 1)
            term = -t * (L @ term)
            new = current + term / factorial
            
            diff = np.linalg.norm(new - current)
            history.append(diff)
            current = new
            
            if diff < tolerance:
                logger.info(f"Heat diffusion converged after {i+1} iterations")
                return PropagationResult(
                    propagated_scores=current,
                    convergence_history=history,
                    converged=True,
                    iterations=i + 1,
                    method=self.method,
                    node_order=self.node_names
                )
        
        return PropagationResult(
            propagated_scores=current,
            convergence_history=history,
            converged=False,
            iterations=max_iterations,
            method=self.method,
            node_order=self.node_names
        )
    
    def _ppr_propagate(
        self,
        seed: np.ndarray,
        alpha: float,
        max_iterations: int,
        tolerance: float
    ) -> PropagationResult:
        """Personalized PageRank propagation."""
        W = self._compute_transition_matrix()
        
        current = np.ones(len(self.node_names)) / len(self.node_names)
        history = []
        
        for i in range(max_iterations):
            new = alpha * seed + (1 - alpha) * (W @ current)
            diff = np.linalg.norm(new - current)
            history.append(diff)
            current = new
            
            if diff < tolerance:
                logger.info(f"PPR converged after {i+1} iterations")
                return PropagationResult(
                    propagated_scores=current,
                    convergence_history=history,
                    converged=True,
                    iterations=i + 1,
                    method=self.method,
                    node_order=self.node_names
                )
        
        return PropagationResult(
            propagated_scores=current,
            convergence_history=history,
            converged=False,
            iterations=max_iterations,
            method=self.method,
            node_order=self.node_names
        )


class PatientStratifier:
    """
    Patient stratification using propagated multi-omics features.
    
    Integrates DGAT1/YARS2 mechanistic signatures for biologically-informed
    clustering of patient cohorts.
    
    Parameters
    ----------
    propagator : NetworkPropagator
        Fitted network propagator instance
    """
    
    # Known resistance loop genes from validated findings
    DGAT1_PATHWAY_GENES = [
        'DGAT1', 'DGAT2', 'GPAT1', 'GPAT2', 'AGPAT', 'LPIN1',
        'SCD', 'FASN', 'ACLY', 'ACC', 'SREBF1', 'MLXIPL'
    ]
    
    YARS2_PATHWAY_GENES = [
        'YARS2', 'MT-TY', 'MRPL12', 'MRPL44', 'GFM1', 'GFM2',
        'MTFMT', 'TUFM', 'DARS2', 'HARS2'
    ]
    
    RESISTANCE_LOOP_MARKERS = [
        'ABCB1', 'ABCG2', 'MGMT', 'ERCC1', 'RRM1', 'TYMS',
        'DHFR', 'TOP1', 'TOP2A', 'PARP1'
    ]
    
    def __init__(self, propagator: NetworkPropagator) -> None:
        self.propagator = propagator
        self._propagated_features: Optional[np.ndarray] = None
        self._patient_ids: Optional[List[str]] = None
        self._feature_names: Optional[List[str]] = None
        
        logger.info("Initialized PatientStratifier")
    
    def compute_propagated_features(
        self,
        patient_expression: pd.DataFrame,
        seed_genes: Optional[List[str]] = None,
        restart_prob: float = 0.5
    ) -> np.ndarray:
        """
        Compute propagated features for all patients.
        
        Parameters
        ----------
        patient_expression : pd.DataFrame
            Rows=patients, columns=genes, values=expression
        seed_genes : List[str], optional
            Genes to use as propagation seeds. If None, uses all expressed genes.
        restart_prob : float
            Restart probability for propagation
            
        Returns
        -------
        np.ndarray
            Propagated feature matrix (patients x propagated_genes)
        """
        self._patient_ids = list(patient_expression.index)
        
        # Find overlapping genes
        network_genes = set(self.propagator.node_names)
        expr_genes = set(patient_expression.columns)
        overlap = list(network_genes & expr_genes)
        
        if len(overlap) < 10:
            raise ValueError(
                f"Only {len(overlap)} genes overlap between network and expression. "
                f"Check gene identifier compatibility."
            )
        
        logger.info(f"Found {len(overlap)} overlapping genes for propagation")
        
        # Subset expression matrix
        expr_subset = patient_expression[overlap].values
        gene_to_idx = {g: i for i, g in enumerate(overlap)}
        
        # Determine seed genes
        if seed_genes is None:
            seed_genes = overlap
        
        # Compute propagated features for each patient
        propagated_features = []
        
        for patient_idx in range(len(self._patient_ids)):
            patient_expr = expr_subset[patient_idx, :]
            
            # Create seed vector from patient expression
            seed_dict = {}
            for gene in seed_genes:
                if gene in gene_to_idx:
                    seed_dict[gene] = max(0, patient_expr[gene_to_idx[gene]])
            
            # Run propagation
            result = self.propagator.propagate(
                seed_dict,
                restart_prob=restart_prob
            )
            
            propagated_features.append(result.propagated_scores)
        
        self._propagated_features = np.array(propagated_features)
        self._feature_names = self.propagator.node_names
        
        logger.info(
            f"Computed propagated features: {self._propagated_features.shape}"
        )
        
        return self._propagated_features
    
    def stratify(
        self,
        n_clusters: int = 3,
        method: StratificationMethod = StratificationMethod.KMEANS,
        compute_silhouette: bool = True
    ) -> StratificationResult:
        """
        Stratify patients based on propagated features.
        
        Parameters
        ----------
        n_clusters : int
            Number of patient clusters
        method : StratificationMethod
            Clustering algorithm to use
        compute_silhouette : bool
            Whether to compute silhouette score
            
        Returns
        -------
        StratificationResult
            Container with stratification results
        """
        if self._propagated_features is None:
            raise RuntimeError(
                "Must call compute_propagated_features() before stratify()"
            )
        
        X = self._propagated_features
        
        # Standardize features
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Run clustering
        if method == StratificationMethod.KMEANS:
            labels, probs, importance = self._kmeans_cluster(
                X_std, n_clusters
            )
        elif method == StratificationMethod.HIERARCHICAL:
            labels, probs, importance = self._hierarchical_cluster(
                X_std, n_clusters
            )
        elif method == StratificationMethod.SPECTRAL:
            labels, probs, importance = self._spectral_cluster(
                X_std, n_clusters
            )
        elif method == StratificationMethod.GMM:
            labels, probs, importance = self._gmm_cluster(
                X_std, n_clusters
            )
        else:
            raise ValueError(f"Unknown stratification method: {method}")
        
        # Compute silhouette score
        sil_score = None
        if compute_silhouette and n_clusters > 1 and len(labels) > n_clusters:
            try:
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(X_std, labels)
            except ImportError:
                logger.warning("sklearn not available, skipping silhouette score")
        
        logger.info(
            f"Stratified {len(labels)} patients into {n_clusters} clusters "
            f"(silhouette={sil_score:.3f})" if sil_score else
            f"Stratified {len(labels)} patients into {n_clusters} clusters"
        )
        
        return StratificationResult(
            patient_ids=self._patient_ids,
            cluster_labels=labels,
            cluster_probabilities=probs,
            method=method,
            n_clusters=n_clusters,
            silhouette_score=sil_score,
            feature_importance=importance
        )
    
    def _kmeans_cluster(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """K-means clustering implementation."""
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(X)
            
            # Feature importance from cluster centers
            centers = kmeans.cluster_centers_
            center_spread = centers.std(axis=0)
            importance = self._compute_feature_importance(center_spread)
            
            return labels, None, importance
            
        except ImportError:
            return self._simple_kmeans(X, n_clusters)
    
    def _simple_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """Simple k-means fallback without sklearn."""
        np.random.seed(42)
        n_samples = X.shape[0]
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[idx].copy()
        
        for _ in range(max_iter):
            # Assign clusters
            distances = np.array([
                np.linalg.norm(X - c, axis=1) for c in centroids
            ]).T
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) 
                else centroids[i]
                for i in range(n_clusters)
            ])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        center_spread = centroids.std(axis=0)
        importance = self._compute_feature_importance(center_spread)
        
        return labels, None, importance
    
    def _hierarchical_cluster(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """Hierarchical clustering implementation."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            hc = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = hc.fit_predict(X)
            
            # Compute feature importance from cluster means
            importance = self._cluster_based_importance(X, labels)
            
            return labels, None, importance
            
        except ImportError:
            logger.warning("sklearn not available, falling back to kmeans")
            return self._simple_kmeans(X, n_clusters)
    
    def _spectral_cluster(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """Spectral clustering implementation."""
        try:
            from sklearn.cluster import SpectralClustering
            
            sc = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                affinity='nearest_neighbors',
                n_neighbors=min(10, X.shape[0] - 1)
            )
            labels = sc.fit_predict(X)
            
            importance = self._cluster_based_importance(X, labels)
            
            return labels, None, importance
            
        except ImportError:
            logger.warning("sklearn not available, falling back to kmeans")
            return self._simple_kmeans(X, n_clusters)
    
    def _gmm_cluster(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """Gaussian Mixture Model clustering implementation."""
        try:
            from sklearn.mixture import GaussianMixture
            
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=42,
                n_init=5
            )
            probs = gmm.fit_predict(X)
            labels = gmm.predict(X)
            prob_matrix = gmm.predict_proba(X)
            
            # Feature importance from component means
            means = gmm.means_
            importance = self._compute_feature_importance(means.std(axis=0))
            
            return labels, prob_matrix, importance
            
        except ImportError:
            logger.warning("sklearn not available, falling back to kmeans")
            return self._simple_kmeans(X, n_clusters)
    
    def _compute_feature_importance(
        self,
        scores: np.ndarray,
        top_n: int = 50
    ) -> Dict[str, float]:
        """Convert feature importance scores to gene-level dict."""
        if self._feature_names is None:
            return {}
        
        # Normalize scores
        scores_norm = scores / (scores.max() + 1e-10)
        
        # Get top features
        top_idx = np.argsort(scores_norm)[-top_n:]
        
        return {
            self._feature_names[i]: float(scores_norm[i])
            for i in top_idx
        }
    
    def _cluster_based_importance(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        top_n: int = 50
    ) -> Dict[str, float]:
        """Compute feature importance based on inter-cluster variance."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return {}
        
        # Compute between-cluster variance for each feature
        overall_mean = X.mean(axis=0)
        between_var = np.zeros(X.shape[1])
        
        for label in unique_labels:
            cluster_mean = X[labels == label].mean(axis=0)
            n_cluster = np.sum(labels == label)
            between_var += n_cluster * (cluster_mean - overall_mean) ** 2
        
        between_var /= X.shape[0]
        
        return self._compute_feature_importance(between_var, top_n)
    
    def detect_resistance_loops(
        self,
        stratification: StratificationResult,
        expression: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[ResistanceLoopSignature]:
        """
        Detect resistance loop signatures in patient clusters.
        
        Identifies clusters with elevated resistance marker propagation,
        particularly through DGAT1/YARS2-mediated pathways.
        
        Parameters
        ----------
        stratification : StratificationResult
            Results from stratify()
        expression : pd.DataFrame
            Original expression data
        threshold : float
            Threshold for resistance signature activation
            
        Returns
        -------
        List[ResistanceLoopSignature]
            Detected resistance loop signatures
        """
        if self._propagated_features is None:
            raise RuntimeError("Must compute propagated features first")
        
        # Map gene names to indices
        gene_to_idx = {g: i for i, g in enumerate(self._feature_names)}
        
        # Get indices for pathway genes (that exist in network)
        dgat1_idx = [gene_to_idx[g] for g in self.DGAT1_PATHWAY_GENES 
                     if g in gene_to_idx]
        yars2_idx = [gene_to_idx[g] for g in self.YARS2_PATHWAY_GENES 
                     if g in gene_to_idx]
        resistance_idx = [gene_to_idx[g] for g in self.RESISTANCE_LOOP_MARKERS 
                          if g in gene_to_idx]
        
        loops = []
        
        for cluster_id in range(stratification.n_clusters):
            cluster_mask = stratification.cluster_labels == cluster_id
            cluster_features = self._propagated_features[cluster_mask]
            
            # Compute mean propagated scores for each pathway
            dgat1_score = cluster_features[:, dgat1_idx].mean() if dgat1_idx else 0
            yars2_score = cluster_features[:, yars2_idx].mean() if yars2_idx else 0
            resistance_score = cluster_features[:, resistance_idx].mean() if resistance_idx else 0
            
            # Detect loops where pathway activation correlates with resistance
            loop_strength = (dgat1_score + yars2_score) * resistance_score
            patient_prevalence = cluster_mask.sum() / len(cluster_mask)
            
            if loop_strength > threshold:
                # Identify top propagated targets
                mean_scores = cluster_features.mean(axis=0)
                top_targets = [
                    self._feature_names[i] 
                    for i in np.argsort(mean_scores)[-10:]
                ]
                
                source_genes = []
                if dgat1_score > 0.3:
                    source_genes.extend(
                        [g for g in self.DGAT1_PATHWAY_GENES if g in gene_to_idx]
                    )
                if yars2_score > 0.3:
                    source_genes.extend(
                        [g for g in self.YARS2_PATHWAY_GENES if g in gene_to_idx]
                    )
                
                loop = ResistanceLoopSignature(
                    loop_id=f"loop_cluster{cluster_id}",
                    source_genes=source_genes,
                    propagated_targets=top_targets,
                    loop_strength=float(loop_strength),
                    is_dgat1_mediated=dgat1_score > 0.3,
                    is_yars2_mediated=yars2_score > 0.3,
                    patient_prevalence=float(patient_prevalence)
                )
                loops.append(loop)
                
                logger.info(
                    f"Detected resistance loop in cluster {cluster_id}: "
                    f"strength={loop_strength:.3f}, DGAT1={dgat1_score:.3f}, "
                    f"YARS2={yars2_score:.3f}"
                )
        
        return loops


def create_propagation_network(
    gene_interactions: pd.DataFrame,
    source_col: str = 'gene1',
    target_col: str = 'gene2',
    weight_col: Optional[str] = 'weight'
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Create sparse adjacency matrix from gene interaction data.
    
    Parameters
    ----------
    gene_interactions : pd.DataFrame
        DataFrame with gene-gene interactions
    source_col : str
        Column name for source genes
    target_col : str
        Column name for target genes
    weight_col : str, optional
        Column name for edge weights. If None, uses unweighted.
        
    Returns
    -------
    Tuple[sparse.csr_matrix, List[str]]
        Sparse adjacency matrix and ordered list of gene names
    """
    # Get all unique genes
    all_genes = set(gene_interactions[source_col]) | set(gene_interactions[target_col])
    gene_list = sorted(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    
    n_genes = len(gene_list)
    
    # Build sparse matrix
    rows = []
    cols = []
    data = []
    
    for _, row in gene_interactions.iterrows():
        g1, g2 = row[source_col], row[target_col]
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            weight = row[weight_col] if weight_col and weight_col in row else 1.0
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([weight, weight])
    
    adj = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_genes, n_genes)
    )
    
    logger.info(
        f"Created propagation network: {n_genes} genes, "
        f"{len(data)//2} edges"
    )
    
    return adj, gene_list


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic network (in practice, load from STRING/Pathway Commons)
    np.random.seed(42)
    
    # Simulate gene interaction network
    genes = ['DGAT1', 'YARS2', 'FASN', 'SCD', 'ABCB1', 'MGMT', 
             'ERCC1', 'SREBF1', 'LPIN1', 'MT-TY', 'MRPL12', 'GFM1',
             'TP53', 'BRCA1', 'EGFR', 'KRAS', 'MYC', 'AKT1', 'MTOR', 'PTEN']
    
    n_genes = len(genes)
    
    # Create random sparse adjacency
    rows, cols, data = [], [], []
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if np.random.random() < 0.3:  # 30% connectivity
                w = np.random.exponential(0.5)
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([w, w])
    
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_genes, n_genes))
    
    # Initialize propagator
    propagator = NetworkPropagator(
        adjacency_matrix=adj,
        node_names=genes,
        method=PropagationMethod.RANDOM_WALK_RESTART
    )
    
    # Test propagation from DGAT1
    seed = {'DGAT1': 1.0, 'FASN': 0.8, 'SCD': 0.6}
    result = propagator.propagate(seed, restart_prob=0.5)
    
    print("\n=== Propagation Results from DGAT1 seed ===")
    print(result.to_dataframe().head(10))
    
    # Create synthetic patient expression data
    n_patients = 50
    patient_expr = pd.DataFrame(
        np.random.randn(n_patients, n_genes) + np.random.choice([-1, 0, 1], n_genes),
        columns=genes,
        index=[f"Patient_{i}" for i in range(n_patients)]
    )
    
    # Add cluster structure
    patient_expr.loc[:15, ['DGAT1', 'FASN', 'ABCB1']] += 2  # Resistance cluster
    patient_expr.loc[15:30, ['YARS2', 'MRPL12', 'MGMT']] += 2  # Mitochondrial cluster
    
    # Initialize stratifier
    stratifier = PatientStratifier(propagator)
    
    # Compute propagated features
    features = stratifier.compute_propagated_features(
        patient_expr,
        seed_genes=['DGAT1', 'YARS2', 'FASN', 'MGMT', 'ABCB1'],
        restart_prob=0.4
    )
    
    # Stratify patients
    strat_result = stratifier.stratify(
        n_clusters=3,
        method=StratificationMethod.KMEANS
    )
    
    print("\n=== Patient Stratification Results ===")
    print(strat_result.to_dataframe())
    print(f"\nSilhouette Score: {strat_result.silhouette_score:.3f}")
    
    # Detect resistance loops
    loops = stratifier.detect_resistance_loops(
        strat_result,
        patient_expr,
        threshold=0.1  # Lower threshold for synthetic data
    )
    
    print("\n=== Detected Resistance Loops ===")
    for loop in loops:
        print(f"\n{loop.loop_id}:")
        print(f"  Strength: {loop.loop_strength:.3f}")
        print(f"  DGAT1-mediated: {loop.is_dgat1_mediated}")
        print(f"  YARS2-mediated: {loop.is_yars2_mediated}")
        print(f"  Source genes: {loop.source_genes[:5]}...")
        print(f"  Prevalence: {loop.patient_prevalence:.1%}")
```

---

## File: `brownbiotech/multiomics/__init__.py`

```python
"""
MultiOmics Agent v3.0 - Network Propagation & Stratification Submodule

This module provides graph-based analysis capabilities for multi-omics data,
integrating validated biological mechanisms (DGAT1/YARS2) into the ARP v3 pipeline.
"""

from .network_propagation import (
    NetworkPropagator,
    PatientStratifier,
    PropagationMethod,
    StratificationMethod,
    PropagationResult,
    StratificationResult,
    ResistanceLoopSignature,
    create_propagation_network,
)

__all__ = [
    'NetworkPropagator',
    'PatientStratifier',
    'PropagationMethod',
    'StratificationMethod',
    'PropagationResult',
    'StratificationResult',
    'ResistanceLoopSignature',
    'create_propagation_network',
]
```

---

## Improvement Summary

### What This Adds

| Component | Purpose |
|-----------|---------|
| `NetworkPropagator` | Implements RWR, Heat Diffusion, and PPR algorithms for spreading biological signals through interaction networks |
| `PatientStratifier` | Clusters patients based on propagated multi-omics features with multiple clustering backends |
| `ResistanceLoopSignature` | Dataclass capturing detected resistance mechanisms mediated by DGAT1/YARS2 pathways |
| `create_propagation_network` | Utility to build sparse adjacency matrices from interaction DataFrames |

### Key Design Decisions

1. **Sparse matrix operations** - Uses `scipy.sparse` for memory-efficient handling of large biological networks (10k+ genes)

2. **Graceful sklearn fallback** - Implements simple k-means when sklearn is unavailable, ensuring portability

3. **Mechanism-aware detection** - Hardcodes validated DGAT1/YARS2 pathway genes and resistance markers from Iteration 8 findings

4. **Type-safe containers** - Uses dataclasses for `PropagationResult`, `StratificationResult`, and `ResistanceLoopSignature` with proper type hints

5. **Convergence tracking** - All propagation methods track convergence history for debugging and quality assurance

### Integration Points

```python
# In your existing MultiOmics Agent
from brownbiotech.multiomics import (
    NetworkPropagator, 
    PatientStratifier,
    create_propagation_network
)

# Build network from your interaction database
adj, genes = create_propagation_network(interaction_df)

# Initialize and run
propagator = NetworkPropagator(adj, genes)
stratifier = PatientStratifier(propagator)
stratifier.compute_propagated_features(expression_matrix)
results = stratifier.stratify(n_clusters=4)
loops = stratifier.detect_resistance_loops(results, expression_matrix)
```