# BrownBioTech Iteration 18→19: Cross-Modal Retrieval & Metabolic Flux Integration

## Module 1: Cross-Modal Knowledge Graph Retriever

```python
"""
brownbiotech/agents/literature/cross_modal_retriever.py

Cross-modal retrieval engine for heterogeneous biomedical data sources.
Enables unified querying across literature, pathways, compounds, and genomic data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Protocol

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ModalType(Enum):
    """Supported data modalities in the knowledge graph."""
    LITERATURE = "literature"
    PATHWAY = "pathway"
    COMPOUND = "compound"
    GENE = "gene"
    PROTEIN = "protein"
    DISEASE = "disease"
    METABOLITE = "metabolite"
    CLINICAL_TRIAL = "clinical_trial"


class RelationType(Enum):
    """Edge types in the knowledge graph."""
    INTERACTS_WITH = "interacts_with"
    REGULATES = "regulates"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    PART_OF = "part_of"
    ASSOCIATED_WITH = "associated_with"
    METABOLIZES = "metabolizes"
    BINDS_TO = "binds_to"
    CAUSES = "causes"
    TREATS = "treats"


@dataclass
class EmbeddingVector:
    """Dense vector representation for semantic similarity."""
    values: np.ndarray
    dimension: int
    modality: ModalType
    model_name: str = "bio-bert-v1"

    def __post_init__(self):
        if len(self.values) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {len(self.values)} != {self.dimension}")

    def cosine_similarity(self, other: EmbeddingVector) -> float:
        """Compute cosine similarity between two embedding vectors."""
        if self.dimension != other.dimension:
            raise ValueError("Cannot compute similarity between different dimensions")
        norm_a = np.linalg.norm(self.values)
        norm_b = np.linalg.norm(other.values)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(self.values, other.values) / (norm_a * norm_b))

    def to_bytes(self) -> bytes:
        """Serialize embedding for storage."""
        return self.values.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, modality: ModalType, model_name: str = "bio-bert-v1") -> EmbeddingVector:
        """Deserialize embedding from storage."""
        values = np.frombuffer(data, dtype=np.float32)
        return cls(values=values, dimension=len(values), modality=modality, model_name=model_name)


class KGNode(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(..., description="Unique node identifier")
    modality: ModalType
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding: EmbeddingVector | None = None
    source: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("id")
    def validate_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Node ID cannot be empty")
        return v.strip()

    def compute_hash(self) -> str:
        """Compute deterministic hash for deduplication."""
        content = f"{self.modality.value}:{self.label}:{json.dumps(self.properties, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class KGEdge(BaseModel):
    """An edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    provenance: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_tuple(self) -> tuple[str, str, str, float]:
        return (self.source_id, self.target_id, self.relation.value, self.weight)


class RetrievalResult(BaseModel):
    """Result from cross-modal retrieval."""
    node: KGNode
    score: float = Field(ge=0.0, le=1.0)
    retrieval_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation backends."""
    async def embed_text(self, text: str, modality: ModalType) -> EmbeddingVector:
        ...

    async def embed_batch(self, texts: list[str], modality: ModalType) -> list[EmbeddingVector]:
        ...


class MockBioEmbeddingProvider:
    """Mock embedding provider for development/testing."""
    
    DEFAULT_DIM = 768

    async def embed_text(self, text: str, modality: ModalType) -> EmbeddingVector:
        """Generate deterministic pseudo-embedding based on text hash."""
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        values = rng.randn(self.DEFAULT_DIM).astype(np.float32)
        values = values / np.linalg.norm(values)
        return EmbeddingVector(
            values=values,
            dimension=self.DEFAULT_DIM,
            modality=modality,
            model_name="mock-bio-embedder"
        )

    async def embed_batch(self, texts: list[str], modality: ModalType) -> list[EmbeddingVector]:
        return [await self.embed_text(t, modality) for t in texts]


class CrossModalRetriever:
    """
    Core cross-modal retrieval engine.
    
    Enables unified semantic search across heterogeneous biomedical data sources
    by projecting all modalities into a shared embedding space.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.7,
        max_results: int = 50,
        cache_size: int = 10000
    ):
        self.embedding_provider = embedding_provider or MockBioEmbeddingProvider()
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # In-memory index (replace with vector DB in production)
        self._node_index: dict[str, KGNode] = {}
        self._edge_index: list[KGEdge] = []
        self._embedding_cache: dict[str, EmbeddingVector] = {}
        self._adjacency: dict[str, list[tuple[str, RelationType, float]]] = {}

    def add_node(self, node: KGNode) -> str:
        """Add a node to the knowledge graph index."""
        dedup_key = node.compute_hash()
        existing = self._find_by_hash(dedup_key)
        if existing:
            logger.debug(f"Node deduplicated: {node.label} -> {existing.id}")
            return existing.id
        
        self._node_index[node.id] = node
        self._adjacency.setdefault(node.id, [])
        
        if node.embedding:
            self._embedding_cache[node.id] = node.embedding
        
        logger.debug(f"Added {node.modality.value} node: {node.id}")
        return node.id

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the knowledge graph."""
        if edge.source_id not in self._node_index:
            logger.warning(f"Edge source not found: {edge.source_id}")
            return
        if edge.target_id not in self._node_index:
            logger.warning(f"Edge target not found: {edge.target_id}")
            return
        
        self._edge_index.append(edge)
        self._adjacency[edge.source_id].append((edge.target_id, edge.relation, edge.weight))
        # Bidirectional for undirected traversal
        self._adjacency.setdefault(edge.target_id, []).append(
            (edge.source_id, edge.relation, edge.weight)
        )

    def _find_by_hash(self, dedup_hash: str) -> KGNode | None:
        """Find existing node by deduplication hash."""
        for node in self._node_index.values():
            if node.compute_hash() == dedup_hash:
                return node
        return None

    async def _get_or_create_embedding(self, node: KGNode) -> EmbeddingVector:
        """Retrieve cached embedding or generate new one."""
        if node.id in self._embedding_cache:
            return self._embedding_cache[node.id]
        
        embed_text = f"{node.modality.value}: {node.label}"
        if node.properties.get("description"):
            embed_text += f" {node.properties['description']}"
        
        embedding = await self.embedding_provider.embed_text(embed_text, node.modality)
        self._embedding_cache[node.id] = embedding
        return embedding

    async def retrieve(
        self,
        query: str,
        target_modalities: list[ModalType] | None = None,
        min_confidence: float = 0.5,
        expand_neighbors: int = 0
    ) -> list[RetrievalResult]:
        """
        Perform cross-modal semantic retrieval.
        
        Args:
            query: Natural language query
            target_modalities: Filter results to specific modalities
            min_confidence: Minimum node confidence threshold
            expand_neighbors: Number of hops to expand in the graph
            
        Returns:
            Ranked list of retrieval results
        """
        query_embedding = await self.embedding_provider.embed_text(query, ModalType.LITERATURE)
        
        results: list[RetrievalResult] = []
        
        for node_id, node in self._node_index.items():
            if node.confidence < min_confidence:
                continue
            if target_modalities and node.modality not in target_modalities:
                continue
            
            node_embedding = await self._get_or_create_embedding(node)
            similarity = query_embedding.cosine_similarity(node_embedding)
            
            if similarity >= self.similarity_threshold:
                results.append(RetrievalResult(
                    node=node,
                    score=similarity,
                    retrieval_path=[node_id]
                ))
        
        # Graph expansion for context
        if expand_neighbors > 0:
            expanded = await self._expand_results(results, expand_neighbors, min_confidence)
            results.extend(expanded)
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:self.max_results]

    async def _expand_results(
        self,
        seed_results: list[RetrievalResult],
        hops: int,
        min_confidence: float
    ) -> list[RetrievalResult]:
        """Expand retrieval results via graph traversal."""
        expanded: list[RetrievalResult] = []
        visited: set[str] = {r.node.id for r in seed_results}
        
        current_frontier = [r.node.id for r in seed_results]
        
        for hop in range(hops):
            next_frontier: list[str] = []
            decay = 0.7 ** (hop + 1)
            
            for node_id in current_frontier:
                for neighbor_id, relation, weight in self._adjacency.get(node_id, []):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    
                    neighbor = self._node_index.get(neighbor_id)
                    if not neighbor or neighbor.confidence < min_confidence:
                        continue
                    
                    # Find best parent score for this neighbor
                    parent_scores = [r for r in seed_results if r.node.id == node_id]
                    parent_score = max((r.score for r in parent_scores), default=0.5)
                    
                    expanded.append(RetrievalResult(
                        node=neighbor,
                        score=parent_score * weight * decay,
                        retrieval_path=[node_id, neighbor_id],
                        metadata={"hop": hop + 1, "relation": relation.value}
                    ))
                    next_frontier.append(neighbor_id)
            
            current_frontier = next_frontier
        
        return expanded

    async def multi_hop_query(
        self,
        source_id: str,
        target_modality: ModalType,
        max_hops: int = 3,
        min_path_score: float = 0.3
    ) -> list[dict[str, Any]]:
        """
        Find paths between nodes of different modalities.
        
        Useful for queries like "Find compounds that interact with proteins
        associated with this disease."
        """
        if source_id not in self._node_index:
            raise ValueError(f"Source node not found: {source_id}")
        
        paths: list[dict[str, Any]] = []
        visited: set[str] = {source_id}
        queue: list[tuple[str, list[str], float]] = [(source_id, [source_id], 1.0)]
        
        while queue:
            current_id, path, cumulative_score = queue.pop(0)
            
            if len(path) > max_hops + 1:
                continue
            
            current_node = self._node_index[current_id]
            
            if current_node.modality == target_modality and len(path) > 1:
                if cumulative_score >= min_path_score:
                    paths.append({
                        "path": path,
                        "score": cumulative_score,
                        "target_node": current_node.dict()
                    })
                continue
            
            for neighbor_id, relation, weight in self._adjacency.get(current_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                
                new_score = cumulative_score * weight
                if new_score >= min_path_score:
                    queue.append((neighbor_id, path + [neighbor_id], new_score))
        
        paths.sort(key=lambda p: p["score"], reverse=True)
        return paths

    def get_subgraph(self, node_ids: list[str], max_edges: int = 100) -> dict[str, Any]:
        """Extract a subgraph containing specified nodes and their edges."""
        node_set = set(node_ids)
        edges = [
            e for e in self._edge_index
            if e.source_id in node_set and e.target_id in node_set
        ][:max_edges]
        
        return {
            "nodes": [self._node_index[nid].dict() for nid in node_ids if nid in self._node_index],
            "edges": [e.dict() for e in edges],
            "statistics": {
                "node_count": len(node_set),
                "edge_count": len(edges),
                "modalities": list({self._node_index[nid].modality.value 
                                   for nid in node_set if nid in self._node_index})
            }
        }

    @property
    def statistics(self) -> dict[str, Any]:
        """Return index statistics."""
        modality_counts: dict[str, int] = {}
        for node in self._node_index.values():
            modality_counts[node.modality.value] = modality_counts.get(node.modality.value, 0) + 1
        
        return {
            "total_nodes": len(self._node_index),
            "total_edges": len(self._edge_index),
            "cached_embeddings": len(self._embedding_cache),
            "modality_distribution": modality_counts
        }


# Convenience function for quick retrieval
async def quick_retrieve(
    query: str,
    nodes: list[KGNode],
    modalities: list[ModalType] | None = None,
    threshold: float = 0.6
) -> list[RetrievalResult]:
    """
    One-shot retrieval without persistent index.
    
    Useful for ad-hoc queries during agent reasoning.
    """
    retriever = CrossModalRetriever(similarity_threshold=threshold)
    for node in nodes:
        retriever.add_node(node)
    return await retriever.retrieve(query, target_modalities=modalities)


if __name__ == "__main__":
    async def demo():
        """Demonstrate cross-modal retrieval capabilities."""
        retriever = CrossModalRetriever(similarity_threshold=0.5)
        
        # Add heterogeneous nodes
        nodes = [
            KGNode(id="compound:statin1", modality=ModalType.COMPOUND, 
                   label="Atorvastatin", properties={"drug_class": "statin", "target": "HMGCR"}),
            KGNode(id="gene:hmgcr", modality=ModalType.GENE,
                   label="HMGCR", properties={"pathway": "cholesterol biosynthesis"}),
            KGNode(id="pathway:mevalonate", modality=ModalType.PATHWAY,
                   label="Mevalonate Pathway", properties={"description": "Cholesterol synthesis pathway"}),
            KGNode(id="disease:hyperchol", modality=ModalType.DISEASE,
                   label="Hypercholesterolemia", properties={"icd10": "E78.5"}),
            KGNode(id="metabolite:mevalonate", modality=ModalType.METABOLITE,
                   label="Mevalonic acid", properties={"kegg": "C00418"}),
        ]
        
        for node in nodes:
            retriever.add_node(node)
        
        # Add relationships
        edges = [
            KGEdge(source_id="compound:statin1", target_id="gene:hmgcr",
                   relation=RelationType.INHIBITS, weight=0.95),
            KGEdge(source_id="gene:hmgcr", target_id="pathway:mevalonate",
                   relation=RelationType.PART_OF, weight=0.9),
            KGEdge(source_id="pathway:mevalonate", target_id="disease:hyperchol",
                   relation=RelationType.ASSOCIATED_WITH, weight=0.85),
            KGEdge(source_id="metabolite:mevalonate", target_id="pathway:mevalonate",
                   relation=RelationType.PART_OF, weight=1.0),
        ]
        for edge in edges:
            retriever.add_edge(edge)
        
        # Cross-modal query
        results = await retriever.retrieve(
            "cholesterol lowering drug mechanism",
            expand_neighbors=1
        )
        
        print("=== Cross-Modal Retrieval Results ===")
        for r in results:
            print(f"  [{r.node.modality.value}] {r.node.label} (score={r.score:.3f})")
        
        # Multi-hop query
        paths = await retriever.multi_hop_query(
            source_id="compound:statin1",
            target_modality=ModalType.DISEASE,
            max_hops=3
        )
        
        print("\n=== Multi-Hop Paths (Compound → Disease) ===")
        for p in paths:
            print(f"  Path: {' → '.join(p['path'])} (score={p['score']:.3f})")
        
        print(f"\n=== Index Statistics ===")
        print(json.dumps(retriever.statistics, indent=2))
    
    asyncio.run(demo())
```

---

## Module 2: Knowledge Graph Builder

```python
"""
brownbiotech/agents/literature/knowledge_graph_builder.py

Constructs knowledge graphs from heterogeneous biomedical sources:
literature (PubMed), pathways (KEGG/Reactome), compounds (ChEMBL), and genomics.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote

import aiohttp

from .cross_modal_retriever import (
    CrossModalRetriever,
    KGEdge,
    KGNode,
    ModalType,
    RelationType,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from a single extraction pass."""
    nodes: list[KGNode]
    edges: list[KGEdge]
    source: str
    extraction_confidence: float


class TextEntityExtractor:
    """
    Extracts biomedical entities from unstructured text using
    pattern matching and dictionary lookup.
    """

    # Gene/protein patterns (HGNC-style)
    GENE_PATTERN = re.compile(
        r'\b([A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]+)?)\b'
    )
    
    # Compound patterns
    COMPOUND_PATTERN = re.compile(
        r'\b([A-Z][a-z]+(?:in|ol|one|ide|ate|ine|amine|azole|nib|vir|tin|lin|mycin|statin))\b'
    )
    
    # Disease patterns
    DISEASE_PATTERN = re.compile(
        r'\b((?:hereditary|familial|idiopathic|chronic|acute|severe|mild)?\s*(?:diabetes|cancer|hypertension|hypercholesterolemia|arthritis|Alzheimer|Parkinson|obesity|fibrosis))\b',
        re.IGNORECASE
    )

    def __init__(self, gene_dictionary: set[str] | None = None):
        self.gene_dictionary = gene_dictionary or set()
        self._compound_cache: set[str] = set()

    def add_gene_dictionary(self, genes: set[str]) -> None:
        """Add known gene symbols for improved extraction."""
        self.gene_dictionary.update(genes)

    def extract_entities(self, text: str, source: str = "") -> ExtractionResult:
        """Extract entities from text and return as extraction result."""
        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        
        # Extract genes
        gene_matches = set()
        for match in self.GENE_PATTERN.finditer(text):
            candidate = match.group(1)
            if len(candidate) >= 2 and (not self.gene_dictionary or candidate in self.gene_dictionary):
                gene_matches.add(candidate)
        
        for gene in gene_matches:
            nodes.append(KGNode(
                id=f"gene:{gene.lower()}",
                modality=ModalType.GENE,
                label=gene,
                source=source,
                confidence=0.8 if gene in self.gene_dictionary else 0.5
            ))
        
        # Extract compounds
        compound_matches = set()
        for match in self.COMPOUND_PATTERN.finditer(text):
            compound_matches.add(match.group(1))
        
        for compound in compound_matches:
            nodes.append(KGNode(
                id=f"compound:{compound.lower()}",
                modality=ModalType.COMPOUND,
                label=compound,
                source=source,
                confidence=0.6
            ))
        
        # Extract diseases
        disease_matches = set()
        for match in self.DISEASE_PATTERN.finditer(text):
            disease_matches.add(match.group(1).strip())
        
        for disease in disease_matches:
            nodes.append(KGNode(
                id=f"disease:{disease.lower().replace(' ', '_')}",
                modality=ModalType.DISEASE,
                label=disease,
                source=source,
                confidence=0.7
            ))
        
        # Infer relationships from co-occurrence patterns
        if len(gene_matches) > 1:
            genes = list(gene_matches)
            for i, g1 in enumerate(genes):
                for g2 in genes[i+1:]:
                    edges.append(KGEdge(
                        source_id=f"gene:{g1.lower()}",
                        target_id=f"gene:{g2.lower()}",
                        relation=RelationType.INTERACTS_WITH,
                        weight=0.4,
                        evidence=[f"Co-occurrence in {source}"],
                        confidence=0.4
                    ))
        
        # Compound-gene interactions
        for compound in compound_matches:
            for gene in gene_matches:
                edges.append(KGEdge(
                    source_id=f"compound:{compound.lower()}",
                    target_id=f"gene:{gene.lower()}",
                    relation=RelationType.BINDS_TO,
                    weight=0.3,
                    evidence=[f"Co-occurrence in {source}"],
                    confidence=0.3
                ))
        
        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            source=source,
            extraction_confidence=0.5
        )


class KnowledgeGraphBuilder:
    """
    Orchestrates knowledge graph construction from multiple data sources.
    
    Supports incremental building with deduplication and confidence scoring.
    """

    def __init__(
        self,
        retriever: CrossModalRetriever | None = None,
        entity_extractor: TextEntityExtractor | None = None,
        session_timeout: int = 30
    ):
        self.retriever = retriever or CrossModalRetriever()
        self.entity_extractor = entity_extractor or TextEntityExtractor()
        self._session: aiohttp.ClientSession | None = None
        self._session_timeout = session_timeout
        self._build_stats: dict[str, int] = {
            "nodes_added": 0,
            "edges_added": 0,
            "nodes_deduplicated": 0,
            "sources_processed": 0
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-initialize HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._session_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    def ingest_extraction(self, extraction: ExtractionResult) -> dict[str, int]:
        """
        Ingest an extraction result into the knowledge graph.
        
        Returns counts of added vs deduplicated items.
        """
        nodes_added = 0
        nodes_deduped = 0
        edges_added = 0
        
        for node in extraction.nodes:
            existing_id = self.retriever.add_node(node)
            if existing_id == node.id:
                nodes_added += 1
            else:
                nodes_deduped += 1
        
        for edge in extraction.edges:
            self.retriever.add_edge(edge)
            edges_added += 1
        
        self._build_stats["nodes_added"] += nodes_added
        self._build_stats["nodes_deduplicated"] += nodes_deduped
        self._build_stats["edges_added"] += edges_added
        self._build_stats["sources_processed"] += 1
        
        return {"nodes_added": nodes_added, "nodes_deduplicated": nodes_deduped, "edges_added": edges_added}

    def ingest_text(self, text: str, source: str = "") -> dict[str, int]:
        """Extract entities from text and ingest into KG."""
        extraction = self.entity_extractor.extract_entities(text, source)
        return self.ingest_extraction(extraction)

    async def ingest_pubmed_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str
    ) -> dict[str, int]:
        """Ingest a PubMed abstract into the knowledge graph."""
        # Add the literature node
        lit_node = KGNode(
            id=f"literature:pmid_{pmid}",
            modality=ModalType.LITERATURE,
            label=title[:200],
            properties={
                "pmid": pmid,
                "title": title,
                "abstract_length": len(abstract)
            },
            source="pubmed",
            confidence=1.0
        )
        self.retriever.add_node(lit_node)
        
        # Extract and ingest entities
        full_text = f"{title} {abstract}"
        result = self.ingest_text(full_text, source=f"pubmed:{pmid}")
        
        # Link extracted entities to the literature node
        for node in self.entity_extractor.extract_entities(full_text, f"pubmed:{pmid}").nodes:
            self.retriever.add_edge(KGEdge(
                source_id=f"literature:pmid_{pmid}",
                target_id=node.id,
                relation=RelationType.ASSOCIATED_WITH,
                weight=0.8,
                evidence=[f"Mentioned in PMID:{pmid}"],
                provenance="pubmed"
            ))
        
        return result

    async def ingest_kegg_pathway(self, pathway_id: str) -> dict[str, int]:
        """
        Ingest a KEGG pathway into the knowledge graph.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa00010")
        """
        session = await self._get_session()
        
        try:
            # Fetch pathway data from KEGG REST API
            url = f"https://rest.kegg.jp/get/{pathway_id}/kgml"
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"KEGG API returned {response.status} for {pathway_id}")
                    return {"nodes_added": 0, "nodes_deduplicated": 0, "edges_added": 0}
                
                kgml = await response.text()
            
            return self._parse_kegg_kgml(kgml, pathway_id)
            
        except Exception as e:
            logger.error(f"Failed to ingest KEGG pathway {pathway_id}: {e}")
            return {"nodes_added": 0, "nodes_deduplicated": 0, "edges_added": 0}

    def _parse_kegg_kgml(self, kgml: str, pathway_id: str) -> dict[str, int]:
        """Parse KGML format and extract nodes/edges."""
        nodes: list[KGNode] = []
        edges: list[KGEdge] = []
        
        try:
            root = ET.fromstring(kgml)
            
            # Extract pathway info
            pathway_name = root.get("name", pathway_id)
            pathway_title = root.get("title", pathway_id)
            
            nodes.append(KGNode(
                id=f"pathway:{pathway_id}",
                modality=ModalType.PATHWAY,
                label=pathway_title,
                properties={"kegg_id": pathway_id, "name": pathway_name},
                source="kegg",
                confidence=0.95
            ))
            
            # Extract entries (genes, compounds, etc.)
            entry_map: dict[str, tuple[str, str]] = {}
            
            for entry in root.findall(".//entry"):
                entry_id = entry.get("id", "")
                entry_name = entry.get("name", "")
                entry_type = entry.get("type", "")
                
                if entry_type == "gene":
                    gene_names = entry_name.split()
                    for gene in gene_names:
                        clean_gene = gene.split(":")[-1] if ":" in gene else gene
                        entry_map[entry_id] = (f"gene:{clean_gene.lower()}", clean_gene)
                        nodes.append(KGNode(
                            id=f"gene:{clean_gene.lower()}",
                            modality=ModalType.GENE,
                            label=clean_gene,
                            source="kegg",
                            confidence=0.9
                        ))
                        
                elif entry_type == "compound":
                    compound_id = entry_name.split(":")[-1] if ":" in entry_name else entry_name
                    entry_map[entry_id] = (f"compound:{compound_id.lower()}", compound_id)
                    nodes.append(KGNode(
                        id=f"compound:{compound_id.lower()}",
                        modality=ModalType.COMPOUND,
                        label=compound_id,
                        source="kegg",
                        confidence=0.9
                    ))
            
            # Extract relations
            for relation in root.findall(".//relation"):
                entry1 = relation.get("entry1", "")
                entry2 = relation.get("entry2", "")
                rel_type = relation.get("type", "")
                
                if entry1 in entry_map and entry2 in entry_map:
                    kg_relation = self._map_kegg_relation(rel_type)
                    edges.append(KGEdge(
                        source_id=entry_map[entry1][0],
                        target_id=entry_map[entry2][0],
                        relation=kg_relation,
                        weight=0.8,
                        evidence=[f"KEGG {pathway_id}"],
                        provenance="kegg",
                        confidence=0.85
                    ))
            
            # Link genes/compounds to pathway
            for entry_id, (node_id, _) in entry_map.items():
                edges.append(KGEdge(
                    source_id=node_id,
                    target_id=f"pathway:{pathway_id}",
                    relation=RelationType.PART_OF,
                    weight=0.9,
                    provenance="kegg"
                ))
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse KGML for {pathway_id}: {e}")
        
        extraction = ExtractionResult(
            nodes=nodes,
            edges=edges,
            source=f"kegg:{pathway_id}",
            extraction_confidence=0.85
        )
        
        return self.ingest_extraction(extraction)

    def _map_kegg_relation(self, kegg_type: str) -> RelationType:
        """Map KEGG relation types to internal relation types."""
        mapping = {
            "PPrel": RelationType.INTERACTS_WITH,
            "ECrel": RelationType.REGULATES,
            "GErel": RelationType.REGULATES,
            "maplink": RelationType.PART_OF,
            "compound": RelationType.METABOLIZES,
        }
        
        # Check for inhibition/activation subtypes
        if "inhibition" in kegg_type.lower():
            return RelationType.INHIBITS
        if "activation" in kegg_type.lower():
            return RelationType.ACTIVATES
        if "induction" in kegg_type.lower():
            return RelationType.ACTIVATES
        if "repression" in kegg_type.lower():
            return RelationType.INHIBITS
        
        return mapping.get(kegg_type, RelationType.ASSOCIATED_WITH)

    async def ingest_compound_data(
        self,
        compound_id: str,
        name: str,
        targets: list[str] | None = None,
        pathways: list[str] | None = None,
        properties: dict[str, Any] | None = None
    ) -> dict[str, int]:
        """Ingest compound data with target and pathway associations."""
        props = properties or {}
        props["name"] = name
        
        compound_node = KGNode(
            id=f"compound:{compound_id.lower()}",
            modality=ModalType.COMPOUND,
            label=name,
            properties=props,
            source="compound_db",
            confidence=0.9
        )
        
        nodes = [compound_node]
        edges = []
        
        # Add target associations
        for target in (targets or []):
            target_node = KGNode(
                id=f"gene:{target.lower()}",
                modality=ModalType.GENE,
                label=target,
                source="compound_db",
                confidence=0.85
            )
            nodes.append(target_node)
            edges.append(KGEdge(
                source_id=compound_node.id,
                target_id=target_node.id,
                relation=RelationType.BINDS_TO,
                weight=0.85,
                evidence=[f"Compound-target annotation for {compound_id}"],
                provenance="compound_db"
            ))
        
        # Add pathway associations
        for pathway in (pathways or []):
            pathway_node = KGNode(
                id=f"pathway:{pathway.lower()}",
                modality=ModalType.PATHWAY,
                label=pathway,
                source="compound_db",
                confidence=0.8
            )
            nodes.append(pathway_node)
            edges.append(KGEdge(
                source_id=compound_node.id,
                target_id=pathway_node.id,
                relation=RelationType.ASSOCIATED_WITH,
                weight=0.75,
                provenance="compound_db"
            ))
        
        extraction = ExtractionResult(
            nodes=nodes,
            edges=edges,
            source=f"compound:{compound_id}",
            extraction_confidence=0.85
        )
        
        return self.ingest_extraction(extraction)

    @property
    def build_statistics(self) -> dict[str, int]:
        """Return build statistics."""
        return {**self._build_stats, **self.retriever.statistics}


if __name__ == "__main__":
    async def demo():
        """Demonstrate knowledge graph building."""
        builder = KnowledgeGraphBuilder()
        
        # Ingest from text
        print("=== Ingesting Text ===")
        text = "Atorvastatin inhibits HMGCR, reducing cholesterol biosynthesis. " \
               "STAT3 regulates inflammation in chronic arthritis patients."
        result = builder.ingest_text(text, source="demo_text")
        print(f"  Added: {result}")
        
        # Ingest PubMed abstract
        print("\n=== Ingesting PubMed Abstract ===")
        result = await builder.ingest_pubmed_abstract(
            pmid="12345678",
            title="Novel HMGCR inhibitor shows promise in hypercholesterolemia treatment",
            abstract="We demonstrate that compound X-123 potently inhibits HMGCR "
                     "(IC50=2.3nM), leading to significant reduction in LDL cholesterol "
                     "in a phase II trial for familial hypercholesterolemia."
        )
        print(f"  Added: {result}")
        
        # Ingest compound data
        print("\n=== Ingesting Compound Data ===")
        result = await builder.ingest_compound_data(
            compound_id="x123",
            name="X-123",
            targets=["HMGCR", "CYP3A4"],
            pathways=["Mevalonate Pathway"],
            properties={"ic50_nm": 2.3, "phase": "II"}
        )
        print(f"  Added: {result}")
        
        # Show statistics
        print(f"\n=== Build Statistics ===")
        for k, v in builder.build_statistics.items():
            print(f"  {k}: {v}")
        
        await builder.close()
    
    asyncio.run(demo())
```

---

## Module 3: Metabolic Liability Scorer

```python
"""
brownbiotech/agents/literature/metabolic_liability_scorer.py

Predicts off-target metabolic liabilities for drug candidates by analyzing
metabolic pathways, CYP interactions, and reactive metabolite formation risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .cross_modal_retriever import (
    CrossModalRetriever,
    KGEdge,
    KGNode,
    ModalType,
    RelationType,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class LiabilityCategory(Enum):
    """Categories of metabolic liability."""
    CYP_INHIBITION = "cyp_inhibition"
    CYP_INDUCTION = "cyp_induction"
    REACTIVE_METABOLITE = "reactive_metabolite"
    TRANSPORTER_INHIBITION = "transporter_inhibition"
    OFF_TARGET_ENZYME = "off_target_enzyme"
    MITOCHONDRIAL_TOXICITY = "mitochondrial_toxicity"
    BILE_SALT_EXPORT_PUMP = "bile_salt_export_pump"
    GLUTATHIONE_DEPLETION = "glutathione_depletion"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StructuralAlert:
    """A structural feature associated with metabolic liability."""
    pattern: str
    category: LiabilityCategory
    risk_level: RiskLevel
    description: str
    weight: float = 1.0


# Known structural alerts for metabolic liability
STRUCTURAL_ALERTS = [
    StructuralAlert(
        pattern="C=CC=O",  # Alpha,beta-unsaturated carbonyl
        category=LiabilityCategory.REACTIVE_METABOLITE,
        risk_level=RiskLevel.HIGH,
        description="Michael acceptor - potential for glutathione conjugation"
    ),
    StructuralAlert(
        pattern="N=O",  # Nitro group
        category=LiabilityCategory.REACTIVE_METABOLITE,
        risk_level=RiskLevel.HIGH,
        description="Nitro group - potential for nitroso/nitro reduction to reactive species"
    ),
    StructuralAlert(
        pattern="c1ccc(cc1)N=NC=O",  # Aromatic amine + isocyanate
        category=LiabilityCategory.REACTIVE_METABOLITE,
        risk_level=RiskLevel.CRITICAL,
        description="Aromatic amine - potential for N-hydroxylation to reactive metabolite"
    ),
    StructuralAlert(
        pattern="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like acyl group
        category=LiabilityCategory.GLUTATHIONE_DEPLETION,
        risk_level=RiskLevel.MODERATE,
        description="Acyl group - potential for acyl glucuronide formation"
    ),
    StructuralAlert(
        pattern="Clc1ccccc1",  # Chloroaromatic
        category=LiabilityCategory.REACTIVE_METABOLITE,
        risk_level=RiskLevel.MODERATE,
        description="Chloroaromatic - potential for quinone imine formation"
    ),
    StructuralAlert(
        pattern="N(C)C=O",  # N,N-dimethylformamide-like
        category=LiabilityCategory.CYP_INHIBITION,
        risk_level=RiskLevel.MODERATE,
        description="Tertiary amide - potential CYP inhibition"
    ),
    StructuralAlert(
        pattern="SC",  # Thioether
        category=LiabilityCategory.REACTIVE_METABOLITE,
        risk_level=RiskLevel.MODERATE,
        description="Thioether - potential for sulfoxidation to reactive species"
    ),
    StructuralAlert(
        pattern="N#C",  # Nitrile
        category=LiabilityCategory.REACTIVE_METABOLITE,
        risk_level=RiskLevel.LOW,
        description="Nitrile - potential for cyanide release"
    ),
]


class LiabilityFinding(BaseModel):
    """A single metabolic liability finding."""
    category: LiabilityCategory
    risk_level: RiskLevel
    description: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float = 1.0
    source_nodes: list[str] = Field(default_factory=list)


class MetabolicLiabilityReport(BaseModel):
    """Complete metabolic liability assessment report."""
    compound_id: str
    compound_name: str
    overall_risk_score: float = Field(ge=0.0, le=1.0, description="0=low risk, 1=high risk")
    overall_risk_level: RiskLevel
    findings: list[LiabilityFinding] = Field(default_factory=list)
    cyp_interaction_profile: dict[str, float] = Field(
        default_factory=dict,
        description="CYP enzyme -> inhibition score mapping"
    )
    ddli_risk_partners: list[str] = Field(
        default_factory=list,
        description="Drug classes with potential DDLI risk"
    )
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetabolicLiabilityScorer:
    """
    Scores metabolic liabilities for drug candidates.
    
    Integrates structural alert analysis with knowledge graph-based
    pathway analysis to predict off-target metabolic risks.
    """

    # CYP enzyme node IDs in the KG
    CYP_ENZYMES = [
        "gene:cyp3a4", "gene:cyp2d6", "gene:cyp2c9", "gene:cyp2c19",
        "gene:cyp1a2", "gene:cyp2e1", "gene:cyp2b6"
    ]
    
    # Drug classes with known DDLI potential
    DDLI_RISK_CLASSES = {
        "statins": ["gene:cyp3a4"],
        "macrolides": ["gene:cyp3a4"],
        "azoles": ["gene:cyp3a4", "gene:cyp2c9", "gene:cyp2c19"],
        "ssris": ["gene:cyp2d6", "gene:cyp2c19"],
        "calcium_channel_blockers": ["gene:cyp3a4"],
        "proton_pump_inhibitors": ["gene:cyp2c19"],
    }

    def __init__(
        self,
        retriever: CrossModalRetriever | None = None,
        structural_alerts: list[StructuralAlert] | None = None,
        cyp_inhibition_threshold: float = 0.3,
        high_risk_threshold: float = 0.7,
        moderate_risk_threshold: float = 0.4
    ):
        self.retriever = retriever or CrossModalRetriever()
        self.structural_alerts = structural_alerts or STRUCTURAL_ALERTS
        self.cyp_inhibition_threshold = cyp_inhibition_threshold
        self.high_risk_threshold = high_risk_threshold
        self.moderate_risk_threshold = moderate_risk_threshold

    def _check_structural_alerts(self, smiles: str) -> list[LiabilityFinding]:
        """Check SMILES string against known structural alerts."""
        findings = []
        
        for alert in self.structural_alerts:
            if alert.pattern.lower() in smiles.lower():
                findings.append(LiabilityFinding(
                    category=alert.category,
                    risk_level=alert.risk_level,
                    description=alert.description,
                    evidence=[f"Structural alert matched: {alert.pattern}"],
                    confidence=0.75,
                    weight=alert.weight
                ))
        
        return findings

    async def _analyze_cyp_interactions(
        self,
        compound_id: str
    ) -> tuple[dict[str, float], list[LiabilityFinding]]:
        """Analyze CYP enzyme interactions via knowledge graph."""
        cyp_scores: dict[str, float] = {}
        findings: list[LiabilityFinding] = []
        
        for cyp_id in self.CYP_ENZYMES:
            # Check for direct inhibition edges
            cyp_node = self.retriever._node_index.get(cyp_id)
            if not cyp_node:
                continue
            
            cyp_name = cyp_id.split(":")[-1].upper()
            inhibition_score = 0.0
            
            # Check edges from compound to CYP
            for edge in self.retriever._edge_index:
                if (edge.source_id == compound_id and 
                    edge.target_id == cyp_id and
                    edge.relation in (RelationType.INHIBITS, RelationType.BINDS_TO)):
                    inhibition_score = max(inhibition_score, edge.weight * edge.confidence)
            
            if inhibition_score > 0:
                cyp_scores[cyp_name] = inhibition_score
                
                if inhibition_score >= 0.7:
                    risk_level = RiskLevel.HIGH
                elif inhibition_score >= self.cyp_inhibition_threshold:
                    risk_level = RiskLevel.MODERATE
                else:
                    risk_level = RiskLevel.LOW
                
                findings.append(LiabilityFinding(
                    category=LiabilityCategory.CYP_INHIBITION,
                    risk_level=risk_level,
                    description=f"Potential {cyp_name} inhibition (score={inhibition_score:.2f})",
                    evidence=[f"KG edge: {compound_id} -> {cyp_id}"],
                    confidence=inhibition_score,
                    weight=1.5  # CYP interactions weighted higher
                ))
        
        return cyp_scores, findings

    async def _analyze_pathway_liabilities(
        self,
        compound_id: str
    ) -> list[LiabilityFinding]:
        """Analyze pathway-based metabolic liabilities."""
        findings = []
        
        # Find pathways the compound is associated with
        pathway_edges = [
            e for e in self.retriever._edge_index
            if e.source_id == compound_id and 
               e.relation in (RelationType.ASSOCIATED_WITH, RelationType.PART_OF) and
               "pathway:" in e.target_id
        ]
        
        for edge in pathway_edges:
            pathway_node = self.retriever._node_index.get(edge.target_id)
            if not pathway_node:
                continue
            
            pathway_name = pathway_node.label.lower()
            
            # Check for high-risk pathway associations
            high_risk_keywords = ["glutathione", "reactive", "quinone", "epoxide", "radical"]
            for keyword in high_risk_keywords:
                if keyword in pathway_name:
                    findings.append(LiabilityFinding(
                        category=LiabilityCategory.REACTIVE_METABOLITE,
                        risk_level=RiskLevel.HIGH,
                        description=f"Compound associated with {pathway_name}",
                        evidence=[f"Pathway association: {pathway_node.id}"],
                        confidence=0.7,
                        weight=1.2
                    ))
                    break
            
            # Check for mitochondrial pathways
            mito_keywords = ["mitochondri", "oxidative phosphorylation", "etc", "respiratory"]
            for keyword in mito_keywords:
                if keyword in pathway_name:
                    findings.append(LiabilityFinding(
                        category=LiabilityCategory.MITOCHONDRIAL_TOXICITY,
                        risk_level=RiskLevel.MODERATE,
                        description=f"Compound associated with mitochondrial pathway: {pathway_name}",
                        evidence=[f"Pathway association: {pathway_node.id}"],
                        confidence=0.6,
                        weight=1.3
                    ))
                    break
        
        return findings

    def _compute_ddli_risk_partners(
        self,
        cyp_scores: dict[str, float]
    ) -> list[str]:
        """Identify drug classes at risk for DDLI based on CYP profile."""
        risk_partners = []
        
        for drug_class, target_cyps in self.DDLI_RISK_CLASSES.items():
            for cyp_id in target_cyps:
                cyp_name = cyp_id.split(":")[-1].upper()
                if cyp_name in cyp_scores and cyp_scores[cyp_name] >= self.cyp_inhibition_threshold:
                    risk_partners.append(drug_class)
                    break
        
        return list(set(risk_partners))

    def _generate_recommendations(
        self,
        findings: list[LiabilityFinding],
        cyp_scores: dict[str, float],
        ddli_partners: list[str]
    ) -> list[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        # CYP-related recommendations
        high_risk_cyps = [cyp for cyp, score in cyp_scores.items() if score >= 0.7]
        if high_risk_cyps:
            recommendations.append(
                f"Strong {', '.join(high_risk_cyps)} inhibition detected. "
                f"Consider structural modification to reduce CYP affinity."
            )
        
        # DDLI recommendations
        if ddli_partners:
            recommendations.append(
                f"Potential DDLI risk with: {', '.join(ddli_partners)}. "
                f"Conduct in vitro CYP inhibition panel (IC50 determination)."
            )
        
        # Reactive metabolite recommendations
        reactive_findings = [f for f in findings if f.category == LiabilityCategory.REACTIVE_METABOLITE]
        if any(f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) for f in reactive_findings):
            recommendations.append(
                "Reactive metabolite risk identified. Conduct glutathione trapping "
                "studies and consider bioisosteric replacement of reactive moiety."
            )
        
        # Mitochondrial toxicity
        mito_findings = [f for f in findings if f.category == LiabilityCategory.MITOCHONDRIAL_TOXICITY]
        if mito_findings:
            recommendations.append(
                "Mitochondrial toxicity risk flagged. Evaluate mitochondrial membrane "
                "potential and oxygen consumption rate in hepatocytes."
            )
        
        if not recommendations:
            recommendations.append("No significant metabolic liabilities identified at this stage.")
        
        return recommendations

    def _compute_overall_risk(
        self,
        findings: list[LiabilityFinding]
    ) -> tuple[float, RiskLevel]:
        """Compute weighted overall risk score."""
        if not findings:
            return 0.0, RiskLevel.LOW
        
        total_weight = sum(f.weight for f in findings)
        weighted_score = sum(
            f.confidence * f.weight * self._risk_level_multiplier(f.risk_level)
            for f in findings
        ) / total_weight
        
        # Cap at 1.0
        weighted_score = min(weighted_score, 1.0)
        
        if weighted_score >= self.high_risk_threshold:
            return weighted_score, RiskLevel.HIGH
        elif weighted_score >= self.moderate_risk_threshold:
            return weighted_score, RiskLevel.MODERATE
        else:
            return weighted_score, RiskLevel.LOW

    def _risk_level_multiplier(self, level: RiskLevel) -> float:
        """Convert risk level to numeric multiplier."""
        return {
            RiskLevel.LOW: 0.3,
            RiskLevel.MODERATE: 0.6,
            RiskLevel.HIGH: 0.85,
            RiskLevel.CRITICAL: 1.0
        }[level]

    async def score_compound(
        self,
        compound_id: str,
        compound_name: str,
        smiles: str | None = None,
        include_structural_alerts: bool = True
    ) -> MetabolicLiabilityReport:
        """
        Generate a complete metabolic liability assessment.
        
        Args:
            compound_id: Unique compound identifier
            compound_name: Human-readable compound name
            smiles: SMILES string for structural analysis (optional)
            include_structural_alerts: Whether to run structural alert analysis
            
        Returns:
            Complete liability report
        """
        all_findings: list[LiabilityFinding] = []
        
        # Structural alert analysis
        if include_structural_alerts and smiles:
            structural_findings = self._check_structural_alerts(smiles)
            all_findings.extend(structural_findings)
        
        # CYP interaction analysis
        cyp_scores, cyp_findings = await self._analyze_cyp_interactions(compound_id)
        all_findings.extend(cyp_findings)
        
        # Pathway liability analysis
        pathway_findings = await self._analyze_pathway_liabilities(compound_id)
        all_findings.extend(pathway_findings)
        
        # Compute DDLI risk partners
        ddli_partners = self._compute_ddli_risk_partners(cyp_scores)
        
        # Compute overall risk
        overall_score, overall_level = self._compute_overall_risk(all_findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_findings, cyp_scores, ddli_partners)
        
        return MetabolicLiabilityReport(
            compound_id=compound_id,
            compound_name=compound_name,
            overall_risk_score=overall_score,
            overall_risk_level=overall_level,
            findings=all_findings,
            cyp_interaction_profile=cyp_scores,
            ddli_risk_partners=ddli_partners,
            recommendations=recommendations,
            metadata={
                "smiles_analyzed": smiles is not None,
                "structural_alerts_checked": include_structural_alerts,
                "kg_nodes_available": len(self.retriever._node_index)
            }
        )

    async def batch_score(
        self,
        compounds: list[tuple[str, str, str | None]]
    ) -> list[MetabolicLiabilityReport]:
        """Score multiple compounds in batch."""
        reports = []
        for compound_id, compound_name, smiles in compounds:
            report = await self.score_compound(compound_id, compound_name, smiles)
            reports.append(report)
        return reports


if __name__ == "__main__":
    async def demo():
        """Demonstrate metabolic liability scoring."""
        from .cross_modal_retriever import KGEdge, KGNode, ModalType, RelationType, CrossModalRetriever
        
        # Setup retriever with test data
        retriever = CrossModalRetriever()
        
        # Add a test compound
        compound = KGNode(
            id="compound:test_drug",
            modality=ModalType.COMPOUND,
            label="Test Drug X-200",
            properties={"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
        )
        retriever.add_node(compound)
        
        # Add CYP nodes
        for cyp in ["CYP3A4", "CYP2D6", "CYP2C9"]:
            node = KGNode(
                id=f"gene:{cyp.lower()}",
                modality=ModalType.GENE,
                label=cyp,
                properties={"type": "cytochrome_p450"}
            )
            retriever.add_node(node)
        
        # Add inhibition edges
        retriever.add_edge(KGEdge(
            source_id="compound:test_drug",
            target_id="gene:cyp3a4",
            relation=RelationType.INHIBITS,
            weight=0.85,
            evidence=["In vitro IC50 = 0.5 μM"]
        ))
        retriever.add_edge(KGEdge(
            source_id="compound:test_drug",
            target_id="gene:cyp2d6",
            relation=RelationType.BINDS_TO,
            weight=0.4,
            evidence=["Moderate binding affinity"]
        ))
        
        # Score the compound
        scorer = MetabolicLiabilityScorer(retriever=retriever)
        report = await scorer.score_compound(
            compound_id="compound:test_drug",
            compound_name="Test Drug X-200",
            smiles="CC(=O)Oc1ccccc1C(=O)O"  # Aspirin-like
        )
        
        print("=== Metabolic Liability Report ===")
        print(f"Compound: {report.compound_name}")
        print(f"Overall Risk: {report.overall_risk_level.value} (score={report.overall_risk_score:.3f})")
        print(f"\nCYP Profile:")
        for cyp, score in report.cyp_interaction_profile.items():
            print(f"  {cyp}: {score:.2f}")
        print(f"\nDDLI Risk Partners: {', '.join(report.ddli_risk_partners) or 'None'}")
        print(f"\nFindings:")
        for f in report.findings:
            print(f"  [{f.risk_level.value}] {f.category.value}: {f.description}")
        print(f"\nRecommendations:")
        for r in report.recommendations:
            print(f"  • {r}")
    
    asyncio.run(demo())
```

---

## Module 4: Metabolic Flux Constraint Engine

```python
"""
brownbiotech/agents/literature/metabolic_flux_engine.py

Constraint-based metabolic flux analysis engine for predicting
metabolic pathway impacts of drug interventions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FluxStatus(Enum):
    """Status of a flux constraint."""
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SUBOPTIMAL = "suboptimal"


class ReactionDirection(Enum):
    """Directionality of metabolic reactions."""
    FORWARD = 1
    REVERSE = -1
    IRREVERSIBLE_FORWARD = 2
    IRREVERSIBLE_REVERSE = -2


@dataclass
class Metabolite:
    """A metabolite in the metabolic network."""
    id: str
    name: str
    compartment: str = "cytosol"
    formula: str = ""
    charge: int = 0
    is_boundary: bool = False


@dataclass
class Reaction:
    """A metabolic reaction in the network."""
    id: str
    name: str
    equation: str
    metabolites: dict[str, float] = field(default_factory=dict)  # metabolite_id -> stoichiometry
    lower_bound: float = -1000.0
    upper_bound: float = 1000.0
    direction: ReactionDirection = ReactionDirection.FORWARD
    gene_association: str = ""
    subsystem: str = ""
    is_exchange: bool = False
    
    @property
    def is_reversible(self) -> bool:
        return self.direction in (ReactionDirection.FORWARD, ReactionDirection.REVERSE)


@dataclass
class FluxConstraint:
    """A constraint on reaction flux."""
    reaction_id: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    fixed_value: float | None = None
    penalty_weight: float = 1.0
    description: str = ""


class FluxSolution(BaseModel):
    """Solution from flux balance analysis."""
    status: FluxStatus
    objective_value: float
    reaction_fluxes: dict[str, float] = Field(default_factory=dict)
    shadow_prices: dict[str, float] = Field(default_factory=dict)
    reduced_costs: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DrugIntervention(BaseModel):
    """A drug intervention affecting metabolic flux."""
    drug_id: str
    drug_name: str
    target_reactions: list[str] = Field(default_factory=list)
    inhibition_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    description: str = ""


class PathwayImpact(BaseModel):
    """Impact analysis of drug on a metabolic pathway."""
    pathway_id: str
    pathway_name: str
    baseline_flux: float
    perturbed_flux: float
    flux_change: float
    percent_change: float
    affected_reactions: list[str] = Field(default_factory=list)
    risk_level: str = "low"


class MetabolicNetwork(BaseModel):
    """A complete metabolic network model."""
    id: str
    name: str
    metabolites: list[Metabolite] = Field(default_factory=list)
    reactions: list[Reaction] = Field(default_factory=list)
    objective_reaction: str = ""
    compartments: list[str] = Field(default_factory=lambda: ["cytosol", "mitochondria", "extracellular"])
    
    def get_metabolite(self, mid: str) -> Metabolite | None:
        for m in self.metabolites:
            if m.id == mid:
                return m
        return None
    
    def get_reaction(self, rid: str) -> Reaction | None:
        for r in self.reactions:
            if r.id == rid:
                return r
        return None
    
    def get_reactions_by_subsystem(self, subsystem: str) -> list[Reaction]:
        return [r for r in self.reactions if r.subsystem == subsystem]
    
    def get_exchange_reactions(self) -> list[Reaction]:
        return [r for r in self.reactions if r.is_exchange]


class MetabolicFluxEngine:
    """
    Constraint-based metabolic flux analysis engine.
    
    Implements simplified FBA (Flux Balance Analysis) for predicting
    metabolic pathway impacts of drug interventions.
    """

    def __init__(
        self,
        network: MetabolicNetwork,
        solver_tolerance: float = 1e-6,
        max_iterations: int = 1000
    ):
        self.network = network
        self.solver_tolerance = solver_tolerance
        self.max_iterations = max_iterations
        self._stoich_matrix: np.ndarray | None = None
        self._metabolite_order: list[str] = []
        self._reaction_order: list[str] = []

    def _build_stoichiometric_matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Build the stoichiometric matrix S for the network."""
        if self._stoich_matrix is not None:
            return self._stoich_matrix, self._metabolite_order, self._reaction_order
        
        # Get non-boundary metabolites
        internal_metabolites = [m for m in self.network.metabolites if not m.is_boundary]
        self._metabolite_order = [m.id for m in internal_metabolites]
        self._reaction_order = [r.id for r in self.network.reactions]
        
        n_metabolites = len(self._metabolite_order)
        n_reactions = len(self._reaction_order)
        
        S = np.zeros((n_metabolites, n_reactions))
        
        metabolite_idx = {mid: i for i, mid in enumerate(self._metabolite_order)}
        reaction_idx = {rid: j for j, rid in enumerate(self._reaction_order)}
        
        for reaction in self.network.reactions:
            if reaction.id not in reaction_idx:
                continue
            j = reaction_idx[reaction.id]
            for metabolite_id, stoich in reaction.metabolites.items():
                if metabolite_id in metabolite_idx:
                    i = metabolite_idx[metabolite_id]
                    S[i, j] = stoich
        
        self._stoich_matrix = S
        return S, self._metabolite_order, self._reaction_order

    def _get_bounds(self, constraints: list[FluxConstraint] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds for all reactions."""
        _, _, reaction_order = self._build_stoichiometric_matrix()
        n = len(reaction_order)
        
        lb = np.zeros(n)
        ub = np.zeros(n)
        
        constraint_map = {}
        if constraints:
            for c in constraints:
                constraint_map[c.reaction_id] = c
        
        for j, rid in enumerate(reaction_order):
            reaction = self.network.get_reaction(rid)
            if reaction is None:
                lb[j] = 0.0
                ub[j] = 0.0
                continue
            
            if rid in constraint_map:
                c = constraint_map[rid]
                if c.fixed_value is not None:
                    lb[j] = c.fixed_value
                    ub[j] = c.fixed_value
                else:
                    lb[j] = c.lower_bound if c.lower_bound is not None else reaction.lower_bound
                    ub[j] = c.upper_bound if c.upper_bound is not None else reaction.upper_bound
            else:
                lb[j] = reaction.lower_bound
                ub[j] = reaction.upper_bound
        
        return lb, ub

    def solve_fba(
        self,
        objective_reaction: str | None = None,
        constraints: list[FluxConstraint] | None = None,
        maximize: bool = True
    ) -> FluxSolution:
        """
        Solve Flux Balance Analysis using iterative simplex-like approach.
        
        This is a simplified implementation. In production, use a proper
        LP solver (e.g., scipy.optimize.linprog, gurobipy, or cobra).
        """
        S, metabolite_order, reaction_order = self._build_stoichiometric_matrix()
        lb, ub = self._get_bounds(constraints)
        
        obj_reaction = objective_reaction or self.network.objective_reaction
        if obj_reaction not in reaction_order:
            return FluxSolution(
                status=FluxStatus.INFEASIBLE,
                objective_value=0.0,
                metadata={"error": f"Objective reaction {obj_reaction} not found"}
            )
        
        obj_idx = reaction_order.index(obj_reaction)
        
        # Build objective vector
        c = np.zeros(len(reaction_order))
        c[obj_idx] = -1.0 if maximize else 1.0
        
        try:
            from scipy.optimize import linprog
            
            result = linprog(
                c,
                A_eq=S,
                b_eq=np.zeros(S.shape[0]),
                bounds=list(zip(lb, ub)),
                method='highs',
                options={'maxiter': self.max_iterations, 'tol': self.solver_tolerance}
            )
            
            if result.success:
                fluxes = dict(zip(reaction_order, result.x))
                obj_value = -result.fun if maximize else result.fun
                
                return FluxSolution(
                    status=FluxStatus.OPTIMAL,
                    objective_value=obj_value,
                    reaction_fluxes=fluxes,
                    metadata={"solver": "scipy-highs", "iterations": result.nit}
                )
            else:
                return FluxSolution(
                    status=FluxStatus.INFEASIBLE,
                    objective_value=0.0,
                    metadata={"error": result.message}
                )
                
        except ImportError:
            logger.warning("scipy not available, using fallback solver")
            return self._fallback_fba(S, lb, ub, reaction_order, obj_idx, maximize)
        except Exception as e:
            return FluxSolution(
                status=FluxStatus.INFEASIBLE,
                objective_value=0.0,
                metadata={"error": str(e)}
            )

    def _fallback_fba(
        self,
        S: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        reaction_order: list[str],
        obj_idx: int,
        maximize: bool
    ) -> FluxSolution:
        """Simple fallback FBA when scipy is not available."""
        # Simple heuristic: try to maximize objective while satisfying constraints
        n_reactions = len(reaction_order)
        fluxes = np.zeros(n_reactions)
        
        # Start with mean of bounds
        for j in range(n_reactions):
            if lb[j] <= 0 <= ub[j]:
                fluxes[j] = 0.0
            elif lb[j] > 0:
                fluxes[j] = lb[j]
            else:
                fluxes[j] = ub[j]
        
        # Iteratively adjust to satisfy S*v = 0
        for iteration in range(self.max_iterations):
            residual = S @ fluxes
            if np.max(np.abs(residual)) < self.solver_tolerance:
                break
            
            # Simple gradient correction
            correction = np.linalg.lstsq(S, -residual, rcond=None)[0]
            fluxes = fluxes + 0.5 * correction
            
            # Clip to bounds
            fluxes = np.clip(fluxes, lb, ub)
        
        obj_value = fluxes[obj_idx]
        
        return FluxSolution(
            status=FluxStatus.SUBOPTIMAL,
            objective_value=obj_value,
            reaction_fluxes=dict(zip(reaction_order, fluxes)),
            metadata={"solver": "fallback", "iterations": iteration}
        )

    def apply_drug_intervention(
        self,
        intervention: DrugIntervention
    ) -> list[FluxConstraint]:
        """
        Generate flux constraints from a drug intervention.
        
        Models enzyme inhibition as reduced upper bound on target reactions.
        """
        constraints = []
        
        for reaction_id in intervention.target_reactions:
            reaction = self.network.get_reaction(reaction_id)
            if reaction is None:
                logger.warning(f"Target reaction not found: {reaction_id}")
                continue
            
            # Reduce upper bound based on inhibition strength
            original_ub = reaction.upper_bound
            new_ub = original_ub * (1.0 - intervention.inhibition_strength)
            
            constraints.append(FluxConstraint(
                reaction_id=reaction_id,
                upper_bound=new_ub,
                penalty_weight=intervention.inhibition_strength,
                description=f"{intervention.drug_name} inhibits {reaction.name} "
                           f"({intervention.inhibition_strength*100:.0f}% inhibition)"
            ))
        
        return constraints

    def analyze_pathway_impact(
        self,
        intervention: DrugIntervention,
        subsystems: list[str] | None = None
    ) -> list[PathwayImpact]:
        """
        Analyze the impact of a drug intervention on metabolic pathways.
        """
        # Get baseline solution
        baseline = self.solve_fba()
        if baseline.status != FluxStatus.OPTIMAL:
            logger.warning(f"Baseline FBA not optimal: {baseline.metadata}")
        
        # Apply intervention constraints
        intervention_constraints = self.apply_drug_intervention(intervention)
        perturbed = self.solve_fba(constraints=intervention_constraints)
        
        # Analyze subsystem impacts
        target_subsystems = subsystems or list({
            r.subsystem for r in self.network.reactions if r.subsystem
        })
        
        impacts = []
        for subsystem in target_subsystems:
            reactions = self.network.get_reactions_by_subsystem(subsystem)
            if not reactions:
                continue
            
            reaction_ids = [r.id for r in reactions]
            
            baseline_flux = sum(
                abs(baseline.reaction_fluxes.get(rid, 0.0)) for rid in reaction_ids
            )
            perturbed_flux = sum(
                abs(perturbed.reaction_fluxes.get(rid, 0.0)) for rid in reaction_ids
            )
            
            flux_change = perturbed_flux - baseline_flux
            percent_change = (flux_change / baseline_flux * 100) if baseline_flux > 0 else 0.0
            
            # Determine risk level
            if abs(percent_change) > 50:
                risk_level = "high"
            elif abs(percent_change) > 20:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            # Find affected reactions
            affected = [
                rid for rid in reaction_ids
                if abs(baseline.reaction_fluxes.get(rid, 0) - perturbed.reaction_fluxes.get(rid, 0)) > self.solver_tolerance
            ]
            
            impacts.append(PathwayImpact(
                pathway_id=subsystem.lower().replace(" ", "_"),
                pathway_name=subsystem,
                baseline_flux=baseline_flux,
                perturbed_flux=perturbed_flux,
                flux_change=flux_change,
                percent_change=percent_change,
                affected_reactions=affected,
                risk_level=risk_level
            ))
        
        impacts.sort(key=lambda p: abs(p.percent_change), reverse=True)
        return impacts

    def compute_flexibility(
        self,
        reaction_id: str,
        constraints: list[FluxConstraint] | None = None
    ) -> dict[str, float]:
        """
        Compute flux variability for a reaction (min/max flux range).
        """
        # Maximize
        max_solution = self.solve_fba(
            objective_reaction=reaction_id,
            constraints=constraints,
            maximize=True
        )
        
        # Minimize
        min_solution = self.solve_fba(
            objective_reaction=reaction_id,
            constraints=constraints,
            maximize=False
        )
        
        return {
            "reaction_id": reaction_id,
            "minimum_flux": min_solution.objective_value,
            "maximum_flux": max_solution.objective_value,
            "range": max_solution.objective_value - min_solution.objective_value,
            "is_blocked": abs(max_solution.objective_value) < self.solver_tolerance and 
                          abs(min_solution.objective_value) < self.solver_tolerance
        }


def build_sample_network() -> MetabolicNetwork:
    """Build a sample metabolic network for testing."""
    metabolites = [
        Metabolite(id="glc__D_e", name="D-Glucose (extracellular)", compartment="extracellular", is_boundary=True),
        Metabolite(id="glc__D_c", name="D-Glucose (cytosol)", compartment="cytosol"),
        Metabolite(id="g6p_c", name="Glucose-6-phosphate", compartment="cytosol"),
        Metabolite(id="f6p_c", name="Fructose-6-phosphate", compartment="cytosol"),
        Metabolite(id="fdp_c", name="Fructose-1,6-bisphosphate", compartment="cytosol"),
        Metabolite(id="dhap_c", name="Dihydroxyacetone phosphate", compartment="cytosol"),
        Metabolite(id="g3p_c", name="Glyceraldehyde-3-phosphate", compartment="cytosol"),
        Metabolite(id="pyr_c", name="Pyruvate", compartment="cytosol"),
        Metabolite(id="ac_c", name="Acetate", compartment="cytosol"),
        Metabolite(id="ac_e", name="Acetate (extracellular)", compartment="extracellular", is_boundary=True),
        Metabolite(id="atp_c", name="ATP", compartment="cytosol"),
        Metabolite(id="adp_c", name="ADP", compartment="cytosol"),
        Metabolite(id="nadh_c", name="NADH", compartment="cytosol"),
        Metabolite(id="nad_c", name="NAD+", compartment="cytosol"),
        Metabolite(id="h_c", name="H+", compartment="cytosol"),
        Metabolite(id="h2o_c", name="H2O", compartment="cytosol"),
        Metabolite(id="pi_c", name="Phosphate", compartment="cytosol"),
    ]
    
    reactions = [
        Reaction(
            id="GLCt", name="Glucose transport",
            equation="glc__D_e -> glc__D_c",
            metabolites={"glc__D_e": -1, "glc__D_c": 1},
            lower_bound=-10, upper_bound=10,
            is_exchange=True
        ),
        Reaction(
            id="HEX1", name="Hexokinase",
            equation="glc__D_c + atp_c -> g6p_c + adp_c + h_c",
            metabolites={"glc__D_c": -1, "atp_c": -1, "g6p_c": 1, "adp_c": 1, "h_c": 1},
            lower_bound=0, upper_bound=1000,
            gene_association="HK1 or HK2 or HK3",
            subsystem="Glycolysis/Gluconeogenesis"
        ),
        Reaction(
            id="PGI", name="Phosphoglucose isomerase",
            equation="g6p_c -> f6p_c",
            metabolites={"g6p_c": -1, "f6p_c": 1},
            lower_bound=-1000, upper_bound=1000,
            subsystem="Glycolysis/Gluconeogenesis"
        ),
        Reaction(
            id="PFK", name="Phosphofructokinase",
            equation="f6p_c + atp_c -> fdp_c + adp_c + h_c",
            metabolites={"f6p_c": -1, "atp_c": -1, "fdp_c": 1, "adp_c": 1, "h_c": 1},
            lower_bound=0, upper_bound=1000,
            gene_association="PFKL or PFKM or PFKP",
            subsystem="Glycolysis/Gluconeogenesis"
        ),
        Reaction(
            id="FBA", name="Fructose-bisphosphate aldolase",
            equation="fdp_c -> dhap_c + g3p_c",
            metabolites={"fdp_c": -1, "dhap_c": 1, "g3p_c": 1},
            lower_bound=-1000, upper_bound=1000,
            subsystem="Glycolysis/Gluconeogenesis"
        ),
        Reaction(
            id="PYK", name="Pyruvate kinase",
            equation="g3p_c + nad_c + adp_c + pi_c -> pyr_c + nadh_c + atp_c + h_c",
            metabolites={"g3p_c": -2, "nad_c": -2, "adp_c": -2, "pi_c": -2, 
                        "pyr_c": 2, "nadh_c": 2, "atp_c": 2, "h_c": 4},
            lower_bound=0, upper_bound=1000,
            gene_association="PKLR or PKM",
            subsystem="Glycolysis/Gluconeogenesis"
        ),
        Reaction(
            id="ACt", name="Acetate transport",
            equation="ac_c -> ac_e",
            metabolites={"ac_c": -1, "ac_e": 1},
            lower_bound=0, upper_bound=1000,
            is_exchange=True
        ),
        Reaction(
            id="ACt2", name="Acetate production",
            equation="pyr_c -> ac_c + h_c",
            metabolites={"pyr_c": -1, "ac_c": 1, "h_c": 1},
            lower_bound=0, upper_bound=1000,
            subsystem="Fermentation"
        ),
        # ATP maintenance
        Reaction(
            id="ATPM", name="ATP maintenance",
            equation="atp_c + h2o_c -> adp_c + pi_c + h_c",
            metabolites={"atp_c": -1, "h2o_c": -1, "adp_c": 1, "pi_c": 1, "h_c": 1},
            lower_bound=1, upper_bound=1,
            subsystem="Energy metabolism"
        ),
    ]
    
    return MetabolicNetwork(
        id="sample_glycolysis",
        name="Sample Glycolysis Network",
        metabolites=metabolites,
        reactions=reactions,
        objective_reaction="ACt"
    )


if __name__ == "__main__":
    def demo():
        """Demonstrate metabolic flux analysis."""
        network = build_sample_network()
        engine = MetabolicFluxEngine(network)
        
        print("=== Baseline FBA ===")
        baseline = engine.solve_fba()
        print(f"Status: {baseline.status.value}")
        print(f"Objective (acetate production): {baseline.objective_value:.4f}")
        print("\nReaction fluxes:")
        for rid, flux in sorted(baseline.reaction_fluxes.items()):
            if abs(flux) > 1e-6:
                print(f"  {rid}: {flux:.4f}")
        
        print("\n=== Drug Intervention: PFK Inhibitor ===")
        intervention = DrugIntervention(
            drug_id="drug:pfk_inhibitor",
            drug_name="PFK-Inhib-X",
            target_reactions=["PFK"],
            inhibition_strength=0.8,
            description="Novel PFK inhibitor reducing glycolytic flux"
        )
        
        constraints = engine.apply_drug_intervention(intervention)
        print(f"Generated {len(constraints)} constraints")
        for c in constraints:
            print(f"  {c.reaction_id}: ub <= {c.upper_bound:.2f}")
        
        perturbed = engine.solve_fba(constraints=constraints)
        print(f"\nPerturbed objective: {perturbed.objective_value:.4f}")
        print(f"Flux reduction: {(1 - perturbed.objective_value/baseline.objective_value)*100:.1f}%")
        
        print("\n=== Pathway Impact Analysis ===")
        impacts = engine.analyze_pathway_impact(intervention)
        for impact in impacts:
            print(f"  [{impact.risk_level.upper()}] {impact.pathway_name}")
            print(f"    Baseline: {impact.baseline_flux:.4f} -> Perturbed: {impact.perturbed_flux:.4f}")
            print(f"    Change: {impact.percent_change:+.1f}%")
        
        print("\n=== Flux Variability ===")
        flexibility = engine.compute_flexibility("HEX1")
        print(f"  HEX1: [{flexibility['minimum_flux']:.4f}, {flexibility['maximum_flux']:.4f}]")
        print(f"  Blocked: {flexibility['is_blocked']}")
    
    demo()
```

---

## Module 5: Literature Agent Integration

```python
"""
brownbiotech/agents/literature/literature_agent.py

Enhanced literature agent integrating cross-modal retrieval,
knowledge graph construction, and metabolic liability analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .cross_modal_retriever import (
    CrossModalRetriever,
    KGNode,
    ModalType,
    RetrievalResult,
)
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .metabolic_flux_engine import (
    DrugIntervention,
    MetabolicFluxEngine,
    MetabolicNetwork,
    PathwayImpact,
)
from .metabolic_liability_scorer import (
    MetabolicLiabilityReport,
    MetabolicLiabilityScorer,
)

logger = logging.getLogger(__name__)


@dataclass
class LiteratureQuery:
    """A structured query to the literature agent."""
    query_text: str
    target_modalities: list[ModalType] | None = None
    include_metabolic_analysis: bool = False
    compound_smiles: str | None = None
    max_results: int = 20


@dataclass
class LiteratureResponse:
    """Response from the literature agent."""
    query: str
    retrieval_results: list[RetrievalResult]
    subgraph: dict[str, Any] | None = None
    liability_report: MetabolicLiabilityReport | None = None
    pathway_impacts: list[PathwayImpact] | None = None
    summary: str = ""
    metadata: dict[str, Any] | None = None


class LiteratureAgent:
    """
    Enhanced literature agent with integrated cross-modal retrieval
    and metabolic analysis capabilities.
    
    Provides a unified interface for:
    - Cross-modal knowledge retrieval
    - Knowledge graph construction from literature
    - Metabolic liability scoring
    - Flux-based pathway impact analysis
    """

    def __init__(
        self,
        retriever: CrossModalRetriever | None = None,
        kg_builder: KnowledgeGraphBuilder | None = None,
        liability_scorer: MetabolicLiabilityScorer | None = None,
        flux_engine: MetabolicFluxEngine | None = None,
        default_similarity_threshold: float = 0.6
    ):
        self.retriever = retriever or CrossModalRetriever(
            similarity_threshold=default_similarity_threshold
        )
        self.kg_builder = kg_builder or KnowledgeGraphBuilder(retriever=self.retriever)
        self.liability_scorer = liability_scorer or MetabolicLiabilityScorer(
            retriever=self.retriever
        )
        self.flux_engine = flux_engine
        self._query_history: list[LiteratureQuery] = []

    async def query(self, query: LiteratureQuery) -> LiteratureResponse:
        """
        Process a literature query with optional metabolic analysis.
        """
        self._query_history.append(query)
        metadata = {"query_type": "standard"}
        
        # Cross-modal retrieval
        retrieval_results = await self.retriever.retrieve(
            query.query_text,
            target_modalities=query.target_modalities,
            expand_neighbors=1
        )
        
        # Extract subgraph if results found
        subgraph = None
        if retrieval_results:
            node_ids = [r.node.id for r in retrieval_results[:10]]
            subgraph = self.retriever.get_subgraph(node_ids)
        
        # Metabolic liability analysis
        liability_report = None
        if query.include_metabolic_analysis:
            # Try to find compound in results
            compound_nodes = [
                r for r in retrieval_results
                if r.node.modality == ModalType.COMPOUND
            ]
            
            if compound_nodes:
                best_compound = max(compound_nodes, key=lambda r: r.score)
                liability_report = await self.liability_scorer.score_compound(
                    compound_id=best_compound.node.id,
                    compound_name=best_compound.node.label,
                    smiles=query.compound_smiles
                )
                metadata["liability_compound"] = best_compound.node.label
        
        # Pathway impact analysis
        pathway_impacts = None
        if query.include_metabolic_analysis and self.flux_engine:
            pathway_impacts = await self._analyze_pathway_impacts(retrieval_results)
        
        # Generate summary
        summary = self._generate_summary(
            retrieval_results, liability_report, pathway_impacts
        )
        
        return LiteratureResponse(
            query=query.query_text,
            retrieval_results=retrieval_results[:query.max_results],
            subgraph=subgraph,
            liability_report=liability_report,
            pathway_impacts=pathway_impacts,
            summary=summary,
            metadata=metadata
        )

    async def _analyze_pathway_impacts(
        self,
        results: list[RetrievalResult]
    ) -> list[PathwayImpact] | None:
        """Analyze pathway impacts based on retrieval results."""
        if not self.flux_engine:
            return None
        
        # Find drug-related nodes
        drug_nodes = [
            r for r in results
            if r.node.modality == ModalType.COMPOUND and r.score > 0.7
        ]
        
        if not drug_nodes:
            return None
        
        # Create intervention from top drug
        top_drug = max(drug_nodes, key=lambda r: r.score)
        
        # Find target reactions via KG
        target_reactions = self._find_target_reactions(top_drug.node.id)
        if not target_reactions:
            return None
        
        intervention = DrugIntervention(
            drug_id=top_drug.node.id,
            drug_name=top_drug.node.label,
            target_reactions=target_reactions,
            inhibition_strength=0.7
        )
        
        return self.flux_engine.analyze_pathway_impact(intervention)

    def _find_target_reactions(self, compound_id: str) -> list[str]:
        """Find metabolic reactions associated with a compound via KG."""
        reactions = []
        
        for edge in self.retriever._edge_index:
            if edge.source_id == compound_id and "reaction:" in edge.target_id:
                reactions.append(edge.target_id.replace("reaction:", ""))
        
        return reactions

    def _generate_summary(
        self,
        results: list[RetrievalResult],
        liability: MetabolicLiabilityReport | None,
        impacts: list[PathwayImpact] | None
    ) -> str:
        """Generate a human-readable summary of findings."""
        parts = []
        
        if results:
            modality_counts: dict[str, int] = {}
            for r in results:
                mod = r.node.modality.value
                modality_counts[mod] = modality_counts.get(mod, 0) + 1
            
            parts.append(
                f"Found {len(results)} relevant results across "
                f"{len(modality_counts)} modalities: "
                + ", ".join(f"{k}({v})" for k, v in modality_counts.items())
            )
            
            top_result = results[0]
            parts.append(f"Top match: {top_result.node.label} (score={top_result.score:.3f})")
        
        if liability:
            parts.append(
                f"Metabolic liability: {liability.overall_risk_level.value} risk "
                f"(score={liability.overall_risk_score:.3f})"
            )
            if liability.findings:
                high_risk = [f for f in liability.findings if f.risk_level.value in ("high", "critical")]
                if high_risk:
                    parts.append(f"  {len(high_risk)} high-risk findings identified")
        
        if impacts:
            high_impact = [p for p in impacts if p.risk_level == "high"]
            if high_impact:
                parts.append(
                    f"Pathway analysis: {len(high_impact)} pathways with high impact"
                )
        
        return "\n".join(parts)

    async def ingest_literature_batch(
        self,
        abstracts: list[tuple[str, str, str]]
    ) -> dict[str, int]:
        """
        Ingest a batch of literature abstracts.
        
        Args:
            abstracts: List of (pmid, title, abstract) tuples
        """
        total = {"nodes_added": 0, "edges_added": 0, "sources_processed": 0}
        
        for pmid, title, abstract in abstracts:
            result = await self.kg_builder.ingest_pubmed_abstract(pmid, title, abstract)
            for k in total:
                total[k] += result.get(k, 0)
        
        logger.info(f"Ingested {len(abstracts)} abstracts: {total}")
        return total

    def set_metabolic_network(self, network: MetabolicNetwork) -> None:
        """Set the metabolic network for flux analysis."""
        self.flux_engine = MetabolicFluxEngine(network)

    @property
    def statistics(self) -> dict[str, Any]:
        """Return agent statistics."""
        return {
            "retriever": self.retriever.statistics,
            "kg_builder": self.kg_builder.build_statistics,
            "queries_processed": len(self._query_history),
            "flux_engine_available": self.flux_engine is not None
        }


if __name__ == "__main__":
    async def demo():
        """Demonstrate integrated literature agent."""
        from .metabolic_flux_engine import build_sample_network
        
        # Initialize agent with metabolic network
        agent = LiteratureAgent()
        agent.set_metabolic_network(build_sample_network())
        
        # Ingest some literature
        print("=== Ingesting Literature ===")
        abstracts = [
            ("PMID001", "HMGCR inhibition reduces cholesterol synthesis",
             "Atorvastatin potently inhibits HMG-CoA reductase, blocking the "
             "mevalonate pathway and reducing cholesterol biosynthesis."),
            ("PMID002", "CYP3A4 interactions with statins",
             "Statins metabolized by CYP3A4 show increased risk of drug-drug "
             "interactions when co-administered with macrolide antibiotics."),
            ("PMID003", "Glycolysis inhibition in cancer therapy",
             "PFK inhibitors show promise in reducing glycolytic flux in cancer "
             "cells, leading to decreased lactate production and tumor growth."),
        ]
        stats = await agent.ingest_literature_batch(abstracts)
        print(f"Ingestion stats: {stats}")
        
        # Query with metabolic analysis
        print("\n=== Query: Statin Mechanism ===")
        response = await agent.query(LiteratureQuery(
            query_text="statin cholesterol mechanism HMGCR",
            include_metabolic_analysis=True,
            compound_smiles="CC(C)C1=C(C(=C(N1)C)C)C"
        ))
        print(response.summary)
        
        print("\n=== Query: Glycolysis Inhibition ===")
        response = await agent.query(LiteratureQuery(
            query_text="glycolysis inhibition cancer therapy",
            include_metabolic_analysis=True
        ))
        print(response.summary)
        
        if response.pathway_impacts:
            print("\nPathway Impacts:")
            for impact in response.pathway_impacts[:3]:
                print(f"  {impact.pathway_name}: {impact.percent_change:+.1f}%")
        
        print(f"\n=== Agent Statistics ===")
        for k, v in agent.statistics.items():
            print(f"  {k}: {v}")
        
        await agent.kg_builder.close()
    
    asyncio.run(demo())
```

---

## Summary of Improvements

| Module | File | Purpose |
|--------|------|---------|
| **Cross-Modal Retriever** | `cross_modal_retriever.py` | Unified semantic search across literature, pathways, compounds, genes, diseases, metabolites |
| **Knowledge Graph Builder** | `knowledge_graph_builder.py` | Constructs KG from PubMed abstracts, KEGG pathways, and compound databases |
| **Metabolic Liability Scorer** | `metabolic_liability_scorer.py` | Predicts off-target risks: CYP inhibition, reactive metabolites, DDLI potential |
| **Metabolic Flux Engine** | `metabolic_flux_engine.py` | Constraint-based FBA for pathway impact prediction under drug intervention |
| **Literature Agent** | `literature_agent.py` | Unified interface integrating all modules for end-to-end analysis |

### Key Design Decisions:
1. **Modular architecture** - Each module can be used independently or composed
2. **Async-first** - All I/O operations are async for scalability
3. **Pydantic models** - Type-safe data structures with validation
4. **Mock providers** - Development without external dependencies
5. **Fallback solvers** - Works without scipy using heuristic methods
6. **Deduplication** - Hash-based entity deduplication during KG construction