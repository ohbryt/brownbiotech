# BrownBioTech Iteration 8/100: Multi-Modal Literature-Graph Integration

## File Structure
```
arp_v3/agents/literature/
├── graph_extractor.py
├── metabolic_linker.py
└── embeddings/
    ├── text_embedder.py
    └── omics_fusion.py
```

---

## 1. `arp_v3/agents/literature/graph_extractor.py`

```python
"""
Knowledge Graph Extractor for Literature Analysis.

Extracts entities and relationships from biomedical text to build
knowledge graphs for DGAT1/YARS2 pathway analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class EntityType(Enum):
    """Biomedical entity types relevant to BrownBioTech focus."""
    GENE = "gene"
    PROTEIN = "protein"
    METABOLITE = "metabolite"
    DISEASE = "disease"
    PATHWAY = "pathway"
    DRUG = "drug"
    TISSUE = "tissue"
    CELL_TYPE = "cell_type"
    PROCESS = "biological_process"


class RelationType(Enum):
    """Relationship types for knowledge graph edges."""
    INTERACTS_WITH = "interacts_with"
    REGULATES = "regulates"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    ASSOCIATED_WITH = "associated_with"
    EXPRESSED_IN = "expressed_in"
    MUTATED_IN = "mutated_in"


@dataclass
class Entity:
    """Represents a biomedical entity extracted from text."""
    text: str
    entity_type: EntityType
    normalized_id: Optional[str] = None  # e.g., NCBI Gene ID, ChEBI ID
    confidence: float = 1.0
    context: str = ""
    source: str = ""
    
    def __hash__(self) -> int:
        return hash((self.text.lower(), self.entity_type))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return (self.text.lower() == other.text.lower() and 
                self.entity_type == other.entity_type)


@dataclass
class Relation:
    """Represents a relationship between two entities."""
    source: Entity
    target: Entity
    relation_type: RelationType
    confidence: float = 1.0
    evidence: str = ""
    source_doc: str = ""
    
    def __hash__(self) -> int:
        return hash((self.source, self.target, self.relation_type))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.relation_type == other.relation_type)


@dataclass
class ExtractedGraph:
    """Container for extracted knowledge graph data."""
    entities: set[Entity] = field(default_factory=set)
    relations: set[Relation] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_networkx(self) -> Optional["nx.DiGraph"]:
        """Convert to NetworkX directed graph if available."""
        if not HAS_NETWORKX:
            return None
        
        G = nx.DiGraph()
        for entity in self.entities:
            G.add_node(
                entity.text.lower(),
                entity_type=entity.entity_type.value,
                normalized_id=entity.normalized_id,
                confidence=entity.confidence,
                context=entity.context
            )
        for rel in self.relations:
            G.add_edge(
                rel.source.text.lower(),
                rel.target.text.lower(),
                relation_type=rel.relation_type.value,
                confidence=rel.confidence,
                evidence=rel.evidence,
                source_doc=rel.source_doc
            )
        return G


# Curated patterns for BrownBioTech focus areas
FOCUS_GENES = {
    "dgat1": {"id": "DGAT1", "ncbi": "8694"},
    "yars2": {"id": "YARS2", "ncbi": "55221"},
    "dgat2": {"id": "DGAT2", "ncbi": "84649"},
    "pgc1a": {"id": "PPARGC1A", "ncbi": "10891"},
    "mt-co1": {"id": "MT-CO1", "ncbi": "4512"},
    "tfam": {"id": "TFAM", "ncbi": "7019"},
}

METABOLITE_PATTERNS = {
    "triacylglycerol": ["triacylglycerol", "triglyceride", "tag", "tg"],
    "diacylglycerol": ["diacylglycerol", "dag"],
    "acyl-coa": ["acyl-coa", "acyl coenzyme a", "acyl coa"],
    "phosphatidic acid": ["phosphatidic acid", "pa"],
    "atp": ["atp", "adenosine triphosphate"],
    "tyrosine": ["tyrosine", "l-tyrosine", "tyr"],
}

RELATION_PATTERNS = {
    RelationType.REGULATES: [
        r"(\w+)\s+(?:regulates?|controls?|modulates?)\s+(\w+)",
        r"(\w+)\s+(?:upregulat|downregulat)\w+\s+(\w+)",
    ],
    RelationType.INHIBITS: [
        r"(\w+)\s+(?:inhibits?|suppresses?|blocks?)\s+(\w+)",
        r"(\w+)\s+(?:knockdown|knockout|silencing)\s+(?:of\s+)?(\w+)",
    ],
    RelationType.ACTIVATES: [
        r"(\w+)\s+(?:activates?|stimulates?|enhances?)\s+(\w+)",
        r"(\w+)\s+(?:overexpress|upregulat)\w+\s+(\w+)",
    ],
    RelationType.PRODUCES: [
        r"(\w+)\s+(?:produces?|synthesiz\w+|generates?)\s+(\w+)",
        r"(\w+)\s+(?:catalyz\w+|convert\w+)\s+.*?\s+to\s+(\w+)",
    ],
    RelationType.ASSOCIATED_WITH: [
        r"(\w+)\s+(?:associated|linked|correlated)\s+(?:with|to)\s+(\w+)",
        r"(\w+)\s+(?:mutation|variant)\s+(?:in|of)\s+(\w+)",
    ],
    RelationType.EXPRESSED_IN: [
        r"(\w+)\s+(?:expressed|localized|enriched)\s+in\s+(\w+)",
    ],
}


class KnowledgeGraphExtractor:
    """
    Extracts knowledge graphs from biomedical literature text.
    
    Uses pattern-based extraction with confidence scoring for
    DGAT1/YARS2 focused pathway analysis.
    """
    
    def __init__(
        self,
        focus_genes: Optional[dict[str, dict]] = None,
        metabolite_patterns: Optional[dict[str, list[str]]] = None,
        min_confidence: float = 0.3,
    ):
        self.focus_genes = focus_genes or FOCUS_GENES
        self.metabolite_patterns = metabolite_patterns or METABOLITE_PATTERNS
        self.min_confidence = min_confidence
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> dict[RelationType, list[re.Pattern]]:
        """Pre-compile regex patterns for performance."""
        compiled = {}
        for rel_type, patterns in RELATION_PATTERNS.items():
            compiled[rel_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def extract_from_text(
        self,
        text: str,
        source_id: str = "",
    ) -> ExtractedGraph:
        """
        Extract entities and relations from a single text passage.
        
        Args:
            text: Input biomedical text (abstract, paragraph, etc.)
            source_id: Identifier for the source document
            
        Returns:
            ExtractedGraph with entities and relations
        """
        graph = ExtractedGraph(metadata={"source": source_id})
        
        # Extract entities
        entities = self._extract_entities(text, source_id)
        graph.entities.update(entities)
        
        # Extract relations
        relations = self._extract_relations(text, entities, source_id)
        graph.relations.update(relations)
        
        # Filter by confidence
        graph.entities = {
            e for e in graph.entities if e.confidence >= self.min_confidence
        }
        graph.relations = {
            r for r in graph.relations if r.confidence >= self.min_confidence
        }
        
        return graph
    
    def _extract_entities(
        self,
        text: str,
        source: str,
    ) -> set[Entity]:
        """Extract biomedical entities from text."""
        entities = set()
        text_lower = text.lower()
        
        # Extract focus genes
        for gene_key, gene_info in self.focus_genes.items():
            # Check for both symbol and full name patterns
            patterns = [rf"\b{re.escape(gene_key)}\b", 
                       rf"\b{re.escape(gene_info['id'])}\b"]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    entities.add(Entity(
                        text=gene_info["id"],
                        entity_type=EntityType.GENE,
                        normalized_id=gene_info.get("ncbi"),
                        confidence=0.95,
                        context=self._get_context(text, gene_info["id"]),
                        source=source
                    ))
                    break
        
        # Extract metabolites
        for metabolite, patterns in self.metabolite_patterns.items():
            for pattern in patterns:
                if re.search(rf"\b{re.escape(pattern)}\b", text_lower):
                    entities.add(Entity(
                        text=metabolite,
                        entity_type=EntityType.METABOLITE,
                        confidence=0.8,
                        context=self._get_context(text, pattern),
                        source=source
                    ))
                    break
        
        # Extract diseases (simple pattern)
        disease_patterns = [
            r"(?:mitochondrial|metabolic|lipid)\s+(?:disease|disorder|dysfunction)",
            r"(?:obesity|diabetes|steatosis|hepatosteatosis|nash)",
            r"(?:cardiomyopathy|myopathy|neuropathy)",
        ]
        for pattern in disease_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entities.add(Entity(
                    text=match.group(0).title(),
                    entity_type=EntityType.DISEASE,
                    confidence=0.7,
                    context=self._get_context(text, match.group(0)),
                    source=source
                ))
        
        return entities
    
    def _extract_relations(
        self,
        text: str,
        entities: set[Entity],
        source: str,
    ) -> set[Relation]:
        """Extract relationships between entities from text."""
        relations = set()
        
        # Build lookup for quick entity matching
        entity_lookup = {e.text.lower(): e for e in entities}
        # Add gene keys to lookup
        for gene_key, gene_info in self.focus_genes.items():
            entity_lookup[gene_key] = Entity(
                text=gene_info["id"],
                entity_type=EntityType.GENE,
                normalized_id=gene_info.get("ncbi")
            )
        
        for rel_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    source_text = match.group(1).lower()
                    target_text = match.group(2).lower()
                    
                    source_entity = entity_lookup.get(source_text)
                    target_entity = entity_lookup.get(target_text)
                    
                    if source_entity and target_entity:
                        confidence = self._calculate_relation_confidence(
                            match.group(0), rel_type
                        )
                        relations.add(Relation(
                            source=source_entity,
                            target=target_entity,
                            relation_type=rel_type,
                            confidence=confidence,
                            evidence=match.group(0),
                            source_doc=source
                        ))
        
        return relations
    
    def _calculate_relation_confidence(
        self,
        evidence: str,
        rel_type: RelationType,
    ) -> float:
        """Calculate confidence score for an extracted relation."""
        confidence = 0.6  # Base confidence
        
        # Boost for explicit language
        explicit_markers = ["directly", "specifically", "significantly"]
        for marker in explicit_markers:
            if marker in evidence.lower():
                confidence += 0.15
                break
        
        # Boost for focus gene involvement
        for gene_key in self.focus_genes:
            if gene_key in evidence.lower():
                confidence += 0.1
                break
        
        # Penalize for uncertain language
        uncertain_markers = ["may", "might", "could", "potentially", "suggest"]
        for marker in uncertain_markers:
            if marker in evidence.lower():
                confidence -= 0.2
                break
        
        return max(0.0, min(1.0, confidence))
    
    def _get_context(self, text: str, term: str, window: int = 100) -> str:
        """Extract surrounding context for an entity mention."""
        idx = text.lower().find(term.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(term) + window)
        return text[start:end].strip()
    
    def merge_graphs(self, graphs: list[ExtractedGraph]) -> ExtractedGraph:
        """Merge multiple extracted graphs into one."""
        merged = ExtractedGraph()
        for graph in graphs:
            merged.entities.update(graph.entities)
            merged.relations.update(graph.relations)
            merged.metadata[f"source_{len(merged.metadata)}"] = graph.metadata.get("source", "")
        return merged
    
    def get_focus_subgraph(
        self,
        graph: ExtractedGraph,
        focus_entities: list[str],
        max_depth: int = 2,
    ) -> ExtractedGraph:
        """
        Extract a subgraph centered on focus entities.
        
        Args:
            graph: Source knowledge graph
            focus_entities: Entity names to center on
            max_depth: Maximum hop distance from focus entities
            
        Returns:
            Filtered ExtractedGraph
        """
        if HAS_NETWORKX:
            nx_graph = graph.to_networkx()
            if nx_graph is None:
                return graph
            
            # Find focus nodes
            focus_nodes = set()
            for entity in graph.entities:
                if entity.text.lower() in [f.lower() for f in focus_entities]:
                    focus_nodes.add(entity.text.lower())
            
            # BFS to find neighborhood
            neighborhood = set()
            for node in focus_nodes:
                neighborhood.add(node)
                current_level = {node}
                for _ in range(max_depth):
                    next_level = set()
                    for n in current_level:
                        next_level.update(nx_graph.predecessors(n))
                        next_level.update(nx_graph.successors(n))
                    neighborhood.update(next_level)
                    current_level = next_level
            
            # Filter entities and relations
            subgraph = ExtractedGraph()
            for entity in graph.entities:
                if entity.text.lower() in neighborhood:
                    subgraph.entities.add(entity)
            
            for rel in graph.relations:
                if (rel.source.text.lower() in neighborhood and 
                    rel.target.text.lower() in neighborhood):
                    subgraph.relations.add(rel)
            
            return subgraph
        
        return graph


# Convenience function for quick extraction
def extract_graph_from_text(text: str, source_id: str = "") -> ExtractedGraph:
    """Quick extraction wrapper for single-text use cases."""
    extractor = KnowledgeGraphExtractor()
    return extractor.extract_from_text(text, source_id)


if __name__ == "__main__":
    # Demo/test
    sample_text = """
    DGAT1 directly regulates triacylglycerol synthesis in hepatocytes by 
    converting diacylglycerol and acyl-CoA to triacylglycerol. DGAT1 
    inhibition significantly reduces lipid accumulation in metabolic 
    dysfunction-associated steatotic liver disease. YARS2 mutations are 
    associated with mitochondrial dysfunction and cardiomyopathy. YARS2 
    is expressed in skeletal muscle and regulates mitochondrial translation.
    """
    
    extractor = KnowledgeGraphExtractor()
    graph = extractor.extract_from_text(sample_text, "demo_pubmed_123")
    
    print(f"Extracted {len(graph.entities)} entities:")
    for e in graph.entities:
        print(f"  - [{e.entity_type.value}] {e.text} (conf: {e.confidence:.2f})")
    
    print(f"\nExtracted {len(graph.relations)} relations:")
    for r in graph.relations:
        print(f"  - {r.source.text} --[{r.relation_type.value}]--> {r.target.text}")
        print(f"    Evidence: {r.evidence}")
    
    if HAS_NETWORKX:
        nx_graph = graph.to_networkx()
        print(f"\nNetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
```

---

## 2. `arp_v3/agents/literature/metabolic_linker.py`

```python
"""
Metabolic Linker for DGAT1/YARS2 Pathway Integration.

Links extracted knowledge graphs to known metabolic pathways and
provides pathway-aware reasoning for BrownBioTech applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .graph_extractor import (
    Entity,
    EntityType,
    ExtractedGraph,
    KnowledgeGraphExtractor,
    RelationType,
)


class PathwayDatabase(Enum):
    """Supported pathway databases."""
    KEGG = "kegg"
    REACTOME = "reactome"
    WIKIPATHWAYS = "wikipathways"
    CUSTOM = "custom"


@dataclass
class PathwayNode:
    """A node in a metabolic pathway."""
    id: str
    name: str
    node_type: EntityType
    database: PathwayDatabase
    synonyms: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    
    def matches_entity(self, entity: Entity) -> bool:
        """Check if this pathway node matches an extracted entity."""
        if entity.normalized_id and entity.normalized_id == self.id:
            return True
        if entity.text.lower() == self.name.lower():
            return True
        return any(
            syn.lower() == entity.text.lower() 
            for syn in self.synonyms
        )


@dataclass
class PathwayEdge:
    """An edge in a metabolic pathway."""
    source_id: str
    target_id: str
    relation: RelationType
    database: PathwayDatabase
    evidence_code: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetabolicPathway:
    """A metabolic pathway definition."""
    id: str
    name: str
    description: str = ""
    database: PathwayDatabase = PathwayDatabase.CUSTOM
    nodes: dict[str, PathwayNode] = field(default_factory=dict)
    edges: list[PathwayEdge] = field(default_factory=list)
    
    def get_node_by_name(self, name: str) -> Optional[PathwayNode]:
        """Find a node by name or synonym."""
        name_lower = name.lower()
        for node in self.nodes.values():
            if node.name.lower() == name_lower:
                return node
            if any(s.lower() == name_lower for s in node.synonyms):
                return node
        return None


# Pre-defined pathways for BrownBioTech focus
DGAT1_PATHWAY = MetabolicPathway(
    id="pw_dgat1_lipid_synthesis",
    name="DGAT1-Mediated Triacylglycerol Synthesis",
    description="Pathway for triglyceride synthesis via DGAT1 enzyme",
    nodes={
        "glycerol_3p": PathwayNode(
            id="glycerol_3p", name="Glycerol-3-phosphate",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["sn-glycerol-3-phosphate", "G3P"]
        ),
        "acyl_coa": PathwayNode(
            id="acyl_coa", name="Acyl-CoA",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["acyl coenzyme A", "fatty acyl-CoA"]
        ),
        "lpa": PathwayNode(
            id="lpa", name="Lysophosphatidic acid",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["1-acyl-sn-glycerol-3-phosphate"]
        ),
        "pa": PathwayNode(
            id="pa", name="Phosphatidic acid",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["1,2-diacyl-sn-glycerol-3-phosphate"]
        ),
        "dag": PathwayNode(
            id="dag", name="Diacylglycerol",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["DAG", "1,2-diacylglycerol"]
        ),
        "tag": PathwayNode(
            id="tag", name="Triacylglycerol",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["TAG", "triglyceride", "triacylglycerol"]
        ),
        "dgat1": PathwayNode(
            id="8694", name="DGAT1",
            node_type=EntityType.GENE, database=PathwayDatabase.KEGG,
            synonyms=["DGAT1", "diacylglycerol O-acyltransferase 1"]
        ),
        "gpam": PathwayNode(
            id="10055", name="GPAM",
            node_type=EntityType.GENE, database=PathwayDatabase.KEGG,
            synonyms=["GPAT1", "glycerol-3-phosphate acyltransferase"]
        ),
        "agpat": PathwayNode(
            id="10059", name="AGPAT1",
            node_type=EntityType.GENE, database=PathwayDatabase.KEGG,
            synonyms=["1-acylglycerol-3-phosphate O-acyltransferase"]
        ),
        "lipin": PathwayNode(
            id="55320", name="LPIN1",
            node_type=EntityType.GENE, database=PathwayDatabase.KEGG,
            synonyms=["Lipin-1", "phosphatidate phosphatase"]
        ),
    },
    edges=[
        PathwayEdge("glycerol_3p", "lpa", RelationType.PRODUCES, PathwayDatabase.CUSTOM, evidence_code="gpam_catalyzed"),
        PathwayEdge("lpa", "pa", RelationType.PRODUCES, PathwayDatabase.CUSTOM, evidence_code="agpat_catalyzed"),
        PathwayEdge("pa", "dag", RelationType.PRODUCES, PathwayDatabase.CUSTOM, evidence_code="lipin_catalyzed"),
        PathwayEdge("dag", "tag", RelationType.PRODUCES, PathwayDatabase.CUSTOM, evidence_code="dgat1_catalyzed"),
        PathwayEdge("dgat1", "tag", RelationType.PRODUCES, PathwayDatabase.CUSTOM),
        PathwayEdge("acyl_coa", "lpa", RelationType.CONSUMES, PathwayDatabase.CUSTOM),
        PathwayEdge("acyl_coa", "pa", RelationType.CONSUMES, PathwayDatabase.CUSTOM),
        PathwayEdge("acyl_coa", "tag", RelationType.CONSUMES, PathwayDatabase.CUSTOM),
    ]
)

YARS2_PATHWAY = MetabolicPathway(
    id="pw_yars2_mito_translation",
    name="YARS2 Mitochondrial Translation Pathway",
    description="YARS2 role in mitochondrial tRNA charging and translation",
    nodes={
        "yars2": PathwayNode(
            id="55221", name="YARS2",
            node_type=EntityType.GENE, database=PathwayDatabase.KEGG,
            synonyms=["YARS2", "tyrosyl-tRNA synthetase 2, mitochondrial"]
        ),
        "tyr": PathwayNode(
            id="tyr", name="L-Tyrosine",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["tyrosine", "Tyr"]
        ),
        "tyr_trna": PathwayNode(
            id="tyr_trna", name="Tyrosine tRNA",
            node_type=EntityType.PROTEIN, database=PathwayDatabase.CUSTOM,
            synonyms=["tRNA-Tyr", "mt-tRNA-Tyr"]
        ),
        "tyr_charged_trna": PathwayNode(
            id="tyr_charged_trna", name="Tyr-tRNA(Tyr)",
            node_type=EntityType.PROTEIN, database=PathwayDatabase.CUSTOM,
            synonyms=["charged tyrosine tRNA", "tyrosyl-tRNA"]
        ),
        "mito_ribosome": PathwayNode(
            id="mito_ribosome", name="Mitochondrial Ribosome",
            node_type=EntityType.PROTEIN, database=PathwayDatabase.CUSTOM,
            synonyms=["55S ribosome", "mitoribosome"]
        ),
        "oxphos_complex": PathwayNode(
            id="oxphos", name="Oxidative Phosphorylation Complex",
            node_type=EntityType.PROTEIN, database=PathwayDatabase.CUSTOM,
            synonyms=["OXPHOS", "ETC complex"]
        ),
        "atp": PathwayNode(
            id="atp", name="ATP",
            node_type=EntityType.METABOLITE, database=PathwayDatabase.CUSTOM,
            synonyms=["adenosine triphosphate"]
        ),
        "tfam": PathwayNode(
            id="7019", name="TFAM",
            node_type=EntityType.GENE, database=PathwayDatabase.KEGG,
            synonyms=["TFAM", "transcription factor A, mitochondrial"]
        ),
    },
    edges=[
        PathwayEdge("yars2", "tyr_charged_trna", RelationType.PRODUCES, PathwayDatabase.CUSTOM),
        PathwayEdge("tyr", "tyr_charged_trna", RelationType.CONSUMES, PathwayDatabase.CUSTOM),
        PathwayEdge("tyr_trna", "tyr_charged_trna", RelationType.CONSUMES, PathwayDatabase.CUSTOM),
        PathwayEdge("tyr_charged_trna", "mito_ribosome", RelationType.ACTIVATES, PathwayDatabase.CUSTOM),
        PathwayEdge("mito_ribosome", "oxphos_complex", RelationType.PRODUCES, PathwayDatabase.CUSTOM),
        PathwayEdge("oxphos_complex", "atp", RelationType.PRODUCES, PathwayDatabase.CUSTOM),
        PathwayEdge("tfam", "mito_ribosome", RelationType.REGULATES, PathwayDatabase.CUSTOM),
    ]
)


class MetabolicLinker:
    """
    Links extracted knowledge graphs to known metabolic pathways.
    
    Provides pathway-aware mapping and identifies novel connections
    in the context of DGAT1/YARS2 research.
    """
    
    def __init__(
        self,
        pathways: Optional[list[MetabolicPathway]] = None,
    ):
        self.pathways = pathways or [DGAT1_PATHWAY, YARS2_PATHWAY]
        self._node_index = self._build_node_index()
    
    def _build_node_index(self) -> dict[str, list[tuple[MetabolicPathway, PathwayNode]]]:
        """Build index for fast node lookup."""
        index: dict[str, list[tuple[MetabolicPathway, PathwayNode]]] = {}
        for pathway in self.pathways:
            for node_id, node in pathway.nodes.items():
                # Index by name
                name_key = node.name.lower()
                if name_key not in index:
                    index[name_key] = []
                index[name_key].append((pathway, node))
                # Index by synonyms
                for syn in node.synonyms:
                    syn_key = syn.lower()
                    if syn_key not in index:
                        index[syn_key] = []
                    index[syn_key].append((pathway, node))
                # Index by ID
                if node.id not in index:
                    index[node.id] = []
                index[node.id].append((pathway, node))
        return index
    
    def link_graph_to_pathways(
        self,
        graph: ExtractedGraph,
    ) -> dict[str, Any]:
        """
        Map extracted graph entities to known pathway nodes.
        
        Args:
            graph: Extracted knowledge graph from literature
            
        Returns:
            Mapping results with matched and unmatched entities
        """
        results = {
            "matched_entities": [],
            "unmatched_entities": [],
            "pathway_coverage": {},
            "novel_connections": [],
        }
        
        # Track pathway coverage
        for pathway in self.pathways:
            results["pathway_coverage"][pathway.id] = {
                "name": pathway.name,
                "matched_nodes": [],
                "total_nodes": len(pathway.nodes),
                "coverage_pct": 0.0,
            }
        
        for entity in graph.entities:
            matches = self._find_pathway_matches(entity)
            
            if matches:
                for pathway, node in matches:
                    results["matched_entities"].append({
                        "entity": entity,
                        "pathway_node": node,
                        "pathway_id": pathway.id,
                        "pathway_name": pathway.name,
                    })
                    results["pathway_coverage"][pathway.id]["matched_nodes"].append(node.id)
            else:
                results["unmatched_entities"].append(entity)
        
        # Calculate coverage percentages
        for pw_id, coverage in results["pathway_coverage"].items():
            unique_matched = len(set(coverage["matched_nodes"]))
            coverage["coverage_pct"] = (
                unique_matched / coverage["total_nodes"] * 100 
                if coverage["total_nodes"] > 0 else 0
            )
        
        # Identify novel connections (relations not in known pathways)
        results["novel_connections"] = self._find_novel_connections(graph)
        
        return results
    
    def _find_pathway_matches(
        self,
        entity: Entity,
    ) -> list[tuple[MetabolicPathway, PathwayNode]]:
        """Find matching pathway nodes for an entity."""
        matches = []
        
        # Try normalized ID first (most reliable)
        if entity.normalized_id:
            id_matches = self._node_index.get(entity.normalized_id, [])
            matches.extend(id_matches)
        
        # Try text matching
        text_matches = self._node_index.get(entity.text.lower(), [])
        for pathway, node in text_matches:
            if (pathway, node) not in matches:
                matches.append((pathway, node))
        
        return matches
    
    def _find_novel_connections(
        self,
        graph: ExtractedGraph,
    ) -> list[dict[str, Any]]:
        """Identify relations not present in known pathways."""
        novel = []
        
        # Build set of known pathway relations
        known_relations = set()
        for pathway in self.pathways:
            for edge in pathway.edges:
                known_relations.add((edge.source_id, edge.target_id, edge.relation))
        
        # Check extracted relations
        for rel in graph.relations:
            # Try to map to pathway IDs
            source_matches = self._find_pathway_matches(rel.source)
            target_matches = self._find_pathway_matches(rel.target)
            
            for src_pw, src_node in source_matches:
                for tgt_pw, tgt_node in target_matches:
                    relation_key = (src_node.id, tgt_node.id, rel.relation_type)
                    if relation_key not in known_relations:
                        novel.append({
                            "source_entity": rel.source.text,
                            "target_entity": rel.target.text,
                            "relation_type": rel.relation_type.value,
                            "source_pathway_node": src_node.name,
                            "target_pathway_node": tgt_node.name,
                            "evidence": rel.evidence,
                            "confidence": rel.confidence,
                        })
        
        return novel
    
    def get_pathway_context(
        self,
        entity_name: str,
        include_neighbors: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Get pathway context for a specific entity.
        
        Args:
            entity_name: Name of the entity to get context for
            include_neighbors: Whether to include neighboring nodes
            
        Returns:
            Pathway context dictionary or None if not found
        """
        matches = self._node_index.get(entity_name.lower(), [])
        if not matches:
            return None
        
        pathway, node = matches[0]
        context = {
            "entity": entity_name,
            "pathway_id": pathway.id,
            "pathway_name": pathway.name,
            "pathway_description": pathway.description,
            "node": {
                "id": node.id,
                "name": node.name,
                "type": node.node_type.value,
            },
            "upstream": [],
            "downstream": [],
        }
        
        if include_neighbors:
            for edge in pathway.edges:
                if edge.target_id == node.id:
                    source_node = pathway.nodes.get(edge.source_id)
                    if source_node:
                        context["upstream"].append({
                            "node": source_node.name,
                            "relation": edge.relation.value,
                        })
                if edge.source_id == node.id:
                    target_node = pathway.nodes.get(edge.target_id)
                    if target_node:
                        context["downstream"].append({
                            "node": target_node.name,
                            "relation": edge.relation.value,
                        })
        
        return context
    
    def generate_pathway_summary(
        self,
        linking_results: dict[str, Any],
    ) -> str:
        """
        Generate human-readable summary of pathway linking results.
        
        Args:
            linking_results: Output from link_graph_to_pathways
            
        Returns:
            Formatted summary string
        """
        lines = ["=== Pathway Linking Summary ===\n"]
        
        # Coverage summary
        lines.append("Pathway Coverage:")
        for pw_id, coverage in linking_results["pathway_coverage"].items():
            lines.append(
                f"  - {coverage['name']}: {coverage['coverage_pct']:.1f}% "
                f"({len(set(coverage['matched_nodes']))}/{coverage['total_nodes']} nodes)"
            )
        
        # Matched entities
        lines.append(f"\nMatched Entities ({len(linking_results['matched_entities'])}):")
        for match in linking_results["matched_entities"][:10]:  # Limit output
            lines.append(
                f"  - {match['entity'].text} -> {match['pathway_name']}"
            )
        if len(linking_results["matched_entities"]) > 10:
            lines.append(f"  ... and {len(linking_results['matched_entities']) - 10} more")
        
        # Novel connections
        novel = linking_results["novel_connections"]
        if novel:
            lines.append(f"\nNovel Connections ({len(novel)}):")
            for conn in novel[:5]:
                lines.append(
                    f"  - {conn['source_entity']} --[{conn['relation_type']}]--> "
                    f"{conn['target_entity']} (conf: {conn['confidence']:.2f})"
                )
                lines.append(f"    Evidence: {conn['evidence']}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    from .graph_extractor import extract_graph_from_text
    
    sample_text = """
    DGAT1 inhibition reduces triacylglycerol accumulation in hepatocytes.
    YARS2 regulates mitochondrial translation and ATP production.
    DGAT1 directly activates PGC1A expression.
    """
    
    graph = extract_graph_from_text(sample_text, "demo_001")
    linker = MetabolicLinker()
    results = linker.link_graph_to_pathways(graph)
    
    print(linker.generate_pathway_summary(results))
    
    # Get specific context
    print("\n=== DGAT1 Context ===")
    context = linker.get_pathway_context("DGAT1")
    if context:
        print(f"Pathway: {context['pathway_name']}")
        print(f"Upstream: {[u['node'] for u in context['upstream']]}")
        print(f"Downstream: {[d['node'] for d in context['downstream']]}")
```

---

## 3. `arp_v3/agents/literature/embeddings/text_embedder.py`

```python
"""
Text Embedder for PubMed/PMC Literature.

Provides embedding generation for biomedical text with support
for multiple embedding backends and caching.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class EmbeddingBackend(Enum):
    """Supported embedding backends."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    MOCK = "mock"  # For testing without dependencies


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: list[float]
    model_name: str
    dimension: int
    metadata: dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def similarity(self, other: "EmbeddingResult") -> float:
        """Calculate cosine similarity with another embedding."""
        if not HAS_NUMPY:
            return 0.0
        
        a = np.array(self.embedding)
        b = np.array(other.embedding)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingResult":
        """Deserialize from dictionary."""
        return cls(**data)


class EmbeddingCache:
    """File-based cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Union[str, Path] = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[EmbeddingResult]:
        """Retrieve cached embedding if available."""
        key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                return EmbeddingResult.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                return None
        return None
    
    def set(self, result: EmbeddingResult) -> None:
        """Cache an embedding result."""
        key = self._get_cache_key(result.text, result.model_name)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(result.to_dict(), f)
        except (IOError, TypeError):
            pass  # Fail silently for cache writes
    
    def clear(self) -> int:
        """Clear all cached embeddings. Returns count of files removed."""
        count = 0
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
            count += 1
        return count


class MockEmbedder:
    """Mock embedder for testing without ML dependencies."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "mock-embedder"
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-embeddings."""
        results = []
        for text in texts:
            # Create deterministic but varied embedding
            hash_val = hashlib.md5(text.encode()).hexdigest()
            embedding = [
                (int(hash_val[i:i+2], 16) / 255.0 - 0.5) * 2
                for i in range(0, min(len(hash_val), self.dimension * 2), 2)
            ]
            # Pad to full dimension
            while len(embedding) < self.dimension:
                embedding.append(0.0)
            results.append(embedding[:self.dimension])
        return results


class TextEmbedder:
    """
    Biomedical text embedder with multiple backend support.
    
    Optimized for PubMed/PMC abstracts and full-text documents
    with domain-aware preprocessing.
    """
    
    # Biomedical-optimized models (priority order)
    BIOMEDICAL_MODELS = [
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    ]
    
    def __init__(
        self,
        backend: EmbeddingBackend = EmbeddingBackend.MOCK,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = ".embedding_cache",
        max_length: int = 512,
        batch_size: int = 32,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize text embedder.
        
        Args:
            backend: Embedding backend to use
            model_name: Specific model name (uses default if None)
            cache_dir: Directory for embedding cache
            max_length: Maximum text length for embedding
            batch_size: Batch size for encoding
            openai_api_key: API key for OpenAI backend
        """
        self.backend = backend
        self.model_name = model_name or self._get_default_model()
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_dir) if cache_dir else None
        
        self._model = None
        self._openai_client = None
        self._initialize_backend(openai_api_key)
    
    def _get_default_model(self) -> str:
        """Get default model for current backend."""
        if self.backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            return self.BIOMEDICAL_MODELS[0]
        elif self.backend == EmbeddingBackend.OPENAI:
            return "text-embedding-ada-002"
        return "mock-embedder"
    
    def _initialize_backend(self, api_key: Optional[str]) -> None:
        """Initialize the selected embedding backend."""
        if self.backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self.model_name = self._model.__class__.__name__
            except ImportError:
                self.backend = EmbeddingBackend.MOCK
                self._model = MockEmbedder()
        elif self.backend == EmbeddingBackend.OPENAI:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                self.backend = EmbeddingBackend.MOCK
                self._model = MockEmbedder()
        else:
            self._model = MockEmbedder()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess biomedical text for embedding.
        
        Args:
            text: Raw biomedical text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common PubMed artifacts
        artifacts = [
            "PMID:", "PMCID:", "DOI:", "[Epub ahead of print]",
            "Copyright ©", "All rights reserved",
        ]
        for artifact in artifacts:
            text = text.replace(artifact, "")
        
        # Truncate to max length (rough word estimate)
        words = text.split()
        if len(words) > self.max_length:
            text = " ".join(words[:self.max_length])
        
        return text.strip()
    
    def embed(
        self,
        text: str,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use caching
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(text, self.model_name)
            if cached:
                cached.metadata["from_cache"] = True
                return cached
        
        # Preprocess and embed
        processed_text = self.preprocess_text(text)
        embedding = self._encode_single(processed_text)
        
        result = EmbeddingResult(
            text=text,
            embedding=embedding,
            model_name=self.model_name,
            dimension=len(embedding),
            metadata={"preprocessed": processed_text != text}
        )
        
        # Cache result
        if use_cache and self.cache:
            self.cache.set(result)
        
        return result
    
    def embed_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use caching
            
        Returns:
            List of EmbeddingResults
        """
        results = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache and self.cache:
                cached = self.cache.get(text, self.model_name)
                if cached:
                    cached.metadata["from_cache"] = True
                    results.append(cached)
                    continue
            
            texts_to_encode.append(text)
            indices_to_encode.append(i)
            results.append(None)  # Placeholder
        
        # Encode uncached texts
        if texts_to_encode:
            processed_texts = [self.preprocess_text(t) for t in texts_to_encode]
            embeddings = self._encode_batch(processed_texts)
            
            for idx, (text, processed, embedding) in enumerate(
                zip(texts_to_encode, processed_texts, embeddings)
            ):
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model_name=self.model_name,
                    dimension=len(embedding),
                    metadata={"preprocessed": processed != text}
                )
                
                if use_cache and self.cache:
                    self.cache.set(result)
                
                results[indices_to_encode[idx]] = result
        
        return results
    
    def _encode_single(self, text: str) -> list[float]:
        """Encode a single text to embedding vector."""
        if self.backend == EmbeddingBackend.OPENAI and self._openai_client:
            response = self._openai_client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        
        return self._encode_batch([text])[0]
    
    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts to embedding vectors."""
        if self.backend == EmbeddingBackend.OPENAI and self._openai_client:
            response = self._openai_client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [item.embedding for item in response.data]
        
        if isinstance(self._model, MockEmbedder):
            return self._model.encode(texts)
        
        # SentenceTransformers
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return [emb.tolist() for emb in embeddings]
        
        # Fallback to mock
        mock = MockEmbedder()
        return mock.encode(texts)
    
    def find_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find most similar texts to a query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            
        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        if not HAS_NUMPY:
            return [(t, 0.0) for t in candidates[:top_k]]
        
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)
        
        similarities = [
            (cand, query_emb.similarity(emb))
            for cand, emb in zip(candidates, candidate_embs)
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


if __name__ == "__main__":
    # Demo
    embedder = TextEmbedder(backend=EmbeddingBackend.MOCK)
    
    # Single embedding
    result = embedder.embed("DGAT1 regulates triglyceride synthesis in liver")
    print(f"Embedding dimension: {result.dimension}")
    print(f"First 5 values: {result.embedding[:5]}")
    
    # Batch embedding
    texts = [
        "DGAT1 inhibition reduces hepatic steatosis",
        "YARS2 mutations cause mitochondrial disease",
        "Triacylglycerol accumulation in hepatocytes",
    ]
    results = embedder.embed_batch(texts)
    
    # Similarity search
    query = "DGAT1 and lipid metabolism"
    similar = embedder.find_similar(query, texts, top_k=2)
    print(f"\nSimilar to '{query}':")
    for text, score in similar:
        print(f"  - {score:.4f}: {text}")
```

---

## 4. `arp_v3/agents/literature/embeddings/omics_fusion.py`

```python
"""
Omics Fusion Module for Multi-Modal Integration.

Fuses text embeddings with omics data (gene expression, metabolomics)
for integrated literature-omics analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .text_embedder import EmbeddingResult, TextEmbedder


class OmicsDataType(Enum):
    """Types of omics data supported for fusion."""
    GENE_EXPRESSION = "gene_expression"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    METHYLATION = "methylation"
    MUTATION = "mutation"


@dataclass
class OmicsProfile:
    """A single omics data profile."""
    data_type: OmicsDataType
    feature_ids: list[str]  # Gene IDs, metabolite IDs, etc.
    values: list[float]  # Expression levels, concentrations, etc.
    sample_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.feature_ids) != len(self.values):
            raise ValueError(
                f"feature_ids length ({len(self.feature_ids)}) must match "
                f"values length ({len(self.values)})"
            )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "data_type": self.data_type.value,
            "feature_ids": self.feature_ids,
            "values": self.values,
            "sample_id": self.sample_id,
            "metadata": self.metadata,
        }
    
    def get_value(self, feature_id: str) -> Optional[float]:
        """Get value for a specific feature."""
        try:
            idx = self.feature_ids.index(feature_id)
            return self.values[idx]
        except ValueError:
            return None
    
    def normalize(self, method: str = "zscore") -> "OmicsProfile":
        """Normalize values in-place."""
        if not HAS_NUMPY:
            return self
        
        values = np.array(self.values)
        
        if method == "zscore":
            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                values = (values - mean) / std
        elif method == "minmax":
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
        
        return OmicsProfile(
            data_type=self.data_type,
            feature_ids=self.feature_ids,
            values=values.tolist(),
            sample_id=self.sample_id,
            metadata={**self.metadata, "normalization": method},
        )


@dataclass
class FusedRepresentation:
    """Fused representation of text and omics data."""
    text_embedding: list[float]
    omics_vector: list[float]
    fused_vector: list[float]
    fusion_method: str
    text_source: str
    omics_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def dimension(self) -> int:
        return len(self.fused_vector)


class OmicsVectorizer:
    """
    Converts omics profiles to fixed-dimension vectors.
    
    Uses feature selection and projection to create comparable
    vector representations across different omics data types.
    """
    
    # Focus genes for BrownBioTech
    FOCUS_GENES = ["DGAT1", "YARS2", "DGAT2", "PPARGC1A", "TFAM"]
    
    # Key metabolites for lipid/mitochondrial pathways
    FOCUS_METABOLITES = [
        "triacylglycerol", "diacylglycerol", "acyl-CoA",
        "phosphatidic acid", "ATP", "tyrosine",
    ]
    
    def __init__(
        self,
        target_dimension: int = 128,
        focus_features: Optional[dict[OmicsDataType, list[str]]] = None,
    ):
        self.target_dimension = target_dimension
        self.focus_features = focus_features or {
            OmicsDataType.GENE_EXPRESSION: self.FOCUS_GENES,
            OmicsDataType.PROTEOMICS: self.FOCUS_GENES,
            OmicsDataType.METABOLOMICS: self.FOCUS_METABOLITES,
        }
        self._projection_matrices: dict[OmicsDataType, Any] = {}
    
    def vectorize(
        self,
        profile: OmicsProfile,
        normalize: bool = True,
    ) -> list[float]:
        """
        Convert omics profile to fixed-dimension vector.
        
        Args:
            profile: OmicsProfile to vectorize
            normalize: Whether to normalize the profile first
            
        Returns:
            Fixed-dimension vector representation
        """
        if normalize:
            profile = profile.normalize()
        
        if not HAS_NUMPY:
            return self._simple_vectorize(profile)
        
        values = np.array(profile.values)
        
        # Get focus features subset if available
        focus = self.focus_features.get(profile.data_type, [])
        if focus:
            focus_indices = []
            for feat in focus:
                if feat in profile.feature_ids:
                    focus_indices.append(profile.feature_ids.index(feat))
            
            if focus_indices:
                focus_values = values[focus_indices]
            else:
                focus_values = values[:len(focus)] if len(values) >= len(focus) else values
        else:
            focus_values = values
        
        # Pad or truncate to target dimension
        result = self._adjust_dimension(focus_values)
        
        return result.tolist()
    
    def _simple_vectorize(self, profile: OmicsProfile) -> list[float]:
        """Simple vectorization without numpy."""
        focus = self.focus_features.get(profile.data_type, [])
        values = profile.values
        
        if focus:
            focus_values = []
            for feat in focus:
                val = profile.get_value(feat)
                focus_values.append(val if val is not None else 0.0)
        else:
            focus_values = values
        
        # Pad or truncate
        while len(focus_values) < self.target_dimension:
            focus_values.append(0.0)
        return focus_values[:self.target_dimension]
    
    def _adjust_dimension(self, values: "np.ndarray") -> "np.ndarray":
        """Adjust array to target dimension."""
        current_len = len(values)
        
        if current_len == self.target_dimension:
            return values
        elif current_len < self.target_dimension:
            # Pad with zeros
            padding = np.zeros(self.target_dimension - current_len)
            return np.concatenate([values, padding])
        else:
            # Truncate or use simple projection
            if current_len <= self.target_dimension * 2:
                return values[:self.target_dimension]
            else:
                # Simple mean pooling to reduce dimension
                chunk_size = current_len // self.target_dimension
                result = []
                for i in range(self.target_dimension):
                    start = i * chunk_size
                    end = start + chunk_size
                    result.append(np.mean(values[start:end]))
                return np.array(result)


class OmicsFusion:
    """
    Fuses text embeddings with omics data vectors.
    
    Supports multiple fusion strategies for integrated analysis
    of literature and experimental data.
    """
    
    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        vectorizer: Optional[OmicsVectorizer] = None,
        fusion_method: str = "concat",
    ):
        """
        Initialize fusion module.
        
        Args:
            embedder: TextEmbedder instance for text encoding
            vectorizer: OmicsVectorizer for omics data
            fusion_method: Fusion strategy ("concat", "weighted", "attention")
        """
        self.embedder = embedder or TextEmbedder()
        self.vectorizer = vectorizer or OmicsVectorizer()
        self.fusion_method = fusion_method
    
    def fuse(
        self,
        text: str,
        omics_profile: OmicsProfile,
        text_weight: float = 0.5,
    ) -> FusedRepresentation:
        """
        Fuse text and omics data into unified representation.
        
        Args:
            text: Biomedical text (abstract, paragraph)
            omics_profile: Omics data profile
            text_weight: Weight for text component (omics gets 1-text_weight)
            
        Returns:
            FusedRepresentation with combined vector
        """
        # Get text embedding
        text_emb = self.embedder.embed(text)
        
        # Get omics vector
        omics_vec = self.vectorizer.vectorize(omics_profile)
        
        # Fuse based on method
        fused = self._fuse_vectors(
            text_emb.embedding,
            omics_vec,
            text_weight,
        )
        
        return FusedRepresentation(
            text_embedding=text_emb.embedding,
            omics_vector=omics_vec,
            fused_vector=fused,
            fusion_method=self.fusion_method,
            text_source=text[:100] + "..." if len(text) > 100 else text,
            omics_source=omics_profile.sample_id,
            metadata={
                "text_weight": text_weight,
                "omics_weight": 1 - text_weight,
                "text_dim": len(text_emb.embedding),
                "omics_dim": len(omics_vec),
            }
        )
    
    def fuse_batch(
        self,
        texts: list[str],
        omics_profiles: list[OmicsProfile],
        text_weight: float = 0.5,
    ) -> list[FusedRepresentation]:
        """
        Fuse multiple text-omics pairs.
        
        Args:
            texts: List of biomedical texts
            omics_profiles: List of omics profiles (same length as texts)
            text_weight: Weight for text component
            
        Returns:
            List of FusedRepresentations
        """
        if len(texts) != len(omics_profiles):
            raise ValueError(
                f"texts length ({len(texts)}) must match "
                f"omics_profiles length ({len(omics_profiles)})"
            )
        
        return [
            self.fuse(text, profile, text_weight)
            for text, profile in zip(texts, omics_profiles)
        ]
    
    def _fuse_vectors(
        self,
        text_vec: list[float],
        omics_vec: list[float],
        text_weight: float,
    ) -> list[float]:
        """Apply fusion strategy to combine vectors."""
        if not HAS_NUMPY:
            return self._simple_fuse(text_vec, omics_vec, text_weight)
        
        text_arr = np.array(text_vec)
        omics_arr = np.array(omics_vec)
        
        if self.fusion_method == "concat":
            return np.concatenate([text_arr, omics_arr]).tolist()
        
        elif self.fusion_method == "weighted":
            # Pad to same dimension if needed
            max_dim = max(len(text_arr), len(omics_arr))
            text_padded = np.pad(text_arr, (0, max_dim - len(text_arr)))
            omics_padded = np.pad(omics_arr, (0, max_dim - len(omics_arr)))
            return (text_weight * text_padded + (1 - text_weight) * omics_padded).tolist()
        
        elif self.fusion_method == "attention":
            # Simple attention-like weighting based on magnitude
            text_norm = np.linalg.norm(text_arr)
            omics_norm = np.linalg.norm(omics_arr)
            
            if text_norm + omics_norm == 0:
                return np.zeros(max(len(text_arr), len(omics_arr))).tolist()
            
            text_attn = text_norm / (text_norm + omics_norm)
            omics_attn = omics_norm / (text_norm + omics_norm)
            
            # Apply attention and concatenate
            weighted_text = text_arr * text_attn
            weighted_omics = omics_arr * omics_attn
            return np.concatenate([weighted_text, weighted_omics]).tolist()
        
        else:
            return self._simple_fuse(text_vec, omics_vec, text_weight)
    
    def _simple_fuse(
        self,
        text_vec: list[float],
        omics_vec: list[float],
        text_weight: float,
    ) -> list[float]:
        """Simple fusion without numpy."""
        if self.fusion_method == "concat":
            return text_vec + omics_vec
        
        # Weighted average (pad shorter vector)
        max_len = max(len(text_vec), len(omics_vec))
        text_padded = text_vec + [0.0] * (max_len - len(text_vec))
        omics_padded = omics_vec + [0.0] * (max_len - len(omics_vec))
        
        return [
            text_weight * t + (1 - text_weight) * o
            for t, o in zip(text_padded, omics_padded)
        ]
    
    def compute_similarity(
        self,
        fused_a: FusedRepresentation,
        fused_b: FusedRepresentation,
    ) -> dict[str, float]:
        """
        Compute similarity between two fused representations.
        
        Returns similarity scores for each component and the fused vector.
        """
        if not HAS_NUMPY:
            return {"fused": 0.0, "text": 0.0, "omics": 0.0}
        
        def cosine_sim(a: list[float], b: list[float]) -> float:
            a_arr, b_arr = np.array(a), np.array(b)
            norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
        
        return {
            "fused": cosine_sim(fused_a.fused_vector, fused_b.fused_vector),
            "text": cosine_sim(fused_a.text_embedding, fused_b.text_embedding),
            "omics": cosine_sim(fused_a.omics_vector, fused_b.omics_vector),
        }


if __name__ == "__main__":
    # Demo
    embedder = TextEmbedder()
    fusion = OmicsFusion(embedder=embedder, fusion_method="concat")
    
    # Create sample omics profile
    profile = OmicsProfile(
        data_type=OmicsDataType.GENE_EXPRESSION,
        feature_ids=["DGAT1", "YARS2", "DGAT2", "PPARGC1A", "TFAM"],
        values=[2.5, 0.8, 1.2, 3.1, 1.5],
        sample_id="liver_sample_001"
    )
    
    # Fuse text with omics
    text = "DGAT1 is highly expressed in liver and regulates triglyceride synthesis"
    fused = fusion.fuse(text, profile, text_weight=0.6)
    
    print(f"Fusion method: {fused.fusion_method}")
    print(f"Fused dimension: {fused.dimension}")
    print(f"Text dim: {fused.metadata['text_dim']}, Omics dim: {fused.metadata['omics_dim']}")
    
    # Compare with another sample
    profile2 = OmicsProfile(
        data_type=OmicsDataType.GENE_EXPRESSION,
        feature_ids=["DGAT1", "YARS2", "DGAT2", "PPARGC1A", "TFAM"],
        values=[0.5, 4.2, 0.3, 0.8, 3.9],
        sample_id="muscle_sample_002"
    )
    text2 = "YARS2 mutations impair mitochondrial translation in skeletal muscle"
    fused2 = fusion.fuse(text2, profile2, text_weight=0.6)
    
    similarities = fusion.compute_similarity(fused, fused2)
    print(f"\nSimilarity scores:")
    for component, score in similarities.items():
        print(f"  {component}: {score:.4f}")
```

---

## 5. `arp_v3/agents/literature/embeddings/__init__.py`

```python
"""
Embeddings package for BrownBioTech literature analysis.

Provides text embedding and omics fusion capabilities.
"""

from .text_embedder import (
    EmbeddingBackend,
    EmbeddingCache,
    EmbeddingResult,
    TextEmbedder,
)
from .omics_fusion import (
    FusedRepresentation,
    OmicsDataType,
    OmicsFusion,
    OmicsProfile,
    OmicsVectorizer,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingCache",
    "EmbeddingResult",
    "TextEmbedder",
    "FusedRepresentation",
    "OmicsDataType",
    "OmicsFusion",
    "OmicsProfile",
    "OmicsVectorizer",
]
```

---

## 6. `arp_v3/agents/literature/__init__.py`

```python
"""
Literature agent package for BrownBioTech.

Provides knowledge graph extraction, metabolic pathway linking,
and multi-modal embedding fusion for biomedical literature analysis.
"""

from .graph_extractor import (
    Entity,
    EntityType,
    ExtractedGraph,
    KnowledgeGraphExtractor,
    Relation,
    RelationType,
    extract_graph_from_text,
)
from .metabolic_linker import (
    MetabolicLinker,
    MetabolicPathway,
    PathwayDatabase,
    PathwayEdge,
    PathwayNode,
    DGAT1_PATHWAY,
    YARS2_PATHWAY,
)
from .embeddings import (
    EmbeddingBackend,
    EmbeddingCache,
    EmbeddingResult,
    FusedRepresentation,
    OmicsDataType,
    OmicsFusion,
    OmicsProfile,
    OmicsVectorizer,
    TextEmbedder,
)

__all__ = [
    # Graph extraction
    "Entity",
    "EntityType",
    "ExtractedGraph",
    "KnowledgeGraphExtractor",
    "Relation",
    "RelationType",
    "extract_graph_from_text",
    # Metabolic linking
    "MetabolicLinker",
    "MetabolicPathway",
    "PathwayDatabase",
    "PathwayEdge",
    "PathwayNode",
    "DGAT1_PATHWAY",
    "YARS2_PATHWAY",
    # Embeddings
    "EmbeddingBackend",
    "EmbeddingCache",
    "EmbeddingResult",
    "FusedRepresentation",
    "OmicsDataType",
    "OmicsFusion",
    "OmicsProfile",
    "OmicsVectorizer",
    "TextEmbedder",
]
```

---

## Summary of Improvements

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `graph_extractor.py` | Extract knowledge graphs from text | Pattern-based entity/relation extraction, confidence scoring, NetworkX export, focus gene support |
| `metabolic_linker.py` | Link extracted graphs to known pathways | Pre-defined DGAT1/YARS2 pathways, coverage analysis, novel connection detection |
| `text_embedder.py` | Embed biomedical text | Multi-backend support (ST, OpenAI, Mock), caching, similarity search |
| `omics_fusion.py` | Fuse text + omics data | Multiple fusion strategies (concat, weighted, attention), component-wise similarity |

### Integration Points
- All modules use type hints and docstrings
- Error handling with graceful fallbacks (e.g., MockEmbedder when dependencies missing)
- Optional numpy dependency with pure-Python fallbacks
- File-based embedding cache for performance
- Pre-curated DGAT1/YARS2 pathway definitions for BrownBioTech focus