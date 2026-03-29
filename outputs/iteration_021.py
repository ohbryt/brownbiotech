# BrownBioTech Iteration 21/100 — Credibility-First & Personalization Enhancement

## File 1: `brownbiotech/visitor_detection.py`

```python
"""
Visitor Role Detection Engine for BrownBioTech Platform.

Classifies visitors into roles based on behavioral signals, source attribution,
and session patterns to enable content personalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import urlparse


class VisitorRole(Enum):
    """Classification of visitor roles for content personalization."""
    SCIENTIST = "scientist"
    INVESTOR = "investor"
    PATIENT = "patient"
    CLINICIAN = "clinician"
    STUDENT = "student"
    GENERAL = "general"
    UNKNOWN = "unknown"


class TrafficSource(Enum):
    """Origin channels for visitor attribution."""
    ORGANIC_SEARCH = "organic_search"
    DIRECT = "direct"
    SOCIAL_MEDIA = "social_media"
    REFERRAL_ACADEMIC = "referral_academic"
    REFERRAL_NEWS = "referral_news"
    PAID_SEARCH = "paid_search"
    EMAIL_CAMPAIGN = "email_campaign"
    UNKNOWN = "unknown"


@dataclass
class BehavioralSignals:
    """Captures visitor behavioral patterns for role inference."""
    pages_visited: list[str] = field(default_factory=list)
    time_on_site_seconds: float = 0.0
    scroll_depth_avg: float = 0.0
    clicked_elements: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    return_visitor: bool = False
    previous_visits: int = 0
    device_type: str = "desktop"


@dataclass
class VisitorProfile:
    """Complete visitor profile after detection analysis."""
    role: VisitorRole
    confidence: float
    traffic_source: TrafficSource
    signals: BehavioralSignals
    detected_interests: list[str] = field(default_factory=list)
    credibility_signals: list[str] = field(default_factory=list)


# URL patterns indicating scientific interest
SCIENTIST_PATTERNS = [
    r"/research",
    r"/publications",
    r"/protocols",
    r"/api-documentation",
    r"/pipelines",
    r"/bioinformatics",
    r"/datasets",
]

# URL patterns indicating investor interest
INVESTOR_PATTERNS = [
    r"/investors",
    r"/financial",
    r"/ir-",
    r"/quarterly",
    r"/board",
    r"/market-",
    r"/valuation",
]

# URL patterns indicating patient interest
PATIENT_PATTERNS = [
    r"/patients",
    r"/trials",
    r"/treatment",
    r"/therapies",
    r"/clinical-trials",
    r"/safety",
]

# URL patterns indicating clinician interest
CLINICIAN_PATTERNS = [
    r"/clinicians",
    r"/hcp",
    r"/prescribing",
    r"/guidelines",
    r"/cds",
]

# Search query indicators
SCIENTIST_QUERY_TERMS = [
    "crispr", "sequencing", "assay", "protocol", "bioinformatics",
    "genomics", "proteomics", "cell line", "vector", "plasmid",
]

INVESTOR_QUERY_TERMS = [
    "revenue", "valuation", "ipo", "funding", "market cap",
    "earnings", "growth", "pipeline value",
]

PATIENT_QUERY_TERMS = [
    "treatment", "therapy", "side effects", "trial enrollment",
    "diagnosis", "prognosis", "drug",
]


class VisitorDetectionEngine:
    """
    Engine for classifying visitor roles based on multi-signal analysis.
    
    Uses a weighted scoring system combining URL patterns, search queries,
    traffic source, and behavioral signals to determine the most likely
    visitor role with confidence scoring.
    """

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize the detection engine.
        
        Args:
            min_confidence: Minimum confidence threshold for role assignment.
                           Below this, role defaults to UNKNOWN.
        """
        self.min_confidence = min_confidence
        self._pattern_weights = {
            VisitorRole.SCIENTIST: 0.35,
            VisitorRole.INVESTOR: 0.35,
            VisitorRole.PATIENT: 0.35,
            VisitorRole.CLINICIAN: 0.35,
            VisitorRole.STUDENT: 0.20,
        }
        self._query_weights = {
            VisitorRole.SCIENTIST: 0.30,
            VisitorRole.INVESTOR: 0.30,
            VisitorRole.PATIENT: 0.30,
        }
        self._source_weights = {
            TrafficSource.REFERRAL_ACADEMIC: {VisitorRole.SCIENTIST: 0.25, VisitorRole.STUDENT: 0.15},
            TrafficSource.REFERRAL_NEWS: {VisitorRole.INVESTOR: 0.20, VisitorRole.GENERAL: 0.10},
            TrafficSource.SOCIAL_MEDIA: {VisitorRole.PATIENT: 0.15, VisitorRole.GENERAL: 0.10},
            TrafficSource.EMAIL_CAMPAIGN: {VisitorRole.INVESTOR: 0.15, VisitorRole.SCIENTIST: 0.10},
        }

    def classify_visitor(
        self,
        signals: BehavioralSignals,
        traffic_source: TrafficSource = TrafficSource.UNKNOWN,
        referrer: str = "",
        utm_params: Optional[dict[str, str]] = None,
    ) -> VisitorProfile:
        """
        Classify a visitor based on behavioral signals and context.
        
        Args:
            signals: Behavioral signals captured from visitor session.
            traffic_source: Detected traffic source channel.
            referrer: Referrer URL string for additional context.
            utm_params: UTM parameters from campaign tracking.
            
        Returns:
            VisitorProfile with role classification, confidence, and metadata.
            
        Raises:
            ValueError: If signals is None or invalid.
        """
        if signals is None:
            raise ValueError("Behavioral signals cannot be None")

        scores: dict[VisitorRole, float] = {role: 0.0 for role in VisitorRole}
        
        # Score based on URL patterns
        self._score_url_patterns(signals.pages_visited, scores)
        
        # Score based on search queries
        self._score_search_queries(signals.search_queries, scores)
        
        # Score based on traffic source
        effective_source = self._determine_effective_source(traffic_source, referrer, utm_params)
        self._score_traffic_source(effective_source, scores)
        
        # Score based on behavioral patterns
        self._score_behavioral_patterns(signals, scores)
        
        # Determine best role
        best_role, best_score = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_score, 1.0)
        
        if confidence < self.min_confidence:
            best_role = VisitorRole.UNKNOWN
            confidence = 0.0

        # Extract interests and credibility signals
        detected_interests = self._extract_interests(signals)
        credibility_signals = self._assess_credibility_signals(signals, best_role)

        return VisitorProfile(
            role=best_role,
            confidence=confidence,
            traffic_source=effective_source,
            signals=signals,
            detected_interests=detected_interests,
            credibility_signals=credibility_signals,
        )

    def _score_url_patterns(
        self, pages: list[str], scores: dict[VisitorRole, float]
    ) -> None:
        """Score visitor based on URL path patterns visited."""
        pattern_map = {
            VisitorRole.SCIENTIST: SCIENTIST_PATTERNS,
            VisitorRole.INVESTOR: INVESTOR_PATTERNS,
            VisitorRole.PATIENT: PATIENT_PATTERNS,
            VisitorRole.CLINICIAN: CLINICIAN_PATTERNS,
        }
        
        for page in pages:
            try:
                path = urlparse(page).path.lower()
            except Exception:
                path = page.lower()
                
            for role, patterns in pattern_map.items():
                weight = self._pattern_weights.get(role, 0.25)
                for pattern in patterns:
                    if re.search(pattern, path):
                        scores[role] += weight
                        break

    def _score_search_queries(
        self, queries: list[str], scores: dict[VisitorRole, float]
    ) -> None:
        """Score visitor based on search query terms."""
        query_map = {
            VisitorRole.SCIENTIST: SCIENTIST_QUERY_TERMS,
            VisitorRole.INVESTOR: INVESTOR_QUERY_TERMS,
            VisitorRole.PATIENT: PATIENT_QUERY_TERMS,
        }
        
        for query in queries:
            query_lower = query.lower()
            for role, terms in query_map.items():
                weight = self._query_weights.get(role, 0.20)
                if any(term in query_lower for term in terms):
                    scores[role] += weight

    def _score_traffic_source(
        self, source: TrafficSource, scores: dict[VisitorRole, float]
    ) -> None:
        """Adjust scores based on traffic source attribution."""
        source_boosts = self._source_weights.get(source, {})
        for role, boost in source_boosts.items():
            scores[role] += boost

    def _score_behavioral_patterns(
        self, signals: BehavioralSignals, scores: dict[VisitorRole, float]
    ) -> None:
        """Score based on behavioral engagement patterns."""
        # Deep engagement suggests professional interest
        if signals.scroll_depth_avg > 0.7:
            scores[VisitorRole.SCIENTIST] += 0.10
            scores[VisitorRole.CLINICIAN] += 0.10
            
        # Return visitors with history likely professionals
        if signals.return_visitor and signals.previous_visits > 3:
            scores[VisitorRole.SCIENTIST] += 0.15
            scores[VisitorRole.CLINICIAN] += 0.10
            scores[VisitorRole.INVESTOR] += 0.10
            
        # Student indicators
        if any(".edu" in page for page in signals.pages_visited):
            scores[VisitorRole.STUDENT] += 0.20

    def _determine_effective_source(
        self,
        base_source: TrafficSource,
        referrer: str,
        utm_params: Optional[dict[str, str]],
    ) -> TrafficSource:
        """Determine effective traffic source considering UTM overrides."""
        if utm_params:
            source = utm_params.get("source", "").lower()
            medium = utm_params.get("medium", "").lower()
            
            if medium == "cpc" or medium == "ppc":
                return TrafficSource.PAID_SEARCH
            if medium == "email":
                return TrafficSource.EMAIL_CAMPAIGN
            if medium == "social":
                return TrafficSource.SOCIAL_MEDIA
                
        if referrer:
            try:
                parsed = urlparse(referrer)
                domain = parsed.netloc.lower()
                if any(edu in domain for edu in [".edu", ".ac.", "scholar.google"]):
                    return TrafficSource.REFERRAL_ACADEMIC
                if any(news in domain for news in ["reuters", "bloomberg", "biopharmadive"]):
                    return TrafficSource.REFERRAL_NEWS
                if any(social in domain for social in ["twitter", "linkedin", "facebook"]):
                    return TrafficSource.SOCIAL_MEDIA
            except Exception:
                pass
                
        return base_source

    def _extract_interests(self, signals: BehavioralSignals) -> list[str]:
        """Extract topic interests from visitor behavior."""
        interests: list[str] = []
        interest_keywords = {
            "genomics": ["genom", "dna", "sequencing"],
            "proteomics": ["protein", "proteom", "mass spec"],
            "cell_therapy": ["cell therapy", "car-t", "immunotherapy"],
            "diagnostics": ["diagnostic", "biomarker", "assay"],
            "ai_ml": ["ai", "machine learning", "deep learning"],
            "regulatory": ["fda", "ema", "regulation", "compliance"],
        }
        
        all_text = " ".join(signals.pages_visited + signals.search_queries).lower()
        
        for interest, keywords in interest_keywords.items():
            if any(kw in all_text for kw in keywords):
                interests.append(interest)
                
        return interests

    def _assess_credibility_signals(
        self, signals: BehavioralSignals, role: VisitorRole
    ) -> list[str]:
        """Assess signals that indicate visitor credibility/trustworthiness."""
        credibility: list[str] = []
        
        if signals.return_visitor:
            credibility.append("return_visitor")
        if signals.previous_visits > 5:
            credibility.append("frequent_visitor")
        if signals.time_on_site_seconds > 120:
            credibility.append("extended_engagement")
        if signals.scroll_depth_avg > 0.6:
            credibility.append("deep_content_consumption")
        if role == VisitorRole.SCIENTIST and signals.search_queries:
            credibility.append("technical_query_evidence")
        if role == VisitorRole.CLINICIAN and any("hcp" in p for p in signals.pages_visited):
            credibility.append("hcp_portal_access")
            
        return credibility


def create_visitor_profile(
    pages: Optional[list[str]] = None,
    search_queries: Optional[list[str]] = None,
    referrer: str = "",
    time_on_site: float = 0.0,
) -> VisitorProfile:
    """
    Convenience function to create a visitor profile with minimal inputs.
    
    Args:
        pages: List of page URLs visited.
        search_queries: List of search queries entered.
        referrer: Referrer URL.
        time_on_site: Time spent on site in seconds.
        
    Returns:
        VisitorProfile with classification results.
    """
    engine = VisitorDetectionEngine()
    signals = BehavioralSignals(
        pages_visited=pages or [],
        search_queries=search_queries or [],
        time_on_site_seconds=time_on_site,
    )
    return engine.classify_visitor(signals, referrer=referrer)


if __name__ == "__main__":
    # Demonstration of visitor detection
    print("=" * 60)
    print("BrownBioTech Visitor Detection Engine - Demo")
    print("=" * 60)
    
    # Test Case 1: Scientist
    scientist_signals = BehavioralSignals(
        pages_visited=[
            "https://brownbiotech.com/research/crispr-platform",
            "https://brownbiotech.com/publications/2024-nature",
            "https://brownbiotech.com/protocols/cell-culture",
        ],
        search_queries=["crispr protocol optimization", "cell line engineering"],
        time_on_site_seconds=245.0,
        scroll_depth_avg=0.82,
        return_visitor=True,
        previous_visits=7,
    )
    
    engine = VisitorDetectionEngine()
    profile1 = engine.classify_visitor(
        scientist_signals,
        traffic_source=TrafficSource.REFERRAL_ACADEMIC,
        referrer="https://pubmed.ncbi.nlm.nih.gov/",
    )
    
    print(f"\nTest 1 - Expected: Scientist")
    print(f"  Detected Role: {profile1.role.value}")
    print(f"  Confidence: {profile1.confidence:.2%}")
    print(f"  Interests: {profile1.detected_interests}")
    print(f"  Credibility Signals: {profile1.credibility_signals}")
    
    # Test Case 2: Investor
    investor_signals = BehavioralSignals(
        pages_visited=[
            "https://brownbiotech.com/investors/overview",
            "https://brownbiotech.com/financial/quarterly-report",
        ],
        search_queries=["brownbiotech valuation", "biotech ipo timeline"],
        time_on_site_seconds=90.0,
        scroll_depth_avg=0.45,
    )
    
    profile2 = engine.classify_visitor(
        investor_signals,
        traffic_source=TrafficSource.REFERRAL_NEWS,
        referrer="https://www.biopharmadive.com/",
        utm_params={"source": "biopharmadive", "medium": "referral"},
    )
    
    print(f"\nTest 2 - Expected: Investor")
    print(f"  Detected Role: {profile2.role.value}")
    print(f"  Confidence: {profile2.confidence:.2%}")
    print(f"  Interests: {profile2.detected_interests}")
    
    # Test Case 3: Patient
    patient_signals = BehavioralSignals(
        pages_visited=[
            "https://brownbiotech.com/patients/clinical-trials",
            "https://brownbiotech.com/treatments/oncology",
        ],
        search_queries=["cancer treatment trial enrollment"],
        time_on_site_seconds=180.0,
    )
    
    profile3 = engine.classify_visitor(
        patient_signals,
        traffic_source=TrafficSource.ORGANIC_SEARCH,
    )
    
    print(f"\nTest 3 - Expected: Patient")
    print(f"  Detected Role: {profile3.role.value}")
    print(f"  Confidence: {profile3.confidence:.2%}")
    print(f"  Interests: {profile3.detected_interests}")
    
    # Test Case 4: Unknown/Low confidence
    unknown_signals = BehavioralSignals(
        pages_visited=["https://brownbiotech.com/"],
        time_on_site_seconds=15.0,
    )
    
    profile4 = engine.classify_visitor(unknown_signals)
    
    print(f"\nTest 4 - Expected: Unknown (low confidence)")
    print(f"  Detected Role: {profile4.role.value}")
    print(f"  Confidence: {profile4.confidence:.2%}")
    
    print("\n" + "=" * 60)
```

---

## File 2: `brownbiotech/content_personalization.py`

```python
"""
Content Personalization Engine for BrownBioTech Platform.

Adapts content presentation based on detected visitor roles,
emphasizing credibility signals and relevant information hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from brownbiotech.visitor_detection import (
    VisitorProfile,
    VisitorRole,
)


class ContentPriority(Enum):
    """Priority levels for content elements."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    HIDDEN = "hidden"


class CredibilityElementType(Enum):
    """Types of credibility-enhancing content elements."""
    PEER_REVIEWED_CITATION = "peer_reviewed_citation"
    CLINICAL_TRIAL_REFERENCE = "clinical_trial_reference"
    REGULATORY_STATUS = "regulatory_status"
    EXPERT_TESTIMONIAL = "expert_testimonial"
    DATA_VISUALIZATION = "data_visualization"
    PUBLICATION_LINK = "publication_link"
    CERTIFICATION_BADGE = "certification_badge"
    PARTNER_LOGO = "partner_logo"


@dataclass
class ContentBlock:
    """Represents a customizable content block."""
    block_id: str
    title: str
    content: str
    base_priority: ContentPriority = ContentPriority.SECONDARY
    applicable_roles: set[VisitorRole] = field(
        default_factory=lambda: {r for r in VisitorRole}
    )
    credibility_elements: list[CredibilityElementType] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalizedContent:
    """Result of content personalization with ordered blocks."""
    blocks: list[PersonalizedBlock]
    visitor_role: VisitorRole
    credibility_score: float
    personalization_signals: list[str] = field(default_factory=list)


@dataclass
class PersonalizedBlock:
    """A content block with personalization adjustments applied."""
    block: ContentBlock
    adjusted_priority: ContentPriority
    credibility_boost: float
    modifications: dict[str, Any] = field(default_factory=dict)


# Default content blocks for BrownBioTech homepage
DEFAULT_CONTENT_BLOCKS: list[ContentBlock] = [
    ContentBlock(
        block_id="hero_mission",
        title="Advancing Human Health Through Biotechnology",
        content="Pioneering next-generation therapeutics with rigorous science.",
        base_priority=ContentPriority.PRIMARY,
        credibility_elements=[CredibilityElementType.CERTIFICATION_BADGE],
    ),
    ContentBlock(
        block_id="research_highlights",
        title="Research Highlights",
        content="Latest breakthroughs from our R&D labs.",
        base_priority=ContentPriority.SECONDARY,
        applicable_roles={VisitorRole.SCIENTIST, VisitorRole.CLINICIAN, VisitorRole.STUDENT},
        credibility_elements=[
            CredibilityElementType.PEER_REVIEWED_CITATION,
            CredibilityElementType.PUBLICATION_LINK,
            CredibilityElementType.DATA_VISUALIZATION,
        ],
    ),
    ContentBlock(
        block_id="pipeline_overview",
        title="Therapeutic Pipeline",
        content="Our portfolio of investigational therapies across multiple modalities.",
        base_priority=ContentPriority.SECONDARY,
        credibility_elements=[
            CredibilityElementType.CLINICAL_TRIAL_REFERENCE,
            CredibilityElementType.REGULATORY_STATUS,
            CredibilityElementType.DATA_VISUALIZATION,
        ],
    ),
    ContentBlock(
        block_id="investor_relations",
        title="Investor Relations",
        content="Financial performance, governance, and shareholder resources.",
        base_priority=ContentPriority.TERTIARY,
        applicable_roles={VisitorRole.INVESTOR},
        credibility_elements=[CredibilityElementType.CERTIFICATION_BADGE],
    ),
    ContentBlock(
        block_id="patient_resources",
        title="Patient Resources",
        content="Information about our clinical trials and support programs.",
        base_priority=ContentPriority.TERTIARY,
        applicable_roles={VisitorRole.PATIENT, VisitorRole.CLINICIAN},
        credibility_elements=[
            CredibilityElementType.CLINICAL_TRIAL_REFERENCE,
            CredibilityElementType.REGULATORY_STATUS,
        ],
    ),
    ContentBlock(
        block_id="publications",
        title="Publications",
        content="Peer-reviewed publications from our research teams.",
        base_priority=ContentPriority.TERTIARY,
        applicable_roles={VisitorRole.SCIENTIST, VisitorRole.CLINICIAN, VisitorRole.STUDENT},
        credibility_elements=[
            CredibilityElementType.PEER_REVIEWED_CITATION,
            CredibilityElementType.PUBLICATION_LINK,
        ],
    ),
    ContentBlock(
        block_id="partnerships",
        title="Strategic Partnerships",
        content="Collaborations with leading academic and industry partners.",
        base_priority=ContentPriority.TERTIARY,
        applicable_roles={VisitorRole.INVESTOR, VisitorRole.SCIENTIST, VisitorRole.GENERAL},
        credibility_elements=[CredibilityElementType.PARTNER_LOGO],
    ),
    ContentBlock(
        block_id="careers",
        title="Join Our Team",
        content="Career opportunities at BrownBioTech.",
        base_priority=ContentPriority.HIDDEN,
        applicable_roles={VisitorRole.SCIENTIST, VisitorRole.STUDENT, VisitorRole.GENERAL},
    ),
]


class ContentPersonalizationEngine:
    """
    Engine for personalizing content based on visitor profiles.
    
    Reorders content blocks, adjusts priorities, and enhances credibility
    signals based on the detected visitor role and confidence level.
    """

    def __init__(
        self,
        content_blocks: Optional[list[ContentBlock]] = None,
        credibility_weight: float = 0.3,
    ):
        """
        Initialize the personalization engine.
        
        Args:
            content_blocks: Available content blocks. Uses defaults if None.
            credibility_weight: Weight given to credibility signals in ranking.
        """
        self.content_blocks = content_blocks or DEFAULT_CONTENT_BLOCKS
        self.credibility_weight = credibility_weight
        self._role_priority_overrides = self._build_role_overrides()

    def personalize(
        self,
        visitor_profile: VisitorProfile,
        page_context: Optional[dict[str, Any]] = None,
    ) -> PersonalizedContent:
        """
        Generate personalized content layout for a visitor.
        
        Args:
            visitor_profile: Detected visitor profile with role and signals.
            page_context: Additional page-specific context (e.g., page type).
            
        Returns:
            PersonalizedContent with ordered and adjusted content blocks.
        """
        role = visitor_profile.role
        confidence = visitor_profile.confidence
        page_context = page_context or {}

        personalized_blocks: list[PersonalizedBlock] = []
        personalization_signals: list[str] = []

        for block in self.content_blocks:
            # Skip blocks not applicable to this role
            if role not in block.applicable_roles and role != VisitorRole.UNKNOWN:
                continue

            # Calculate adjusted priority
            adjusted_priority = self._calculate_priority(
                block, role, confidence, page_context
            )

            # Calculate credibility boost
            credibility_boost = self._calculate_credibility_boost(
                block, visitor_profile
            )

            # Generate modifications
            modifications = self._generate_modifications(
                block, role, visitor_profile
            )

            if modifications:
                personalization_signals.append(
                    f"modified_{block.block_id}"
                )

            personalized_blocks.append(
                PersonalizedBlock(
                    block=block,
                    adjusted_priority=adjusted_priority,
                    credibility_boost=credibility_boost,
                    modifications=modifications,
                )
            )

        # Sort blocks by priority and credibility
        sorted_blocks = self._sort_blocks(personalized_blocks)

        # Calculate overall credibility score
        credibility_score = self._calculate_overall_credibility(
            sorted_blocks, visitor_profile
        )

        if role != VisitorRole.UNKNOWN:
            personalization_signals.insert(0, f"role_matched_{role.value}")

        return PersonalizedContent(
            blocks=sorted_blocks,
            visitor_role=role,
            credibility_score=credibility_score,
            personalization_signals=personalization_signals,
        )

    def _calculate_priority(
        self,
        block: ContentBlock,
        role: VisitorRole,
        confidence: float,
        page_context: dict[str, Any],
    ) -> ContentPriority:
        """Calculate adjusted content priority based on role."""
        base = block.base_priority
        
        # Get role-specific override
        overrides = self._role_priority_overrides.get(role, {})
        override_priority = overrides.get(block.block_id)
        
        if override_priority is not None:
            # Apply override with confidence scaling
            if confidence > 0.6:
                return override_priority
            elif confidence > 0.4:
                # Moderate confidence: move one step toward override
                return self._interpolate_priority(base, override_priority)
        
        return base

    def _calculate_credibility_boost(
        self,
        block: ContentBlock,
        profile: VisitorProfile,
    ) -> float:
        """Calculate credibility boost for a block based on visitor signals."""
        boost = 0.0
        
        # Boost credibility elements for scientists and clinicians
        if profile.role in (VisitorRole.SCIENTIST, VisitorRole.CLINICIAN):
            element_count = len(block.credibility_elements)
            boost += element_count * 0.1
            
        # Boost for visitors with credibility signals
        visitor_credibility = len(profile.credibility_signals)
        boost += visitor_credibility * 0.05
        
        # Reduce boost for low-confidence classifications
        if profile.confidence < 0.4:
            boost *= 0.5
            
        return min(boost, 1.0)

    def _generate_modifications(
        self,
        block: ContentBlock,
        role: VisitorRole,
        profile: VisitorProfile,
    ) -> dict[str, Any]:
        """Generate content modifications for the visitor role."""
        modifications: dict[str, Any] = {}
        
        # Role-specific title adjustments
        title_overrides = {
            VisitorRole.SCIENTIST: {
                "research_highlights": "Latest Research & Publications",
                "pipeline_overview": "Therapeutic Pipeline — Mechanism & Data",
            },
            VisitorRole.INVESTOR: {
                "pipeline_overview": "Pipeline Valuation & Milestones",
                "hero_mission": "Science-Driven Value Creation",
            },
            VisitorRole.PATIENT: {
                "pipeline_overview": "Therapies in Development",
                "hero_mission": "Hope Through Science",
            },
            VisitorRole.CLINICIAN: {
                "pipeline_overview": "Clinical Pipeline & Evidence",
                "patient_resources": "Clinical Resources & Referral",
            },
        }
        
        role_titles = title_overrides.get(role, {})
        if block.block_id in role_titles:
            modifications["title_override"] = role_titles[block.block_id]
        
        # Add credibility emphasis for professional roles
        if role in (VisitorRole.SCIENTIST, VisitorRole.CLINICIAN):
            if block.credibility_elements:
                modifications["show_credibility_badges"] = True
                modifications["credibility_placement"] = "prominent"
        
        # Interest-based content hints
        if profile.detected_interests:
            modifications["relevant_interests"] = profile.detected_interests[:3]
        
        return modifications

    def _sort_blocks(
        self, blocks: list[PersonalizedBlock]
    ) -> list[PersonalizedBlock]:
        """Sort blocks by priority and credibility boost."""
        priority_order = {
            ContentPriority.PRIMARY: 0,
            ContentPriority.SECONDARY: 1,
            ContentPriority.TERTIARY: 2,
            ContentPriority.HIDDEN: 3,
        }
        
        return sorted(
            blocks,
            key=lambda b: (
                priority_order.get(b.adjusted_priority, 4),
                -b.credibility_boost,
            ),
        )

    def _calculate_overall_credibility(
        self,
        blocks: list[PersonalizedBlock],
        profile: VisitorProfile,
    ) -> float:
        """Calculate overall credibility score for the personalized page."""
        if not blocks:
            return 0.0
            
        total_credibility = sum(b.credibility_boost for b in blocks)
        max_possible = len(blocks) * 1.0
        
        base_score = total_credibility / max_possible if max_possible > 0 else 0.0
        
        # Boost from visitor credibility signals
        visitor_boost = len(profile.credibility_signals) * 0.05
        
        return min(base_score + visitor_boost, 1.0)

    def _interpolate_priority(
        self,
        base: ContentPriority,
        target: ContentPriority,
    ) -> ContentPriority:
        """Interpolate between two priorities (move one step toward target)."""
        order = [
            ContentPriority.PRIMARY,
            ContentPriority.SECONDARY,
            ContentPriority.TERTIARY,
            ContentPriority.HIDDEN,
        ]
        base_idx = order.index(base)
        target_idx = order.index(target)
        
        if base_idx < target_idx:
            return order[base_idx + 1]
        elif base_idx > target_idx:
            return order[base_idx - 1]
        return base

    def _build_role_overrides(self) -> dict[VisitorRole, dict[str, ContentPriority]]:
        """Build role-specific priority overrides for content blocks."""
        return {
            VisitorRole.SCIENTIST: {
                "research_highlights": ContentPriority.PRIMARY,
                "publications": ContentPriority.SECONDARY,
                "pipeline_overview": ContentPriority.PRIMARY,
                "investor_relations": ContentPriority.HIDDEN,
                "patient_resources": ContentPriority.HIDDEN,
            },
            VisitorRole.INVESTOR: {
                "investor_relations": ContentPriority.PRIMARY,
                "pipeline_overview": ContentPriority.PRIMARY,
                "partnerships": ContentPriority.SECONDARY,
                "research_highlights": ContentPriority.TERTIARY,
                "publications": ContentPriority.HIDDEN,
            },
            VisitorRole.PATIENT: {
                "patient_resources": ContentPriority.PRIMARY,
                "pipeline_overview": ContentPriority.SECONDARY,
                "research_highlights": ContentPriority.HIDDEN,
                "publications": ContentPriority.HIDDEN,
                "investor_relations": ContentPriority.HIDDEN,
            },
            VisitorRole.CLINICIAN: {
                "patient_resources": ContentPriority.PRIMARY,
                "pipeline_overview": ContentPriority.PRIMARY,
                "publications": ContentPriority.SECONDARY,
                "research_highlights": ContentPriority.SECONDARY,
                "investor_relations": ContentPriority.HIDDEN,
            },
            VisitorRole.STUDENT: {
                "research_highlights": ContentPriority.PRIMARY,
                "publications": ContentPriority.PRIMARY,
                "careers": ContentPriority.SECONDARY,
                "investor_relations": ContentPriority.HIDDEN,
            },
        }


def get_personalized_layout(
    pages: Optional[list[str]] = None,
    search_queries: Optional[list[str]] = None,
) -> PersonalizedContent:
    """
    End-to-end convenience function: detect visitor and personalize content.
    
    Args:
        pages: Pages visited by the visitor.
        search_queries: Search queries entered.
        
    Returns:
        PersonalizedContent with role-based layout.
    """
    from brownbiotech.visitor_detection import create_visitor_profile
    
    profile = create_visitor_profile(
        pages=pages,
        search_queries=search_queries,
    )
    
    engine = ContentPersonalizationEngine()
    return engine.personalize(profile)


if __name__ == "__main__":
    print("=" * 60)
    print("BrownBioTech Content Personalization Engine - Demo")
    print("=" * 60)
    
    engine = ContentPersonalizationEngine()
    
    # Scenario 1: Scientist visitor
    from brownbiotech.visitor_detection import (
        BehavioralSignals,
        TrafficSource,
        VisitorDetectionEngine,
        VisitorRole,
    )
    
    scientist_signals = BehavioralSignals(
        pages_visited=[
            "https://brownbiotech.com/research",
            "https://brownbiotech.com/publications",
        ],
        search_queries=["crispr protocol"],
        return_visitor=True,
        previous_visits=5,
        time_on_site_seconds=180,
    )
    
    detection_engine = VisitorDetectionEngine()
    scientist_profile = detection_engine.classify_visitor(
        scientist_signals,
        traffic_source=TrafficSource.REFERRAL_ACADEMIC,
    )
    
    scientist_content = engine.personalize(scientist_profile)
    
    print(f"\n--- Scientist Layout ---")
    print(f"Role: {scientist_content.visitor_role.value}")
    print(f"Credibility Score: {scientist_content.credibility_score:.2%}")
    print(f"Signals: {scientist_content.personalization_signals}")
    print("Block Order:")
    for pb in scientist_content.blocks:
        mods = f" [{pb.modifications.get('title_override', '')}]" if pb.modifications.get('title_override') else ""
        print(f"  {pb.adjusted_priority.value:10} | {pb.block.block_id}{mods}")
    
    # Scenario 2: Investor visitor
    investor_signals = BehavioralSignals(
        pages_visited=["https://brownbiotech.com/investors"],
        search_queries=["brownbiotech stock"],
    )
    
    investor_profile = detection_engine.classify_visitor(
        investor_signals,
        traffic_source=TrafficSource.REFERRAL_NEWS,
    )
    
    investor_content = engine.personalize(investor_profile)
    
    print(f"\n--- Investor Layout ---")
    print(f"Role: {investor_content.visitor_role.value}")
    print(f"Credibility Score: {investor_content.credibility_score:.2%}")
    print("Block Order:")
    for pb in investor_content.blocks:
        mods = f" [{pb.modifications.get('title_override', '')}]" if pb.modifications.get('title_override') else ""
        print(f"  {pb.adjusted_priority.value:10} | {pb.block.block_id}{mods}")
    
    # Scenario 3: Quick convenience function
    print(f"\n--- Quick Convenience Test ---")
    quick_result = get_personalized_layout(
        pages=["https://brownbiotech.com/patients/clinical-trials"],
        search_queries=["cancer trial enrollment"],
    )
    print(f"Role: {quick_result.visitor_role.value}")
    print(f"Blocks: {[b.block.block_id for b in quick_result.blocks]}")
    
    print("\n" + "=" * 60)
```

---

## File 3: `brownbiotech/analytics_integration.py`

```python
"""
Analytics Integration for BrownBioTech Personalization Tracking.

Tracks personalization events, measures effectiveness, and provides
insights for continuous optimization of role-based content delivery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from brownbiotech.content_personalization import PersonalizedContent
from brownbiotech.visitor_detection import VisitorProfile, VisitorRole

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of personalization analytics events."""
    VISITOR_CLASSIFIED = "visitor_classified"
    CONTENT_PERSONALIZED = "content_personalized"
    BLOCK_VIEWED = "block_viewed"
    BLOCK_INTERACTED = "block_interacted"
    CREDIBILITY_SIGNAL_SHOWN = "credibility_signal_shown"
    PERSONALIZATION_OVERRIDDEN = "personalization_overridden"
    CONVERSION_INITIATED = "conversion_initiated"


class ConversionType(Enum):
    """Types of conversion events."""
    CONTACT_FORM = "contact_form"
    TRIAL_SIGNUP = "trial_signup"
    DOCUMENT_DOWNLOAD = "document_download"
    NEWSLETTER_SUBSCRIBE = "newsletter_subscribe"
    INVESTOR_INQUIRY = "investor_inquiry"
    HCP_REGISTRATION = "hcp_registration"


@dataclass
class AnalyticsEvent:
    """Represents a single analytics event."""
    event_type: EventType
    timestamp: datetime
    visitor_id: str
    session_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    role_at_event: Optional[VisitorRole] = None
    confidence_at_event: float = 0.0


@dataclass
class PersonalizationMetrics:
    """Aggregated metrics for personalization effectiveness."""
    total_visitors: int = 0
    classified_visitors: int = 0
    classification_rate: float = 0.0
    role_distribution: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    credibility_signal_impressions: int = 0
    block_engagement_by_role: dict[str, dict[str, int]] = field(default_factory=dict)
    conversions_by_role: dict[str, int] = field(default_factory=dict)
    conversion_rate_by_role: dict[str, float] = field(default_factory=dict)
    personalization_overrides: int = 0
    override_rate: float = 0.0


class PersonalizationAnalytics:
    """
    Analytics tracker for personalization system.
    
    Collects events, computes metrics, and provides reporting
    for personalization optimization decisions.
    """

    def __init__(self, flush_threshold: int = 100):
        """
        Initialize analytics tracker.
        
        Args:
            flush_threshold: Number of events before auto-flush.
        """
        self._events: list[AnalyticsEvent] = []
        self._flush_threshold = flush_threshold
        self._metrics_cache: Optional[PersonalizationMetrics] = None
        self._metrics_dirty = True

    def track_classification(
        self,
        visitor_id: str,
        session_id: str,
        profile: VisitorProfile,
    ) -> None:
        """
        Track a visitor classification event.
        
        Args:
            visitor_id: Unique visitor identifier.
            session_id: Current session identifier.
            profile: Resulting visitor profile.
        """
        event = AnalyticsEvent(
            event_type=EventType.VISITOR_CLASSIFIED,
            timestamp=datetime.utcnow(),
            visitor_id=visitor_id,
            session_id=session_id,
            role_at_event=profile.role,
            confidence_at_event=profile.confidence,
            properties={
                "traffic_source": profile.traffic_source.value,
                "interests": profile.detected_interests,
                "credibility_signals": profile.credibility_signals,
                "pages_visited_count": len(profile.signals.pages_visited),
            },
        )
        self._add_event(event)

    def track_personalization(
        self,
        visitor_id: str,
        session_id: str,
        content: PersonalizedContent,
    ) -> None:
        """
        Track content personalization delivery.
        
        Args:
            visitor_id: Unique visitor identifier.
            session_id: Current session identifier.
            content: Personalized content delivered.
        """
        block_ids = [b.block.block_id for b in content.blocks]
        modified_blocks = [
            b.block.block_id
            for b in content.blocks
            if b.modifications
        ]
        
        event = AnalyticsEvent(
            event_type=EventType.CONTENT_PERSONALIZED,
            timestamp=datetime.utcnow(),
            visitor_id=visitor_id,
            session_id=session_id,
            role_at_event=content.visitor_role,
            confidence_at_event=0.0,
            properties={
                "block_order": block_ids,
                "modified_blocks": modified_blocks,
                "credibility_score": content.credibility_score,
                "personalization_signals": content.personalization_signals,
                "block_count": len(content.blocks),
            },
        )
        self._add_event(event)

    def track_block_engagement(
        self,
        visitor_id: str,
        session_id: str,
        block_id: str,
        engagement_type: str,
        role: Optional[VisitorRole] = None,
    ) -> None:
        """
        Track user engagement with a content block.
        
        Args:
            visitor_id: Unique visitor identifier.
            session_id: Current session identifier.
            block_id: Identifier of the engaged block.
            engagement_type: Type of engagement (view, click, scroll, etc.).
            role: Visitor role at time of engagement.
        """
        event_type = (
            EventType.BLOCK_INTERACTED
            if engagement_type != "view"
            else EventType.BLOCK_VIEWED
        )
        
        event = AnalyticsEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            visitor_id=visitor_id,
            session_id=session_id,
            role_at_event=role,
            properties={
                "block_id": block_id,
                "engagement_type": engagement_type,
            },
        )
        self._add_event(event)

    def track_conversion(
        self,
        visitor_id: str,
        session_id: str,
        conversion_type: ConversionType,
        role: Optional[VisitorRole] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Track a conversion event.
        
        Args:
            visitor_id: Unique visitor identifier.
            session_id: Current session identifier.
            conversion_type: Type of conversion.
            role: Visitor role at time of conversion.
            metadata: Additional conversion metadata.
        """
        event = AnalyticsEvent(
            event_type=EventType.CONVERSION_INITIATED,
            timestamp=datetime.utcnow(),
            visitor_id=visitor_id,
            session_id=session_id,
            role_at_event=role,
            properties={
                "conversion_type": conversion_type.value,
                **(metadata or {}),
            },
        )
        self._add_event(event)

    def track_personalization_override(
        self,
        visitor_id: str,
        session_id: str,
        original_role: VisitorRole,
        override_role: VisitorRole,
        reason: str,
    ) -> None:
        """
        Track when a user overrides personalization (e.g., manual role selection).
        
        Args:
            visitor_id: Unique visitor identifier.
            session_id: Current session identifier.
            original_role: System-detected role.
            override_role: User-selected role.
            reason: Stated reason for override.
        """
        event = AnalyticsEvent(
            event_type=EventType.PERSONALIZATION_OVERRIDDEN,
            timestamp=datetime.utcnow(),
            visitor_id=visitor_id,
            session_id=session_id,
            role_at_event=override_role,
            properties={
                "original_role": original_role.value,
                "override_role": override_role.value,
                "reason": reason,
            },
        )
        self._add_event(event)

    def compute_metrics(self) -> PersonalizationMetrics:
        """
        Compute aggregated personalization metrics from collected events.
        
        Returns:
            PersonalizationMetrics with computed statistics.
        """
        if not self._metrics_dirty and self._metrics_cache is not None:
            return self._metrics_cache

        metrics = PersonalizationMetrics()
        
        # Unique visitors
        unique_visitors = set()
        classified_visitors = set()
        confidence_sum = 0.0
        confidence_count = 0
        
        for event in self._events:
            unique_visitors.add(event.visitor_id)
            
            if event.event_type == EventType.VISITOR_CLASSIFIED:
                classified_visitors.add(event.visitor_id)
                
                if event.role_at_event:
                    role_key = event.role_at_event.value
                    metrics.role_distribution[role_key] = (
                        metrics.role_distribution.get(role_key, 0) + 1
                    )
                    
                if event.confidence_at_event > 0:
                    confidence_sum += event.confidence_at_event
                    confidence_count += 1
                    
            elif event.event_type == EventType.CREDIBILITY_SIGNAL_SHOWN:
                metrics.credibility_signal_impressions += 1
                
            elif event.event_type == EventType.BLOCK_INTERACTED:
                if event.role_at_event:
                    role_key = event.role_at_event.value
                    if role_key not in metrics.block_engagement_by_role:
                        metrics.block_engagement_by_role[role_key] = {}
                    block_id = event.properties.get("block_id", "unknown")
                    metrics.block_engagement_by_role[role_key][block_id] = (
                        metrics.block_engagement_by_role[role_key].get(block_id, 0) + 1
                    )
                    
            elif event.event_type == EventType.CONVERSION_INITIATED:
                if event.role_at_event:
                    role_key = event.role_at_event.value
                    metrics.conversions_by_role[role_key] = (
                        metrics.conversions_by_role.get(role_key, 0) + 1
                    )
                    
            elif event.event_type == EventType.PERSONALIZATION_OVERRIDDEN:
                metrics.personalization_overrides += 1

        # Compute derived metrics
        metrics.total_visitors = len(unique_visitors)
        metrics.classified_visitors = len(classified_visitors)
        metrics.classification_rate = (
            metrics.classified_visitors / metrics.total_visitors
            if metrics.total_visitors > 0
            else 0.0
        )
        metrics.avg_confidence = (
            confidence_sum / confidence_count if confidence_count > 0 else 0.0
        )
        
        # Conversion rates by role
        for role_key, conversions in metrics.conversions_by_role.items():
            role_visitors = metrics.role_distribution.get(role_key, 0)
            metrics.conversion_rate_by_role[role_key] = (
                conversions / role_visitors if role_visitors > 0 else 0.0
            )
        
        # Override rate
        metrics.override_rate = (
            metrics.personalization_overrides / metrics.classified_visitors
            if metrics.classified_visitors > 0
            else 0.0
        )

        self._metrics_cache = metrics
        self._metrics_dirty = False
        return metrics

    def get_metrics_report(self) -> str:
        """
        Generate a human-readable metrics report.
        
        Returns:
            Formatted string with key personalization metrics.
        """
        metrics = self.compute_metrics()
        
        lines = [
            "=" * 50,
            "BrownBioTech Personalization Analytics Report",
            "=" * 50,
            f"Total Visitors: {metrics.total_visitors}",
            f"Classified Visitors: {metrics.classified_visitors}",
            f"Classification Rate: {metrics.classification_rate:.1%}",
            f"Average Confidence: {metrics.avg_confidence:.1%}",
            "",
            "Role Distribution:",
        ]
        
        for role, count in sorted(
            metrics.role_distribution.items(), key=lambda x: -x[1]
        ):
            pct = count / metrics.classified_visitors * 100 if metrics.classified_visitors else 0
            lines.append(f"  {role:15} {count:5} ({pct:.1f}%)")
        
        lines.append("")
        lines.append("Conversion Rates by Role:")
        
        for role, rate in sorted(
            metrics.conversion_rate_by_role.items(), key=lambda x: -x[1]
        ):
            conversions = metrics.conversions_by_role.get(role, 0)
            lines.append(f"  {role:15} {rate:.1%} ({conversions} conversions)")
        
        lines.append("")
        lines.append(f"Credibility Signal Impressions: {metrics.credibility_signal_impressions}")
        lines.append(f"Personalization Overrides: {metrics.personalization_overrides}")
        lines.append(f"Override Rate: {metrics.override_rate:.1%}")
        lines.append("=" * 50)
        
        return "\n".join(lines)

    def export_events(self) -> str:
        """
        Export collected events as JSON string.
        
        Returns:
            JSON-formatted string of all events.
        """
        export_data = []
        for event in self._events:
            export_data.append({
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "visitor_id": event.visitor_id,
                "session_id": event.session_id,
                "role": event.role_at_event.value if event.role_at_event else None,
                "confidence": event.confidence_at_event,
                "properties": event.properties,
            })
        return json.dumps(export_data, indent=2)

    def flush(self) -> list[AnalyticsEvent]:
        """
        Flush all collected events and reset state.
        
        Returns:
            List of flushed events.
        """
        flushed = self._events.copy()
        self._events.clear()
        self._metrics_cache = None
        self._metrics_dirty = True
        logger.info(f"Flushed {len(flushed)} analytics events")
        return flushed

    def _add_event(self, event: AnalyticsEvent) -> None:
        """Add event and check flush threshold."""
        self._events.append(event)
        self._metrics_dirty = True
        
        if len(self._events) >= self._flush_threshold:
            logger.info(
                f"Event threshold reached ({self._flush_threshold}), "
                "consider flushing"
            )


if __name__ == "__main__":
    print("=" * 60)
    print("BrownBioTech Analytics Integration - Demo")
    print("=" * 60)
    
    analytics = PersonalizationAnalytics(flush_threshold=50)
    
    # Simulate visitor sessions
    from brownbiotech.visitor_detection import (
        BehavioralSignals,
        TrafficSource,
        VisitorDetectionEngine,
        VisitorRole,
    )
    from brownbiotech.content_personalization import ContentPersonalizationEngine
    
    detection_engine = VisitorDetectionEngine()
    personalization_engine = ContentPersonalizationEngine()
    
    # Session 1: Scientist
    scientist_signals = BehavioralSignals(
        pages_visited=["/research", "/publications"],
        search_queries=["crispr"],
        return_visitor=True,
        previous_visits=4,
    )
    profile1 = detection_engine.classify_visitor(
        scientist_signals, traffic_source=TrafficSource.REFERRAL_ACADEMIC
    )
    content1 = personalization_engine.personalize(profile1)
    
    analytics.track_classification("v001", "s001", profile1)
    analytics.track_personalization("v001", "s001", content1)
    analytics.track_block_engagement("v001", "s001", "research_highlights", "click", profile1.role)
    analytics.track_conversion("v001", "s001", ConversionType.DOCUMENT_DOWNLOAD, profile1.role)
    
    # Session 2: Investor
    investor_signals = BehavioralSignals(
        pages_visited=["/investors"],
        search_queries=["valuation"],
    )
    profile2 = detection_engine.classify_visitor(
        investor_signals, traffic_source=TrafficSource.REFERRAL_NEWS
    )
    content2 = personalization_engine.personalize(profile2)
    
    analytics.track_classification("v002", "s002", profile2)
    analytics.track_personalization("v002", "s002", content2)
    analytics.track_conversion("v002", "s002", ConversionType.INVESTOR_INQUIRY, profile2.role)
    
    # Session 3: Patient
    patient_signals = BehavioralSignals(
        pages_visited=["/patients", "/trials"],
        search_queries=["treatment trial"],
    )
    profile3 = detection_engine.classify_visitor(patient_signals)
    content3 = personalization_engine.personalize(profile3)
    
    analytics.track_classification("v003", "s003", profile3)
    analytics.track_personalization("v003", "s003", content3)
    
    # Session 4: Override case
    profile4 = detection_engine.classify_visitor(
        BehavioralSignals(pages_visited=["/"])
    )
    analytics.track_classification("v004", "s004", profile4)
    analytics.track_personalization_override(
        "v004", "s004", VisitorRole.UNKNOWN, VisitorRole.SCIENTIST, "I am a researcher"
    )
    
    # Session 5: Another scientist for stats
    profile5 = detection_engine.classify_visitor(
        BehavioralSignals(
            pages_visited=["/bioinformatics"],
            search_queries=["genomics pipeline"],
            return_visitor=True,
            previous_visits=10,
        ),
        traffic_source=TrafficSource.ORGANIC_SEARCH,
    )
    analytics.track_classification("v005", "s005", profile5)
    analytics.track_conversion("v005", "s005", ConversionType.CONTACT_FORM, profile5.role)
    
    # Generate report
    print("\n" + analytics.get_metrics_report())
    
    # Export sample
    print("\nSample Event Export (first 2 events):")
    events_json = analytics.export_events()
    events_parsed = json.loads(events_json)
    print(json.dumps(events_parsed[:2], indent=2))
    
    print("\n" + "=" * 60)
```

---

## File 4: `brownbiotech/__init__.py`

```python
"""
BrownBioTech Platform - Personalization & Credibility Enhancement Module.

Iteration 21/100 - Credibility-First & Personalization Enhancement

This module provides:
- Visitor role detection based on behavioral signals
- Content personalization engine for role-based layouts
- Analytics integration for optimization tracking

Quick Start:
    from brownbiotech import get_personalized_layout
    
    layout = get_personalized_layout(
        pages=["/research", "/publications"],
        search_queries=["crispr protocol"],
    )
    
    for block in layout.blocks:
        print(f"{block.adjusted_priority.value}: {block.block.title}")
"""

from brownbiotech.analytics_integration import (
    ConversionType,
    EventType,
    PersonalizationAnalytics,
    PersonalizationMetrics,
)
from brownbiotech.content_personalization import (
    ContentBlock,
    ContentPersonalizationEngine,
    ContentPriority,
    CredibilityElementType,
    PersonalizedBlock,
    PersonalizedContent,
    get_personalized_layout,
)
from brownbiotech.visitor_detection import (
    BehavioralSignals,
    TrafficSource,
    VisitorDetectionEngine,
    VisitorProfile,
    VisitorRole,
    create_visitor_profile,
)

__version__ = "21.0.0"
__iteration__ = "21/100"

__all__ = [
    # Visitor Detection
    "VisitorRole",
    "TrafficSource",
    "BehavioralSignals",
    "VisitorProfile",
    "VisitorDetectionEngine",
    "create_visitor_profile",
    # Content Personalization
    "ContentPriority",
    "CredibilityElementType",
    "ContentBlock",
    "PersonalizedBlock",
    "PersonalizedContent",
    "ContentPersonalizationEngine",
    "get_personalized_layout",
    # Analytics
    "EventType",
    "ConversionType",
    "AnalyticsEvent",
    "PersonalizationMetrics",
    "PersonalizationAnalytics",
]
```

---

## Summary of Improvements

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `visitor_detection.py` | Role classification engine | Multi-signal scoring (URL patterns, queries, source, behavior), confidence thresholds, credibility signal detection |
| `content_personalization.py` | Dynamic content adaptation | Role-based priority overrides, credibility element boosting, title customization per role, interest-aware hints |
| `analytics_integration.py` | Tracking & optimization | Event collection, metrics computation (classification rate, conversion by role, override rate), JSON export |
| `__init__.py` | Public API surface | Clean re-exports, version tracking, quick-start documentation |

**Credibility-First Design Decisions:**
- Credibility elements (citations, trial refs, certifications) are **prominently surfaced** for scientist/clinician roles
- Credibility score is computed per-page and can trigger additional trust signals
- Visitor credibility signals (return visits, deep engagement) boost personalization quality

**Personalization Approach:**
- Weighted multi-signal scoring prevents over-reliance on any single indicator
- Confidence-gated overrides prevent miscategorization from degrading experience
- Override tracking enables continuous model improvement