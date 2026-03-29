# BrownBioTech Iteration 7→8: Investor-Ready Validation Dashboard

## File 1: `brownbiotech/agents/multiomics/PathwayEnrichmentViz.py`

```python
"""
PathwayEnrichmentViz.py - Visual-Finance Synergy Module

Generates DGAT1/YARS2 pathway enrichment visualizations with embedded
financial metrics for investor presentations. Bridges multiomics analysis
with commercial viability indicators.

BrownBioTech Iteration 7→8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TherapeuticArea(Enum):
    """Therapeutic classification for pathway targets."""
    METABOLIC = "metabolic"
    MITOCHONDRIAL = "mitochondrial"
    ONCOLOGY = "oncology"
    RARE_DISEASE = "rare_disease"
    AUTOIMMUNE = "autoimmune"


class FinancialTier(Enum):
    """Investor confidence tier based on validation metrics."""
    TIER_1_PROVEN = "tier_1_proven"          # >$500M TAM, p<0.001
    TIER_2_VALIDATED = "tier_2_validated"    # $100-500M TAM, p<0.01
    TIER_3_PROMISING = "tier_3_promising"    # $50-100M TAM, p<0.05
    TIER_4_EXPLORATORY = "tier_4_exploratory"  # <$50M TAM, p<0.1


@dataclass
class PathwayTarget:
    """Represents a therapeutic pathway target with validation data."""
    gene_symbol: str
    pathway_id: str
    pathway_name: str
    therapeutic_area: TherapeuticArea
    
    # Enrichment statistics
    p_value: float
    fold_enrichment: float
    adjusted_p_value: float = 0.0
    
    # Financial metrics
    tam_usd_millions: float = 0.0
    development_cost_millions: float = 0.0
    projected_roi: float = 0.0
    time_to_market_years: float = 0.0
    
    # Validation status
    validation_score: float = 0.0  # 0-100 composite
    financial_tier: Optional[FinancialTier] = None
    
    def __post_init__(self) -> None:
        """Calculate derived metrics after initialization."""
        if self.adjusted_p_value == 0.0:
            self.adjusted_p_value = self.p_value
        self._calculate_validation_score()
        self._assign_financial_tier()
    
    def _calculate_validation_score(self) -> None:
        """Calculate composite validation score (0-100)."""
        # Statistical significance component (0-40 points)
        if self.adjusted_p_value < 0.001:
            stat_score = 40.0
        elif self.adjusted_p_value < 0.01:
            stat_score = 30.0
        elif self.adjusted_p_value < 0.05:
            stat_score = 20.0
        else:
            stat_score = 10.0
        
        # Fold enrichment component (0-30 points)
        fe_score = min(30.0, self.fold_enrichment * 10)
        
        # Financial component (0-30 points)
        if self.tam_usd_millions > 500:
            fin_score = 30.0
        elif self.tam_usd_millions > 100:
            fin_score = 20.0
        elif self.tam_usd_millions > 50:
            fin_score = 10.0
        else:
            fin_score = 5.0
        
        self.validation_score = stat_score + fe_score + fin_score
    
    def _assign_financial_tier(self) -> None:
        """Assign investor confidence tier based on metrics."""
        if (self.tam_usd_millions > 500 and 
            self.adjusted_p_value < 0.001 and
            self.validation_score >= 80):
            self.financial_tier = FinancialTier.TIER_1_PROVEN
        elif (self.tam_usd_millions > 100 and 
              self.adjusted_p_value < 0.01 and
              self.validation_score >= 60):
            self.financial_tier = FinancialTier.TIER_2_VALIDATED
        elif (self.tam_usd_millions > 50 and 
              self.adjusted_p_value < 0.05 and
              self.validation_score >= 40):
            self.financial_tier = FinancialTier.TIER_3_PROMISING
        else:
            self.financial_tier = FinancialTier.TIER_4_EXPLORATORY


@dataclass
class EnrichmentResult:
    """Container for pathway enrichment analysis results."""
    targets: List[PathwayTarget] = field(default_factory=list)
    analysis_timestamp: str = ""
    total_pathways_tested: int = 0
    fdr_method: str = "benjamini_hochberg"
    
    @property
    def significant_targets(self) -> List[PathwayTarget]:
        """Return targets with adjusted p-value < 0.05."""
        return [t for t in self.targets if t.adjusted_p_value < 0.05]
    
    @property
    def tier_distribution(self) -> Dict[FinancialTier, int]:
        """Count targets by financial tier."""
        dist: Dict[FinancialTier, int] = {tier: 0 for tier in FinancialTier}
        for target in self.targets:
            if target.financial_tier:
                dist[target.financial_tier] += 1
        return dist
    
    @property
    def aggregate_tam(self) -> float:
        """Total addressable market across all targets."""
        return sum(t.tam_usd_millions for t in self.targets)


class PathwayEnrichmentViz:
    """
    Generates investor-ready pathway enrichment visualizations.
    
    Combines traditional pathway analysis outputs with financial metrics
    to create compelling visual narratives for investor presentations.
    
    Example:
        >>> viz = PathwayEnrichmentViz()
        >>> result = viz.create_sample_dgat1_results()
        >>> fig = viz.generate_investor_dashboard(result)
        >>> fig.savefig("investor_dashboard.png", dpi=300)
    """
    
    # Color scheme for financial tiers
    TIER_COLORS = {
        FinancialTier.TIER_1_PROVEN: "#2E7D32",      # Dark green
        FinancialTier.TIER_2_VALIDATED: "#66BB6A",   # Light green
        FinancialTier.TIER_3_PROMISING: "#FFA726",   # Orange
        FinancialTier.TIER_4_EXPLORATORY: "#EF5350", # Red
    }
    
    TIER_LABELS = {
        FinancialTier.TIER_1_PROVEN: "Tier 1: Proven",
        FinancialTier.TIER_2_VALIDATED: "Tier 2: Validated",
        FinancialTier.TIER_3_PROMISING: "Tier 3: Promising",
        FinancialTier.TIER_4_EXPLORATORY: "Tier 4: Exploratory",
    }
    
    def __init__(self, figsize: Tuple[float, float] = (14, 10)) -> None:
        """Initialize visualizer with default figure size."""
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def generate_investor_dashboard(
        self,
        result: EnrichmentResult,
        title: str = "BrownBioTech Pathway Validation Dashboard"
    ) -> plt.Figure:
        """
        Generate comprehensive investor-ready dashboard.
        
        Creates a multi-panel figure showing:
        1. Enrichment volcano plot with financial tier coloring
        2. Validation score distribution
        3. TAM breakdown by therapeutic area
        4. Tier distribution summary
        
        Args:
            result: Enrichment analysis results
            title: Dashboard title
            
        Returns:
            matplotlib Figure object
            
        Raises:
            ValueError: If result has no targets
        """
        if not result.targets:
            raise ValueError("Cannot generate dashboard: no targets in results")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        self._plot_volcano_with_tiers(axes[0, 0], result)
        self._plot_validation_scores(axes[0, 1], result)
        self._plot_tam_breakdown(axes[1, 0], result)
        self._plot_tier_summary(axes[1, 1], result)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    
    def _plot_volcano_with_tiers(
        self,
        ax: plt.Axes,
        result: EnrichmentResult
    ) -> None:
        """Plot enrichment volcano with financial tier coloring."""
        ax.set_title("Pathway Enrichment by Investor Tier", fontsize=11)
        
        for tier in FinancialTier:
            tier_targets = [
                t for t in result.targets if t.financial_tier == tier
            ]
            if not tier_targets:
                continue
            
            x_vals = [t.fold_enrichment for t in tier_targets]
            y_vals = [-np.log10(max(t.adjusted_p_value, 1e-10)) for t in tier_targets]
            
            ax.scatter(
                x_vals, y_vals,
                c=self.TIER_COLORS[tier],
                label=self.TIER_LABELS[tier],
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add significance threshold line
        ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.text(
            ax.get_xlim()[0] + 0.1, -np.log10(0.05) + 0.1,
            "p = 0.05", fontsize=8, color='gray'
        )
        
        ax.set_xlabel("Fold Enrichment", fontsize=9)
        ax.set_ylabel("-log10(Adjusted P-value)", fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        
        # Annotate top targets
        top_targets = sorted(result.targets, key=lambda t: t.validation_score, reverse=True)[:3]
        for target in top_targets:
            ax.annotate(
                target.gene_symbol,
                (target.fold_enrichment, -np.log10(max(target.adjusted_p_value, 1e-10))),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold'
            )
    
    def _plot_validation_scores(
        self,
        ax: plt.Axes,
        result: EnrichmentResult
    ) -> None:
        """Plot validation score distribution with tier coloring."""
        ax.set_title("Target Validation Scores", fontsize=11)
        
        sorted_targets = sorted(
            result.targets,
            key=lambda t: t.validation_score,
            reverse=True
        )
        
        genes = [t.gene_symbol for t in sorted_targets]
        scores = [t.validation_score for t in sorted_targets]
        colors = [
            self.TIER_COLORS[t.financial_tier] 
            for t in sorted_targets
        ]
        
        bars = ax.barh(range(len(genes)), scores, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes, fontsize=8)
        ax.set_xlabel("Validation Score (0-100)", fontsize=9)
        ax.set_xlim(0, 105)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height()/2,
                f"{score:.0f}",
                va='center',
                fontsize=7
            )
        
        # Add tier threshold lines
        ax.axvline(x=80, color=self.TIER_COLORS[FinancialTier.TIER_1_PROVEN], 
                   linestyle=':', alpha=0.5)
        ax.axvline(x=60, color=self.TIER_COLORS[FinancialTier.TIER_2_VALIDATED], 
                   linestyle=':', alpha=0.5)
    
    def _plot_tam_breakdown(
        self,
        ax: plt.Axes,
        result: EnrichmentResult
    ) -> None:
        """Plot TAM breakdown by therapeutic area."""
        ax.set_title("Market Opportunity by Therapeutic Area", fontsize=11)
        
        tam_by_area: Dict[str, float] = {}
        for target in result.targets:
            area = target.therapeutic_area.value
            tam_by_area[area] = tam_by_area.get(area, 0) + target.tam_usd_millions
        
        if not tam_by_area:
            ax.text(0.5, 0.5, "No TAM data", ha='center', va='center')
            return
        
        areas = list(tam_by_area.keys())
        tams = list(tam_by_area.values())
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(areas)))
        
        wedges, texts, autotexts = ax.pie(
            tams,
            labels=[a.capitalize() for a in areas],
            autopct=lambda pct: f"${pct*sum(tams)/100:.0f}M" if pct > 5 else "",
            colors=colors,
            startangle=90,
            textprops={'fontsize': 8}
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(7)
    
    def _plot_tier_summary(
        self,
        ax: plt.Axes,
        result: EnrichmentResult
    ) -> None:
        """Plot tier distribution summary with key metrics."""
        ax.set_title("Investor Confidence Summary", fontsize=11)
        ax.axis('off')
        
        tier_dist = result.tier_distribution
        total = sum(tier_dist.values()) or 1
        
        # Build summary text
        y_pos = 0.95
        line_height = 0.08
        
        # Header
        ax.text(0.05, y_pos, "Pipeline Distribution", fontsize=11, 
                fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height * 1.5
        
        # Tier counts
        for tier in FinancialTier:
            count = tier_dist[tier]
            pct = (count / total) * 100
            color = self.TIER_COLORS[tier]
            
            ax.text(0.05, y_pos, f"■ {self.TIER_LABELS[tier]}:", 
                    fontsize=9, color=color, fontweight='bold', transform=ax.transAxes)
            ax.text(0.55, y_pos, f"{count} targets ({pct:.0f}%)", 
                    fontsize=9, transform=ax.transAxes)
            y_pos -= line_height
        
        y_pos -= line_height * 0.5
        
        # Key metrics
        ax.text(0.05, y_pos, "Key Metrics", fontsize=11, 
                fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height * 1.2
        
        metrics = [
            f"Total TAM: ${result.aggregate_tam:,.0f}M",
            f"Significant Pathways: {len(result.significant_targets)}",
            f"Pathways Tested: {result.total_pathways_tested}",
            f"Hit Rate: {len(result.significant_targets)/max(result.total_pathways_tested, 1)*100:.1f}%",
        ]
        
        for metric in metrics:
            ax.text(0.05, y_pos, f"• {metric}", fontsize=9, transform=ax.transAxes)
            y_pos -= line_height
    
    def create_sample_dgat1_results(self) -> EnrichmentResult:
        """
        Create sample enrichment results for DGAT1/YARS2 pathways.
        
        Useful for testing and demonstration purposes.
        
        Returns:
            EnrichmentResult with realistic DGAT1/YARS2 pathway data
        """
        targets = [
            PathwayTarget(
                gene_symbol="DGAT1",
                pathway_id="KEGG:00100",
                pathway_name="Steroid Biosynthesis",
                therapeutic_area=TherapeuticArea.METABOLIC,
                p_value=0.0001,
                fold_enrichment=4.2,
                tam_usd_millions=850,
                development_cost_millions=120,
                projected_roi=6.1,
                time_to_market_years=4.5
            ),
            PathwayTarget(
                gene_symbol="YARS2",
                pathway_id="KEGG:00190",
                pathway_name="Oxidative Phosphorylation",
                therapeutic_area=TherapeuticArea.MITOCHONDRIAL,
                p_value=0.0005,
                fold_enrichment=3.8,
                tam_usd_millions=420,
                development_cost_millions=95,
                projected_roi=4.4,
                time_to_market_years=5.0
            ),
            PathwayTarget(
                gene_symbol="ACSL1",
                pathway_id="KEGG:00061",
                pathway_name="Fatty Acid Biosynthesis",
                therapeutic_area=TherapeuticArea.METABOLIC,
                p_value=0.003,
                fold_enrichment=2.9,
                tam_usd_millions=320,
                development_cost_millions=85,
                projected_roi=3.8,
                time_to_market_years=4.0
            ),
            PathwayTarget(
                gene_symbol="MT-ATP6",
                pathway_id="KEGG:00190",
                pathway_name="Oxidative Phosphorylation",
                therapeutic_area=TherapeuticArea.MITOCHONDRIAL,
                p_value=0.008,
                fold_enrichment=2.5,
                tam_usd_millions=180,
                development_cost_millions=70,
                projected_roi=2.6,
                time_to_market_years=5.5
            ),
            PathwayTarget(
                gene_symbol="CPT1A",
                pathway_id="KEGG:00071",
                pathway_name="Fatty Acid Degradation",
                therapeutic_area=TherapeuticArea.METABOLIC,
                p_value=0.025,
                fold_enrichment=2.1,
                tam_usd_millions=95,
                development_cost_millions=60,
                projected_roi=1.6,
                time_to_market_years=4.5
            ),
            PathwayTarget(
                gene_symbol="SLC25A1",
                pathway_id="KEGG:00020",
                pathway_name="Citrate Cycle",
                therapeutic_area=TherapeuticArea.METABOLIC,
                p_value=0.045,
                fold_enrichment=1.8,
                tam_usd_millions=65,
                development_cost_millions=55,
                projected_roi=1.2,
                time_to_market_years=5.0
            ),
            PathwayTarget(
                gene_symbol="HADHA",
                pathway_id="KEGG:00071",
                pathway_name="Fatty Acid Degradation",
                therapeutic_area=TherapeuticArea.RARE_DISEASE,
                p_value=0.08,
                fold_enrichment=1.5,
                tam_usd_millions=35,
                development_cost_millions=45,
                projected_roi=0.8,
                time_to_market_years=6.0
            ),
        ]
        
        return EnrichmentResult(
            targets=targets,
            analysis_timestamp=pd.Timestamp.now().isoformat(),
            total_pathways_tested=150,
            fdr_method="benjamini_hochberg"
        )
    
    def export_metrics_dataframe(self, result: EnrichmentResult) -> pd.DataFrame:
        """
        Export pathway metrics as investor-ready DataFrame.
        
        Args:
            result: Enrichment analysis results
            
        Returns:
            DataFrame with all metrics formatted for export
        """
        data = []
        for target in result.targets:
            data.append({
                "Gene": target.gene_symbol,
                "Pathway": target.pathway_name,
                "Pathway ID": target.pathway_id,
                "Therapeutic Area": target.therapeutic_area.value.capitalize(),
                "P-value": f"{target.p_value:.2e}",
                "Adj. P-value": f"{target.adjusted_p_value:.2e}",
                "Fold Enrichment": f"{target.fold_enrichment:.1f}x",
                "Validation Score": target.validation_score,
                "Investor Tier": self.TIER_LABELS.get(
                    target.financial_tier, "Unknown"
                ) if target.financial_tier else "Unknown",
                "TAM ($M)": f"${target.tam_usd_millions:,.0f}",
                "Dev Cost ($M)": f"${target.development_cost_millions:,.0f}",
                "Projected ROI": f"{target.projected_roi:.1f}x",
                "Time to Market (yr)": target.time_to_market_years,
            })
        
        return pd.DataFrame(data)


def main() -> None:
    """Run demonstration of PathwayEnrichmentViz module."""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Initializing PathwayEnrichmentViz...")
    viz = PathwayEnrichmentViz()
    
    logger.info("Creating sample DGAT1/YARS2 results...")
    result = viz.create_sample_dgat1_results()
    
    logger.info(f"Generated {len(result.targets)} pathway targets")
    logger.info(f"Aggregate TAM: ${result.aggregate_tam:,.0f}M")
    logger.info(f"Significant pathways: {len(result.significant_targets)}")
    
    # Generate dashboard
    logger.info("Generating investor dashboard...")
    fig = viz.generate_investor_dashboard(
        result,
        title="BrownBioTech DGAT1/YARS2 Pathway Validation"
    )
    
    # Save figure
    output_path = "investor_dashboard.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Dashboard saved to {output_path}")
    
    # Export metrics
    df = viz.export_metrics_dataframe(result)
    csv_path = "pathway_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Metrics exported to {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BROWNBBIOTECH PATHWAY VALIDATION SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    plt.close(fig)


if __name__ == "__main__":
    main()
```

---

## File 2: `brownbiotech/dashboard/investor_metrics.py`

```python
"""
investor_metrics.py - Pitch Data Extraction Module

Extracts and formats key investor metrics from BrownBioTech pipeline
analysis results for pitch decks and investor communications.

BrownBioTech Iteration 7→8
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PitchSection(Enum):
    """Sections of a standard biotech pitch deck."""
    PROBLEM = "problem"
    SOLUTION = "solution"
    MARKET = "market"
    TECHNOLOGY = "technology"
    PIPELINE = "pipeline"
    TRACTION = "traction"
    TEAM = "team"
    FINANCIALS = "financials"
    ASK = "ask"


class InvestorStage(Enum):
    """Investment stage classifications."""
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    GROWTH = "growth"


@dataclass
class PipelineAsset:
    """Individual pipeline asset with investment metrics."""
    name: str
    target: str
    indication: str
    stage: str  # Discovery, Preclinical, Phase I, Phase II, Phase III
    probability_success: float  # 0-1
    peak_sales_millions: float
    development_cost_millions: float
    npv_millions: float = 0.0
    timeline_years: float = 0.0
    
    def __post_init__(self) -> None:
        """Calculate NPV if not provided."""
        if self.npv_millions == 0.0 and self.probability_success > 0:
            # Simplified NPV calculation (10% discount rate, 10-year horizon)
            discount_rate = 0.10
            years = 10
            annual_sales = self.peak_sales_millions * 0.6  # Average over lifecycle
            pv_sales = annual_sales * (1 - (1 + discount_rate)**-years) / discount_rate
            self.npv_millions = (pv_sales * self.probability_success) - self.development_cost_millions
        self.timeline_years = self._estimate_timeline()
    
    def _estimate_timeline(self) -> float:
        """Estimate time to market based on stage."""
        stage_timelines = {
            "Discovery": 8.0,
            "Preclinical": 6.0,
            "Phase I": 5.0,
            "Phase II": 3.5,
            "Phase III": 2.0,
        }
        return stage_timelines.get(self.stage, 5.0)


@dataclass
class InvestorMetrics:
    """
    Comprehensive investor metrics container.
    
    Aggregates pipeline, market, and financial data into
    investor-ready formats.
    """
    company_name: str = "BrownBioTech"
    pitch_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    investment_stage: InvestorStage = InvestorStage.SEED
    pipeline_assets: List[PipelineAsset] = field(default_factory=list)
    
    # Market metrics
    total_tam_millions: float = 0.0
    serviceable_tam_millions: float = 0.0
    market_growth_rate: float = 0.0  # CAGR percentage
    
    # Financial metrics
    pre_money_valuation_millions: float = 0.0
    raise_amount_millions: float = 0.0
    runway_months: int = 18
    burn_rate_millions_per_month: float = 0.0
    
    # Technical metrics
    patents_pending: int = 0
    patents_granted: int = 0
    publications: int = 0
    data_points_generated: int = 0
    
    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.burn_rate_millions_per_month > 0:
            self.runway_months = int(
                self.raise_amount_millions / self.burn_rate_millions_per_month
            )
    
    @property
    def total_pipeline_npv(self) -> float:
        """Sum of NPV across all pipeline assets."""
        return sum(a.npv_millions for a in self.pipeline_assets)
    
    @property
    def risk_adjusted_peak_sales(self) -> float:
        """Probability-weighted peak sales."""
        return sum(
            a.peak_sales_millions * a.probability_success 
            for a in self.pipeline_assets
        )
    
    @property
    def weighted_probability_success(self) -> float:
        """Portfolio-weighted probability of success."""
        if not self.pipeline_assets:
            return 0.0
        total_npv = sum(abs(a.npv_millions) for a in self.pipeline_assets)
        if total_npv == 0:
            return 0.0
        return sum(
            a.probability_success * abs(a.npv_millions) / total_npv
            for a in self.pipeline_assets
        )
    
    @property
    def npv_per_dollar_raised(self) -> float:
        """Value creation efficiency metric."""
        if self.raise_amount_millions == 0:
            return 0.0
        return self.total_pipeline_npv / self.raise_amount_millions


class InvestorMetricsExporter:
    """
    Exports investor metrics in various formats.
    
    Supports JSON, CSV, and formatted text outputs suitable
    for pitch decks and data rooms.
    
    Example:
        >>> exporter = InvestorMetricsExporter()
        >>> metrics = exporter.create_sample_metrics()
        >>> exporter.to_json(metrics, "metrics.json")
        >>> exporter.to_pitch_text(metrics)
    """
    
    def create_sample_metrics(self) -> InvestorMetrics:
        """
        Create sample metrics for DGAT1/YARS2 pipeline.
        
        Returns:
            InvestorMetrics with realistic BrownBioTech data
        """
        assets = [
            PipelineAsset(
                name="BBT-001",
                target="DGAT1",
                indication="NAFLD/NASH",
                stage="Preclinical",
                probability_success=0.15,
                peak_sales_millions=1200,
                development_cost_millions=150
            ),
            PipelineAsset(
                name="BBT-002",
                target="YARS2",
                indication="Mitochondrial Myopathy",
                stage="Discovery",
                probability_success=0.08,
                peak_sales_millions=450,
                development_cost_millions=120
            ),
            PipelineAsset(
                name="BBT-003",
                target="ACSL1",
                indication="Metabolic Dysfunction",
                stage="Discovery",
                probability_success=0.05,
                peak_sales_millions=350,
                development_cost_millions=100
            ),
        ]
        
        return InvestorMetrics(
            company_name="BrownBioTech",
            investment_stage=InvestorStage.SEED,
            pipeline_assets=assets,
            total_tam_millions=2800,
            serviceable_tam_millions=850,
            market_growth_rate=12.5,
            pre_money_valuation_millions=15,
            raise_amount_millions=5,
            runway_months=24,
            burn_rate_millions_per_month=0.21,
            patents_pending=3,
            patents_granted=1,
            publications=8,
            data_points_generated=2500000
        )
    
    def to_json(
        self,
        metrics: InvestorMetrics,
        filepath: Optional[str] = None,
        indent: int = 2
    ) -> str:
        """
        Export metrics to JSON format.
        
        Args:
            metrics: InvestorMetrics to export
            filepath: Optional path to save JSON file
            indent: JSON indentation level
            
        Returns:
            JSON string
        """
        data = {
            "company": metrics.company_name,
            "pitch_date": metrics.pitch_date,
            "investment_stage": metrics.investment_stage.value,
            "market": {
                "total_tam_millions": metrics.total_tam_millions,
                "serviceable_tam_millions": metrics.serviceable_tam_millions,
                "market_growth_rate_pct": metrics.market_growth_rate,
            },
            "financials": {
                "pre_money_valuation_millions": metrics.pre_money_valuation_millions,
                "raise_amount_millions": metrics.raise_amount_millions,
                "runway_months": metrics.runway_months,
                "burn_rate_millions_per_month": metrics.burn_rate_millions_per_month,
            },
            "pipeline": {
                "total_assets": len(metrics.pipeline_assets),
                "total_npv_millions": round(metrics.total_pipeline_npv, 1),
                "risk_adjusted_peak_sales_millions": round(
                    metrics.risk_adjusted_peak_sales, 1
                ),
                "weighted_probability_success": round(
                    metrics.weighted_probability_success, 3
                ),
                "npv_per_dollar_raised": round(metrics.npv_per_dollar_raised, 1),
                "assets": [
                    {
                        "name": a.name,
                        "target": a.target,
                        "indication": a.indication,
                        "stage": a.stage,
                        "probability_success": a.probability_success,
                        "peak_sales_millions": a.peak_sales_millions,
                        "npv_millions": round(a.npv_millions, 1),
                        "timeline_years": a.timeline_years,
                    }
                    for a in metrics.pipeline_assets
                ]
            },
            "intellectual_property": {
                "patents_pending": metrics.patents_pending,
                "patents_granted": metrics.patents_granted,
                "publications": metrics.publications,
                "data_points_generated": metrics.data_points_generated,
            }
        }
        
        json_str = json.dumps(data, indent=indent)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Metrics exported to {filepath}")
        
        return json_str
    
    def to_dataframe(self, metrics: InvestorMetrics) -> pd.DataFrame:
        """
        Export pipeline assets to DataFrame.
        
        Args:
            metrics: InvestorMetrics to export
            
        Returns:
            DataFrame with pipeline asset details
        """
        rows = []
        for asset in metrics.pipeline_assets:
            rows.append({
                "Asset": asset.name,
                "Target": asset.target,
                "Indication": asset.indication,
                "Stage": asset.stage,
                "Prob. Success": f"{asset.probability_success:.0%}",
                "Peak Sales ($M)": f"${asset.peak_sales_millions:,.0f}",
                "Risk-Adj Sales ($M)": f"${asset.peak_sales_millions * asset.probability_success:,.0f}",
                "NPV ($M)": f"${asset.npv_millions:,.0f}",
                "Timeline (yr)": asset.timeline_years,
            })
        return pd.DataFrame(rows)
    
    def to_pitch_text(self, metrics: InvestorMetrics) -> str:
        """
        Generate formatted text for pitch deck inclusion.
        
        Args:
            metrics: InvestorMetrics to format
            
        Returns:
            Formatted string with key metrics
        """
        lines = [
            "=" * 55,
            f"  {metrics.company_name.upper()} - INVESTOR SUMMARY",
            f"  {metrics.pitch_date}",
            "=" * 55,
            "",
            "MARKET OPPORTUNITY",
            "-" * 40,
            f"  Total Addressable Market:    ${metrics.total_tam_millions:,.0f}M",
            f"  Serviceable TAM:             ${metrics.serviceable_tam_millions:,.0f}M",
            f"  Market CAGR:                 {metrics.market_growth_rate:.1f}%",
            "",
            "PIPELINE VALUE",
            "-" * 40,
            f"  Pipeline Assets:             {len(metrics.pipeline_assets)}",
            f"  Total NPV:                   ${metrics.total_pipeline_npv:,.0f}M",
            f"  Risk-Adj Peak Sales:         ${metrics.risk_adjusted_peak_sales:,.0f}M",
            f"  Wtd. Probability of Success: {metrics.weighted_probability_success:.1%}",
            "",
            "INVESTMENT TERMS",
            "-" * 40,
            f"  Stage:                       {metrics.investment_stage.value.replace('_', ' ').title()}",
            f"  Pre-Money Valuation:         ${metrics.pre_money_valuation_millions:,.0f}M",
            f"  Raising:                     ${metrics.raise_amount_millions:,.0f}M",
            f"  Runway:                      {metrics.runway_months} months",
            f"  NPV/$ Raised:                {metrics.npv_per_dollar_raised:.1f}x",
            "",
            "INTELLECTUAL PROPERTY",
            "-" * 40,
            f"  Patents Granted:             {metrics.patents_granted}",
            f"  Patents Pending:             {metrics.patents_pending}",
            f"  Publications:                {metrics.publications}",
            f"  Data Points:                 {metrics.data_points_generated:,}",
            "",
        ]
        
        # Add pipeline table
        lines.append("PIPELINE DETAIL")
        lines.append("-" * 40)
        lines.append(f"  {'Asset':<10} {'Target':<8} {'Stage':<12} {'NPV':>10}")
        lines.append(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*10}")
        
        for asset in metrics.pipeline_assets:
            lines.append(
                f"  {asset.name:<10} {asset.target:<8} {asset.stage:<12} ${asset.npv_millions:>8,.0f}M"
            )
        
        lines.append("")
        lines.append("=" * 55)
        
        return "\n".join(lines)
    
    def generate_one_liner(self, metrics: InvestorMetrics) -> str:
        """
        Generate investor one-liner for elevator pitch.
        
        Args:
            metrics: InvestorMetrics to summarize
            
        Returns:
            Concise one-sentence pitch
        """
        return (
            f"{metrics.company_name} is a {metrics.investment_stage.value.replace('_', '-')}-stage "
            f"biotech targeting ${metrics.total_tam_millions:,.0f}M in metabolic/mitochondrial "
            f"diseases with {len(metrics.pipeline_assets)} pipeline assets valued at "
            f"${metrics.total_pipeline_npv:,.0f}M NPV, raising ${metrics.raise_amount_millions:,.0f}M."
        )


def main() -> None:
    """Run demonstration of investor_metrics module."""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Initializing InvestorMetricsExporter...")
    exporter = InvestorMetricsExporter()
    
    logger.info("Creating sample metrics...")
    metrics = exporter.create_sample_metrics()
    
    # Export to JSON
    json_path = "investor_metrics.json"
    exporter.to_json(metrics, json_path)
    
    # Export to CSV
    df = exporter.to_dataframe(metrics)
    csv_path = "pipeline_assets.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Pipeline exported to {csv_path}")
    
    # Generate pitch text
    pitch_text = exporter.to_pitch_text(metrics)
    print("\n" + pitch_text)
    
    # Generate one-liner
    one_liner = exporter.generate_one_liner(metrics)
    print(f"\nELEVATOR PITCH:\n{one_liner}\n")


if __name__ == "__main__":
    main()
```

---

## File 3: `brownbiotech/agents/multiomics/enrichment_engine.py` (Extension)

```python
"""
enrichment_engine.py - Extended with Financial Integration

Core pathway enrichment engine extended to support financial metric
calculation and investor-ready output generation.

BrownBioTech Iteration 7→8 Extension
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class FinancialPathwayCalculator:
    """
    Calculates financial metrics for pathway targets.
    
    Integrates with enrichment results to provide
    investment-relevant valuations.
    """
    
    # Default TAM estimates by therapeutic area (millions USD)
    DEFAULT_TAM_ESTIMATES = {
        "metabolic": 2800,
        "mitochondrial": 850,
        "oncology": 4500,
        "rare_disease": 420,
        "autoimmune": 3200,
    }
    
    # Default development costs by stage (millions USD)
    DEV_COST_BY_STAGE = {
        "discovery": 30,
        "preclinical": 80,
        "phase_i": 150,
        "phase_ii": 300,
        "phase_iii": 500,
    }
    
    def __init__(
        self,
        tam_estimates: Optional[Dict[str, float]] = None,
        discount_rate: float = 0.12
    ) -> None:
        """
        Initialize financial calculator.
        
        Args:
            tam_estimates: Custom TAM estimates by therapeutic area
            discount_rate: Annual discount rate for NPV calculations
        """
        self.tam_estimates = tam_estimates or self.DEFAULT_TAM_ESTIMATES
        self.discount_rate = discount_rate
    
    def estimate_tam(
        self,
        therapeutic_area: str,
        market_share_assumption: float = 0.10
    ) -> float:
        """
        Estimate addressable market for a therapeutic area.
        
        Args:
            therapeutic_area: Therapeutic classification
            market_share_assumption: Assumed market capture (default 10%)
            
        Returns:
            Estimated serviceable market in millions USD
        """
        base_tam = self.tam_estimates.get(
            therapeutic_area.lower(),
            self.tam_estimates.get("metabolic", 500)  # Default fallback
        )
        return base_tam * market_share_assumption
    
    def calculate_npv(
        self,
        peak_sales_millions: float,
        probability_success: float,
        development_cost_millions: float,
        years_to_peak: float = 8.0,
        patent_life_years: float = 10.0
    ) -> float:
        """
        Calculate risk-adjusted NPV for a pipeline asset.
        
        Args:
            peak_sales_millions: Expected peak annual sales
            probability_success: Probability of reaching market
            development_cost_millions: Total development costs
            years_to_peak: Years until peak sales
            patent_life_years: Remaining patent life
            
        Returns:
            Risk-adjusted NPV in millions USD
        """
        # Present value of sales (simplified trapezoidal ramp)
        avg_sales = peak_sales_millions * 0.65  # Average over lifecycle
        pv_sales = avg_sales * (
            1 - (1 + self.discount_rate) ** -patent_life_years
        ) / self.discount_rate
        
        # Present value of costs (front-loaded)
        pv_costs = development_cost_millions * (
            1 - (1 + self.discount_rate) ** -(years_to_peak * 0.6)
        ) / self.discount_rate
        
        # Risk adjustment
        risk_adjusted_npv = (pv_sales * probability_success) - pv_costs
        
        return round(risk_adjusted_npv, 1)
    
    def calculate_roi(
        self,
        npv_millions: float,
        development_cost_millions: float
    ) -> float:
        """
        Calculate return on investment ratio.
        
        Args:
            npv_millions: Net present value
            development_cost_millions: Total development costs
            
        Returns:
            ROI as a multiple (e.g., 3.5x)
        """
        if development_cost_millions == 0:
            return 0.0
        return round((npv_millions + development_cost_millions) / development_cost_millions, 1)
    
    def generate_financial_summary(
        self,
        enrichment_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate financial summary from enrichment results.
        
        Args:
            enrichment_results: List of enrichment result dictionaries
            
        Returns:
            DataFrame with financial metrics
        """
        rows = []
        for result in enrichment_results:
            therapeutic_area = result.get("therapeutic_area", "metabolic")
            tam = self.estimate_tam(therapeutic_area)
            
            # Assume probability based on p-value
            p_val = result.get("adjusted_p_value", 1.0)
            if p_val < 0.001:
                prob_success = 0.15
            elif p_val < 0.01:
                prob_success = 0.10
            elif p_val < 0.05:
                prob_success = 0.06
            else:
                prob_success = 0.03
            
            # Estimate development cost based on fold enrichment
            fold_enrichment = result.get("fold_enrichment", 1.0)
            if fold_enrichment > 3.0:
                dev_stage = "preclinical"
            elif fold_enrichment > 2.0:
                dev_stage = "discovery"
            else:
                dev_stage = "discovery"
            
            dev_cost = self.DEV_COST_BY_STAGE.get(dev_stage, 50)
            
            # Calculate NPV and ROI
            npv = self.calculate_npv(
                peak_sales_millions=tam,
                probability_success=prob_success,
                development_cost_millions=dev_cost
            )
            roi = self.calculate_roi(npv, dev_cost)
            
            rows.append({
                "gene": result.get("gene", "Unknown"),
                "pathway": result.get("pathway", "Unknown"),
                "therapeutic_area": therapeutic_area,
                "p_value": p_val,
                "fold_enrichment": fold_enrichment,
                "estimated_tam_millions": tam,
                "probability_success": prob_success,
                "development_cost_millions": dev_cost,
                "npv_millions": npv,
                "roi_multiple": roi,
            })
        
        return pd.DataFrame(rows)


def main() -> None:
    """Demonstrate financial pathway calculator."""
    logging.basicConfig(level=logging.INFO)
    
    calc = FinancialPathwayCalculator(discount_rate=0.12)
    
    # Sample enrichment results
    sample_results = [
        {"gene": "DGAT1", "pathway": "Steroid Biosynthesis", 
         "therapeutic_area": "metabolic", "adjusted_p_value": 0.0001, 
         "fold_enrichment": 4.2},
        {"gene": "YARS2", "pathway": "Oxidative Phosphorylation", 
         "therapeutic_area": "mitochondrial", "adjusted_p_value": 0.0005, 
         "fold_enrichment": 3.8},
        {"gene": "ACSL1", "pathway": "Fatty Acid Biosynthesis", 
         "therapeutic_area": "metabolic", "adjusted_p_value": 0.003, 
         "fold_enrichment": 2.9},
    ]
    
    df = calc.generate_financial_summary(sample_results)
    
    print("\nFinancial Pathway Summary:")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Summary of Improvements

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `PathwayEnrichmentViz.py` | Visual-finance synergy | 4-panel investor dashboard, tier-based coloring, validation scoring |
| `investor_metrics.py` | Pitch data extraction | JSON/CSV/text export, NPV calculations, one-liner generation |
| `enrichment_engine.py` | Financial integration | TAM estimation, risk-adjusted NPV, ROI calculation |

### Key Design Decisions:
1. **FinancialTier enum** - Provides clear investor confidence classification
2. **Composite validation score** - Combines statistical + financial metrics (0-100)
3. **Multi-format export** - JSON for data rooms, CSV for analysis, text for slides
4. **Sample data generators** - Enables immediate testing without external dependencies
5. **Type hints throughout** - Full mypy compatibility for enterprise adoption