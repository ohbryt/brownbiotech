# BrownBioTech Pipeline Expansion
## Cancer Metabolism Target Discovery Platform

---

## Vision

BrownBioTech is building Korea's first AI-powered cancer metabolism target discovery platform that combines:
1. **Multi-omics analysis** (proteomics, transcriptomics, metabolomics)
2. **Cancer metabolism specialization** 
3. **Automated target validation pipeline**
4. **Wet lab verification**

---

## Expanded Pipeline: BROWN-X Series

### BROWN-1 (Lead)
| Field | Value |
|-------|-------|
| **Target** | DGAT1 (Diacylglycerol O-Acyltransferase 1) |
| **Indication** | NSCLC |
| **Mechanism** | Lipid metabolism inhibition |
| **Stage** | Preclinical — In vivo efficacy CONFIRMED |
| **Milestone** | IND filing Q3 2026 |

### BROWN-2 (Second)
| Field | Value |
|-------|-------|
| **Target** | YARS2 (Mitochondrial Tyrosyl-tRNA Synthetase) |
| **Indication** | NSCLC |
| **Mechanism** | Mitochondrial protein synthesis inhibition |
| **Stage** | Discovery — In vitro validated |
| **Milestone** | In vivo studies H2 2026 |

### BROWN-3 (Discovery)
| Field | Value |
|-------|-------|
| **Target** | TBD via TCGA/DepMap analysis |
| **Indication** | Lung cancer + metabolic dependencies |
| **Mechanism** | To be determined |
| **Stage** | Target identification |
| **Milestone** | Target nomination Q4 2026 |

---

## Service Platform

### 1. Cancer Metabolism Target Discovery

**Input:** Customer-specified disease + omics data
**Process:**
1. Multi-omics integration (TCGA, DepMap, CCLE, RPPA500)
2. Metabolism pathway analysis
3. CRISPR dependency screening
4. AI-powered target ranking

**Output:** Validated target candidates with:
- Expression levels (tumor vs normal)
- Survival correlation
- CRISPR dependency scores
- Drugability assessment

### 2. Multi-Omics Analysis Service

**Data Sources:**
- TCGA (The Cancer Genome Atlas) — 32 cancer types
- CCLE (Cancer Cell Line Encyclopedia) — 900+ cell lines
- RPPA500 (Reverse Phase Protein Array) — 447 proteins
- DepMap (CRISPR, drug sensitivity)

**Analysis Types:**
- Differential expression analysis
- Pathway enrichment (Hallmark, KEGG, Reactome)
- Survival analysis (KM curves, Cox regression)
- Correlation analysis (protein-protein, protein-mutation)
- Drug sensitivity association

### 3. Wet Validation Service

**GIST Wet Lab Capabilities:**
- Cell culture (cancer cell lines, primary cells)
- siRNA/shRNA knockdown
- Compound profiling
- Viability assays (MTT, WST-1)
- Apoptosis assays (Caspase, Annexin V)
- Western blot validation
- Mouse xenograft models

**Automated Workflow:**
1. AI designs experiment
2. Lab executes
3. Results fed back to AI
4. Iterative optimization

---

## Technology Stack

### ARP: Autonomous Research Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARP PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐ │
│  │ Research  │ →  │ Design    │ →  │ Predict   │ →  │ Validate │ │
│  │ Layer     │    │ Layer     │    │ Layer     │    │ Layer    │ │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘ │
│       ↑                                                        │    │
│       └────────────── Feedback Loop ←──────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Agent Architecture

| Agent | Role | Models |
|-------|------|--------|
| **Research Agent** | Literature, target mining | Nemotron (free) |
| **Analysis Agent** | Multi-omics, statistics | Gemini Flash Lite |
| **Design Agent** | Molecular generation | GLM-5 |
| **Review Agent** | Quality control | Stepfun (free) |
| **Supervisor** | Orchestration, routing | Gemini Flash Lite |

### Data Integration

```
TCGA ──────┐
            │
CCLE ──────┼──→ Unified HDF5 ──→ DrBioRight-style
            │      Database          Chat Interface
RPPA500 ───┤
            │
DepMap ────┘
```

---

## Target Discovery Workflow

### Step 1: Disease Selection
- Customer specifies cancer type (e.g., NSCLC, breast cancer)
- System retrieves all relevant multi-omics data

### Step 2: Metabolism Focus
- Filter for metabolic pathways:
  - Glycolysis/TCA cycle
  - Lipid metabolism (FA synthesis, β-oxidation)
  - Glutamine metabolism
  - Serine/glycine one-carbon metabolism
  - Mitochondrial function

### Step 3: Multi-Omics Integration
- RNA-seq: Gene expression
- RPPA: Protein expression + PTMs
- Mutation: SNVs, copy number
- Methylation: Epigenetic regulation

### Step 4: AI-Powered Ranking
Criteria:
- Differential expression (tumor vs normal)
- Survival correlation (high = bad prognosis)
- CRISPR dependency (essential in cancer, not normal)
- Drugability (small molecule / antibody / PROTAC)
- Novelty (unexplored in disease)

### Step 5: Wet Lab Validation
- siRNA knockdown → viability
- Compound testing (IC50)
- Mechanistic studies
- In vivo mouse models

### Step 6: IP & Partnership
- Novel targets → patent filing
- Partnership with pharma
- Milestone payments + royalties

---

## Infographic: Automated Pipeline

See: `infographic.html` (separate file for website)

---

## Pricing Model

### Target Discovery Service
| Tier | Description | Price |
|------|-------------|-------|
| Bronze | Single cancer type, standard analysis | ₩500만 |
| Silver | Multi-omics, pathway analysis | ₩1,000万 |
| Gold | Full platform + 3 targets validated | ₩2,500万 |

### Ongoing Partnership
- Annual subscription: ₩5,000万/year
- Includes: Quarterly target updates, priority validation
- Equity: Negotiable based on value

---

## Competitive Advantages

| Factor | BrownBioTech | Competitors |
|--------|-------------|-------------|
| **Focus** | Cancer metabolism | General oncology |
| **Speed** | 2-4 weeks per target | 3-6 months |
| **Cost** | 30-50% lower | Traditional CROs |
| **AI Integration** | Full loop automation | Partial |
| **Korean Market** | Local + global | Mostly US/EU |

---

## Roadmap

| Quarter | Milestone |
|---------|-----------|
| 2026 Q2 | BROWN-1 IND enabling, BROWN-3 target identification |
| 2026 Q3 | BROWN-1 IND filing, Service platform launch |
| 2026 Q4 | First service customers, BROWN-3 in vivo |
| 2027 Q1 | Phase 1 trial start (BROWN-1), 3 service clients |
| 2027 Q4 | BROWN-2 IND filing, Platform expansion |

---

## Contact

Dr. Chang-Myung Oh
BrownBioTech
contact@brownbiotech.kr

---
*Last updated: March 2026*
