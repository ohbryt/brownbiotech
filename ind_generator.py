#!/usr/bin/env python3
"""
IND Filing Document Generator for BrownBioTech
================================================
Generates Investigational New Drug (IND) application sections for BROWN-1 (DGAT1 inhibitor)

Target: Q3 2026 IND Submission
Compound: BROWN-1 (DGAT1 inhibitor)
Indication: NSCLC (Non-Small Cell Lung Cancer)
Founder: Dr. Chang-Myung Oh (GIST)

Sections:
    1. Introduction
    2. Target Validation
    3. Preclinical Pharmacology
    4. Toxicology
    5. Manufacturing
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# =============================================================================
# Document Structure - Dataclasses
# =============================================================================

@dataclass
class Section:
    """Base class for an IND document section."""
    number: str
    title: str
    content: str = ""

    def render(self) -> str:
        return f"## {self.number} {self.title}\n\n{self.content}"


@dataclass
class INDMetadata:
    """Metadata for the IND application."""
    compound_name: str = "BROWN-1"
    generic_name: str = "[GENERIC_NAME_PENDING]"
    indication: str = "Non-Small Cell Lung Cancer (NSCLC)"
    target: str = "DGAT1 (Diacylglycerol O-Acyltransferase 1)"
    target_pathway: str = "Lipid Metabolism / Triglyceride Synthesis"
    sponsor: str = "BrownBioTech"
    founder: str = "Dr. Chang-Myung Oh"
    institution: str = "Gwangju Institute of Science and Technology (GIST)"
    ind_target: str = "Q3 2026"
    date: str = field(default_factory=lambda: datetime.now().strftime("%B %d, %Y"))
    version: str = "1.0"
    status: str = "Draft"


@dataclass
class IntroductionSection(Section):
    """Section 1: Introduction to the compound and IND purpose."""
    def __init__(self, meta: INDMetadata):
        self.number = "1"
        self.title = "Introduction"
        self.content = self._build_content(meta)

    def _build_content(self, meta: INDMetadata) -> str:
        return f"""### 1.1 Purpose of Submission

This Investigational New Drug (IND) application is submitted by BrownBioTech to seek authorization from the relevant regulatory authorities to initiate clinical evaluation of {meta.compound_name} ({meta.generic_name}), a novel DGAT1 inhibitor, for the treatment of {meta.indication}.

### 1.2 Compound Overview

{meta.compound_name} is a small molecule inhibitor targeting Diacylglycerol O-Acyltransferase 1 (DGAT1), a key enzyme in the triglyceride synthesis pathway. The compound has demonstrated potent anti-tumor activity in preclinical models of non-small cell lung cancer (NSCLC).

**Key Attributes:**
- **Drug Candidate:** {meta.generic_name}
- **Mechanism of Action:** DGAT1 inhibition
- **Therapeutic Area:** Oncology
- **Indication:** {meta.indication}
- **Route of Administration:** [TO BE DETERMINED]
- **Dosage Form:** [TO BE DETERMINED]

### 1.3 Background and Rationale

DGAT1 is a critical enzyme catalyzing the final step of triglyceride synthesis by converting diacylglycerol (DAG) and fatty acyl-CoA into triglycerides (TAG). Emerging evidence demonstrates that DGAT1 plays a significant role in lipid metabolism within tumors, where enhanced de novo lipogenesis supports rapid tumor growth and survival.

Inhibition of DGAT1 in preclinical NSCLC models has demonstrated:
- Reduced tumor cell proliferation
- Induction of apoptosis
- Suppression of tumor growth in vivo (mouse xenograft/PDX models confirmed)

### 1.4 Regulatory History

- **Pre-IND Consultation:** [DATE_PENDING]
- **IND Target Submission:** {meta.ind_target}
- **Clinical Trial Phase:** Phase I/II planned

### 1.5 Compliance Statement

This application has been prepared in accordance with applicable regulatory guidelines and contains all information required under 21 CFR Part 312 (United States) or equivalent international standards.
"""


@dataclass
class TargetValidationSection(Section):
    """Section 2: Target Validation - DGAT1 role in NSCLC."""
    def __init__(self, meta: INDMetadata):
        self.number = "2"
        self.title = "Target Validation"
        self.content = self._build_content(meta)

    def _build_content(self, meta: INDMetadata) -> str:
        return f"""### 2.1 Target Identification

#### 2.1.1 DGAT1 Biology

Diacylglycerol O-Acyltransferase 1 (DGAT1) is a membrane-bound enzyme belonging to the acyltransferase family. It catalyzes the esterification of fatty acids into the sn-3 position of diacylglycerol, representing the committed step in triglyceride synthesis.

**Gene Information:**
- **Gene Symbol:** DGAT1
- **Gene ID:** [TO BE POPULATED]
- **Chromosomal Location:** [TO BE POPULATED]
- **Protein Family:** Diacylglycerol acyltransferases (DGAT)

#### 2.1.2 DGAT1 Expression in NSCLC

DGAT1 expression has been documented in various tumor types, including lung cancer. Analysis of publicly available datasets (TCGA, GEO) indicates [ELEVATED_REDUCED] DGAT1 expression in NSCLC tumor samples compared to normal lung tissue.

**mRNA Expression Data:**
- Tumor samples: [TO BE POPULATED - mean ± SD]
- Normal adjacent tissue: [TO BE POPULATED - mean ± SD]
- p-value: [TO BE POPULATED]

#### 2.1.3 DGAT1 Isoforms

Two major DGAT isoforms have been identified:
- **DGAT1:** Primarily localized to the endoplasmic reticulum; high expression in intestine, liver, and adipose tissue
- **DGAT2:** Localized to the ER; essential for baseline TAG synthesis

BROWN-1 is designed to selectively inhibit DGAT1 with minimal activity against DGAT2.

---

### 2.2 Target Validation Studies

#### 2.2.1 In Vitro Validation

**Cell Line Studies:**

| Cell Line | Tissue | DGAT1 Expression | IC50 (nM) | Reference |
|-----------|--------|------------------|-----------|-----------|
| [CELL_LINE_1] | NSCLC | [STATUS] | [VALUE] | [REF] |
| [CELL_LINE_2] | NSCLC | [STATUS] | [VALUE] | [REF] |
| [CELL_LINE_3] | Normal Lung | [STATUS] | [VALUE] | [REF] |

**Key Findings:**
- DGAT1 knockdown via siRNA reduces cell viability in NSCLC cell lines by [X]%
- DGAT1 knockout (CRISPR) induces apoptosis in [CELL_LINE]
- Lipidomic analysis shows decreased TAG levels upon DGAT1 inhibition

#### 2.2.2 In Vivo Validation

**Xenograft/PDX Models:**

In vivo efficacy of BROWN-1 has been confirmed in mouse models:

| Model | Treatment | Dose | Tumor Growth Inhibition | p-value |
|-------|-----------|------|------------------------|---------|
| [MODEL_1] | BROWN-1 | [DOSE] | [TGI%] | [p-value] |
| [MODEL_2] | BROWN-1 | [DOSE] | [TGI%] | [p-value] |

**Study Details:**
- **Species:** Mus musculus (immunodeficient mice)
- **Model Type:** [Xenograft / PDX]
- **Tumor Model:** [CELL_LINE or PATIENT TUMOR]
- **Administration:** [ROUTE, e.g., oral gavage]
- **Treatment Duration:** [DURATION]
- **Efficacy Endpoint:** Tumor volume measurement

#### 2.2.3 Mechanistic Validation

**Lipid Metabolism Effects:**
- Decreased cellular triglyceride levels: [X]%
- Accumulation of diacylglycerol (DAG): [X]%
- Altered fatty acid flux: [DESCRIPTION]

**Downstream Pathway Effects:**
- [PATHWAY_1] modulation: [EFFECT]
- [PATHWAY_2] modulation: [EFFECT]
- ER stress induction: [YES/NO - OBSERVED/NOT OBSERVED]

---

### 2.3 Translational Rationale

#### 2.3.1 Clinical Precedent

[TO BE POPULATED: Cite any DGAT1 inhibitors that have reached clinical development, e.g., DGAT1 inhibitors for metabolic diseases.]

#### 2.3.2 Patient Selection Biomarkers

Potential biomarkers for patient selection include:
- DGAT1 expression levels (IHC, mRNA)
- Lipid metabolism signatures
- [ADDITIONAL BIOMARKERS]

#### 2.3.3 Unmet Medical Need

- NSCLC remains a leading cause of cancer-related mortality worldwide
- Limited efficacy of current standard-of-care in [SUBGROUP]
- Novel targeted approaches are needed to address treatment-resistant disease

---

### 2.4 Safety Considerations Related to Target

DGAT1 global knockout mice are viable but exhibit:
- Skin barrier defects
- Lactation impairment in females
- [ADDITIONAL FINDINGS]

These findings suggest that systemic DGAT1 inhibition may be associated with certain toxicities that require monitoring in clinical trials.

### 2.5 Summary and Conclusions

DGAT1 is a biologically validated target for NSCLC therapy based on:
1. Preclinical evidence of DGAT1 dependence in NSCLC models
2. In vivo anti-tumor efficacy of BROWN-1 in mouse models
3. Mechanistic link between DGAT1 activity and tumor lipid metabolism

[ADDITIONAL CONCLUSIONS]
"""


@dataclass
class PreclinicalPharmacologySection(Section):
    """Section 3: Preclinical Pharmacology."""
    def __init__(self, meta: INDMetadata):
        self.number = "3"
        self.title = "Preclinical Pharmacology"
        self.content = self._build_content(meta)

    def _build_content(self, meta: INDMetadata) -> str:
        return f"""### 3.1 Compound Information

#### 3.1.1 Chemical Structure

**Chemical Name:** [FULL CHEMICAL NAME]
**Molecular Formula:** [FORMULA]
**Molecular Weight:** [MW] g/mol
**CAS Number:** [CAS]

**Structure Description:**
[DESCRIPTION OF CHEMICAL STRUCTURE / ATTACH STRUCTURE IMAGE]

#### 3.1.2 Physicochemical Properties

| Property | Value |
|----------|-------|
| Appearance | [DESCRIPTION] |
| Solubility | [SOLUBILITY DATA] |
| Melting Point | [VALUE] |
| LogP | [VALUE] |
| pKa | [VALUE] |
| Chemical Stability | [DATA] |

---

### 3.2 In Vitro Pharmacology

#### 3.2.1 Enzyme Activity Assay

**Assay Description:**
Recombinant human DGAT1 enzyme was used to assess inhibitory activity of BROWN-1.

**Results:**

| Parameter | Value |
|-----------|-------|
| IC50 (human DGAT1) | [VALUE] nM |
| IC50 (mouse DGAT1) | [VALUE] nM |
| Hill Coefficient | [VALUE] |
| Assay Conditions | [BUFFER, SUBSTRATE CONCENTRATIONS] |
| Positive Control | [REFERENCE COMPOUND] |

**Selectivity Panel:**
| Enzyme/Target | IC50 (nM) | Selectivity (fold) |
|---------------|-----------|-------------------|
| DGAT1 (human) | [VALUE] | - |
| DGAT2 | [VALUE] | [X]-fold |
| [OFF-TARGET_1] | [VALUE] | [X]-fold |
| [OFF-TARGET_2] | [VALUE] | [X]-fold |

#### 3.2.2 Cell-Based Activity

**Cell Viability Assays:**

| Cell Line | Tissue | EC50 (nM) | Emax (%) | Assay Duration |
|-----------|--------|-----------|----------|----------------|
| [CELL_LINE_1] | NSCLC | [VALUE] | [VALUE] | [TIME] |
| [CELL_LINE_2] | NSCLC | [VALUE] | [VALUE] | [TIME] |
| [CELL_LINE_3] | Normal | [VALUE] | [VALUE] | [TIME] |

**Mechanism of Action Studies:**
- Apoptosis induction: [DATA]
- Cell cycle analysis: [DATA]
- Lipid droplet quantification: [DATA]

#### 3.2.3 Metabolite Profiling

[TO BE POPULATED - Metabolite identification studies]

---

### 3.3 In Vivo Pharmacology

#### 3.3.1 Pharmacokinetics (PK) - Single Dose

**Species:** [Mouse / Rat / Dog / NHP]

| Parameter | Value (Mean ± SD) |
|-----------|-------------------|
| Route | [ROUTE] |
| Dose | [DOSE] |
| Cmax (ng/mL) | [VALUE] |
| Tmax (h) | [VALUE] |
| AUC (ng·h/mL) | [VALUE] |
| t1/2 (h) | [VALUE] |
| F (%) | [VALUE] |
| Vd (L/kg) | [VALUE] |
| CL (mL/h/kg) | [VALUE] |

#### 3.3.2 Pharmacodynamics (PD)

**Target Engagement:**
- [METHOD: e.g., biomarker analysis, imaging]
- [RESULTS]

**Anti-Tumor Efficacy:**

| Study ID | Model | Route | Dose (mg/kg) | Schedule | Tumor Growth Inhibition | p-value |
|----------|-------|-------|--------------|----------|------------------------|---------|
| [STUDY_001] | [MODEL] | [ROUTE] | [DOSE] | [SCHEDULE] | [TGI%] | [p] |
| [STUDY_002] | [MODEL] | [ROUTE] | [DOSE] | [SCHEDULE] | [TGI%] | [p] |

**Xenograft Study Details:**

Study [STUDY_ID]:
- **Objective:** Evaluate anti-tumor efficacy of BROWN-1 in [MODEL] xenograft model
- **Animals:** [SPECIES, STRAIN, SEX, N=]
- **Tumor Induction:** [METHOD]
- **Treatment Groups:**
  - Vehicle control (N=[X])
  - BROWN-1 [DOSE_1] (N=[X])
  - BROWN-1 [DOSE_2] (N=[X])
  - [POSITIVE CONTROL] (N=[X])
- **Results:** [SUMMARY]

---

### 3.4 Safety Pharmacology

#### 3.4.1 Cardiovascular

**hERG Assay:**
- IC50: [VALUE] μM
- Interpretation: [LOW / MODERATE / HIGH risk]

**In Vivo Cardiovascular (Telemetry):**
- Species: [ANIMAL MODEL]
- Findings: [DATA]

#### 3.4.2 Central Nervous System

** Irwin Test / Functional Observation Battery:**
- Species: [ANIMAL MODEL]
- Findings: [DATA]

#### 3.4.3 Respiratory

- Species: [ANIMAL MODEL]
- Findings: [DATA]

---

### 3.5 Summary

BROWN-1 demonstrates:
- [X] nM potency against DGAT1
- [X]-fold selectivity over DGAT2 and off-targets
- In vivo anti-tumor efficacy in NSCLC models
- Acceptable PK profile supporting [ROUTE] administration

[ADDITIONAL CONCLUSIONS AND INTERPRETATION]
"""


@dataclass
class ToxicologySection(Section):
    """Section 4: Toxicology."""
    def __init__(self, meta: INDMetadata):
        self.number = "4"
        self.title = "Toxicology"
        self.content = self._build_content(meta)

    def _build_content(self, meta: INDMetadata) -> str:
        return f"""### 4.1 Toxicology Overview

The nonclinical toxicology program for {meta.compound_name} ({meta.generic_name}) has been designed to support clinical development for the treatment of {meta.indication}. Studies were conducted in accordance with GLP regulations where applicable.

---

### 4.2 Single-Dose Toxicity

#### 4.2.1 Single Ascending Dose (SAD) Study

**Species:** [Mouse / Rat]

| Parameter | Findings |
|-----------|----------|
| Route | [ROUTE] |
| Dose Range | [DOSE RANGE] |
| MTD | [VALUE] |
| NOAEL | [VALUE] |
| Clinical Signs | [DESCRIPTION] |
| Mortality | [FINDINGS] |

**Conclusion:** [SUMMARY]

---

### 4.3 Repeat-Dose Toxicity

#### 4.3.1 [STUDY NAME / SPECIES]

**Study Design:**
- **Species/Strain:** [SPECIES, STRAIN]
- **Sex:** [MALES / FEMALES / BOTH]
- **Group Size:** [N PER GROUP]
- **Duration:** [X] weeks
- **Route:** [ROUTE]
- **Doses:** [DOSE LEVELS]

**Results Summary:**

| Dose (mg/kg) | Mortality | Clinical Signs | Food Consumption | Body Weight |
|--------------|-----------|----------------|------------------|-------------|
| [VEHICLE] | [FINDINGS] | [FINDINGS] | [FINDINGS] | [FINDINGS] |
| [LOW] | [FINDINGS] | [FINDINGS] | [FINDINGS] | [FINDINGS] |
| [MID] | [FINDINGS] | [FINDINGS] | [FINDINGS] | [FINDINGS] |
| [HIGH] | [FINDINGS] | [FINDINGS] | [FINDINGS] | [FINDINGS] |

**Toxicokinetics:**
| Dose (mg/kg) | Day 1 AUC | Day X AUC | Accumulation Ratio |
|--------------|-----------|-----------|-------------------|
| [LOW] | [VALUE] | [VALUE] | [RATIO] |
| [MID] | [VALUE] | [VALUE] | [RATIO] |
| [HIGH] | [VALUE] | [VALUE] | [RATIO] |

**Target Organs of Toxicity:**
- [ORGAN_1]: [FINDINGS]
- [ORGAN_2]: [FINDINGS]

**NOAEL:** [VALUE] mg/kg (corresponding to [EXPOSURE] AUC)

---

### 4.4 Genetic Toxicology

#### 4.4.1 Bacterial Reverse Mutation (Ames)

**Result:** [POSITIVE / NEGATIVE / INCONCLUSIVE]
**Conclusion:** [INTERPRETATION]

#### 4.4.2 In Vitro Chromosomal Aberration

**Result:** [POSITIVE / NEGATIVE / INCONCLUSIVE]
**Conclusion:** [INTERPRETATION]

#### 4.4.3 In Vivo Micronucleus (Bone Marrow)

**Species:** [SPECIES]
**Result:** [POSITIVE / NEGATIVE / INCONCLUSIVE]
**Conclusion:** [INTERPRETATION]

---

### 4.5 Carcinogenicity

[Carcinogenicity studies are not required for IND submission. Typically conducted post-Phase 2 for chronic indications. TO BE DETERMINED based on clinical indication and duration.]

---

### 4.6 Reproductive and Developmental Toxicology

#### 4.6.1 Fertility and Early Embryonic Development

**Species:** [RAT / MOUSE]
**Result:** [FINDINGS]
**NOAEL (Males):** [VALUE]
**NOAEL (Females):** [VALUE]

#### 4.6.2 Embryo-Fetal Developmental Toxicity (EFD)

**Species:** [RAT / RABBIT]
**Result:** [FINDINGS]
**Maternal NOAEL:** [VALUE]
**Developmental NOAEL:** [VALUE]

**Conclusion:** [SUMMARY]

---

### 4.7 Immunotoxicology

[STUDY DATA OR "NOT CONDUCTED - TO BE INCLUDED PRIOR TO PHASE 1"]

---

### 4.8 Safety Margins

| Study | NOAEL (mg/kg/day) | Human Equivalent Dose (mg/kg/day) | Safety Margin (fold) |
|-------|-------------------|----------------------------------|---------------------|
| [STUDY_1] | [VALUE] | [HED_VALUE] | [MARGIN] |
| [STUDY_2] | [VALUE] | [HED_VALUE] | [MARGIN] |

**Clinical Starting Dose:** [TO BE DETERMINED BASED ON ALLOMETRIC SCALING AND MABEL]

---

### 4.9 Summary and Conclusions

The nonclinical toxicology profile of {meta.compound_name} supports advancement to clinical testing:

1. **Safety Margins:** [X]-fold margin at NOAEL relative to anticipated clinical exposure
2. **Target Organs:** [LIST]
3. **Genotoxic Potential:** [ASSESSMENT]
4. **Reproductive Risk:** [ASSESSMENT]

[ADDITIONAL CONCLUSIONS AND CLINICAL IMPLICATIONS]
"""


@dataclass
class ManufacturingSection(Section):
    """Section 5: Manufacturing and Controls."""
    def __init__(self, meta: INDMetadata):
        self.number = "5"
        self.title = "Manufacturing and Controls"
        self.content = self._build_content(meta)

    def _build_content(self, meta: INDMetadata) -> str:
        return f"""### 5.1 Drug Substance

#### 5.1.1 Description

**Chemical Name:** [FULL IUPAC NAME]
**INN:** [INTERNATIONAL NONPROPRIETARY NAME]
**Code Name:** {meta.compound_name}
**Molecular Formula:** [FORMULA]
**Molecular Weight:** [MW] g/mol
**CAS Number:** [CAS]

**Structure:**

```
[MOLECULAR STRUCTURE - TO BE ATTACHED]
```

#### 5.1.2 Physical and Chemical Properties

| Property | Specification | Test Method |
|----------|---------------|-------------|
| Appearance | [DESCRIPTION] | Visual |
| Solubility | [DATA] | [METHOD] |
| Melting Point | [RANGE] °C | [METHOD] |
| Optical Rotation | [VALUE] | [METHOD] |
| LogP | [VALUE] | [METHOD] |
| pKa | [VALUE] | [METHOD] |
| Hygroscopicity | [YES/NO] | [METHOD] |
| Polymorphism | [STATUS] | [METHOD] |

#### 5.1.3 Manufacturing Process

**Manufacturer:** [NAME AND ADDRESS]

**Synthetic Route Summary:**

**Step 1:** [DESCRIPTION]
```
[REACTION SCHEME]
```

**Step 2:** [DESCRIPTION]

**Step 3:** [DESCRIPTION]

**Critical Process Parameters:**
- [PARAMETER 1]: [CONTROL RANGE]
- [PARAMETER 2]: [CONTROL RANGE]

**Critical Quality Attributes:**
- [ATTRIBUTE 1]
- [ATTRIBUTE 2]

#### 5.1.4 Specifications

**Drug Substance Release Specifications:**

| Test | Acceptance Criteria | Method |
|------|---------------------|--------|
| Appearance | [WHITE TO OFF-WHITE POWDER] | Visual |
| Identity (IR) | Conforms to reference spectrum | IR |
| Identity (HPLC) | Retention time matches reference | HPLC |
| Assay (Anhydrous) | [95.0 - 105.0] % | HPLC |
| Related Substances | | HPLC |
| - Unknown impurities | NMT [0.10] % | |
| - Total impurities | NMT [0.50] % | |
| Residual Solvents | | GC |
| - [SOLVENT_1] | NMT [X] ppm | |
| - [SOLVENT_2] | NMT [X] ppm | |
| Water Content | NMT [X] % | Karl Fischer |
| Heavy Metals | NMT [X] ppm | [METHOD] |
| Particle Size | [SPECIFICATION] | [METHOD] |

#### 5.1.5 Stability

**Stability Summary:**

| Condition | Duration | Results |
|-----------|----------|---------|
| [30°C/65%RH] | [X] months | [DATA] |
| [40°C/75%RH] | [X] months | [DATA] |
| [5°C] | [X] months | [DATA] |

**Proposed Storage Condition:** [TO BE FINALIZED]
**Re-test Period:** [TO BE DETERMINED]

---

### 5.2 Drug Product

#### 5.2.1 Description

**Dosage Form:** [TABLET / CAPSULE / INJECTION / etc.]
**Strength:** [X] mg per [UNIT]
**Route of Administration:** [ORAL / IV / etc.]

#### 5.2.2 Composition

**Formulation:**

| Ingredient | Function | Quantity per [UNIT] |
|------------|----------|--------------------|
| {meta.generic_name} | Active | [X] mg |
| [EXCIPIENT_1] | [FUNCTION] | [X] mg |
| [EXCIPIENT_2] | [FUNCTION] | [X] mg |
| [EXCIPIENT_N] | [FUNCTION] | [X] mg |

**Proposed Formulations:** [TO BE DETERMINED]

#### 5.2.3 Manufacturing Process

[PROCESS DESCRIPTION - TO BE PROVIDED BY CMO]

#### 5.2.4 Specifications

**Drug Product Release Specifications:**

| Test | Acceptance Criteria | Method |
|------|---------------------|--------|
| Appearance | [DESCRIPTION] | Visual |
| Identity | Conforms | [METHOD] |
| Assay | [X] % of label claim | HPLC |
| Related Substances | As per ICH Q3B | HPLC |
| Dissolution | NLT [X] % at [Y] min | USP Apparatus [X] |
| Water Content | NMT [X] % | Karl Fischer |
| [ADDITIONAL TESTS] | [CRITERIA] | [METHOD] |

#### 5.2.5 Stability

**Stability Summary:**

| Condition | Duration | Results |
|-----------|----------|---------|
| [30°C/65%RH] | [X] months | [DATA] |
| [40°C/75%RH] | [X] months | [DATA] |

**Proposed Shelf Life:** [TO BE DETERMINED]
**Storage Condition:** [TO BE SPECIFIED]

---

### 5.3 Current Batch Information

| Batch | Size | Date | Process | Status |
|-------|------|------|---------|--------|
| [BATCH_001] | [SIZE] | [DATE] | [PROCESS] | [RELEASED / UNDER TESTING] |
| [BATCH_002] | [SIZE] | [DATE] | [PROCESS] | [RELEASED / UNDER TESTING] |
| [BATCH_003] | [SIZE] | [DATE] | [PROCESS] | [RELEASED / UNDER TESTING] |

---

### 5.4 Container Closure System

**Primary Packaging:** [DESCRIPTION]
**Secondary Packaging:** [DESCRIPTION]
**Closure System:** [DESCRIPTION]

---

### 5.5 Manufacturing Summary and Conclusions

{meta.compound_name} ({meta.generic_name}) has been manufactured according to appropriate quality standards. The drug substance and drug product specifications are designed to ensure identity, purity, potency, and stability.

[ADDITIONAL CONCLUSIONS AND CMC STRATEGY]
"""


@dataclass
class INDDocument:
    """Complete IND application document."""
    metadata: INDMetadata
    sections: list = field(default_factory=list)

    def __init__(self, metadata: Optional[INDMetadata] = None):
        self.metadata = metadata or INDMetadata()
        self.sections = [
            IntroductionSection(self.metadata),
            TargetValidationSection(self.metadata),
            PreclinicalPharmacologySection(self.metadata),
            ToxicologySection(self.metadata),
            ManufacturingSection(self.metadata),
        ]

    def render(self) -> str:
        """Render the complete IND document as markdown."""
        lines = [
            f"# INVESTIGATIONAL NEW DRUG (IND) APPLICATION",
            f"",
            f"## {self.metadata.compound_name} ({self.metadata.generic_name})",
            f"",
            f"**Target:** {self.metadata.target}",
            f"**Indication:** {self.metadata.indication}",
            f"**Sponsor:** {self.metadata.sponsor}",
            f"**Founder:** {self.metadata.founder}, {self.metadata.institution}",
            f"**IND Target Submission:** {self.metadata.ind_target}",
            f"**Document Version:** {self.metadata.version}",
            f"**Date:** {self.metadata.date}",
            f"**Status:** {self.metadata.status}",
            f"",
            f"---",
            f"",
        ]
        for section in self.sections:
            lines.append(section.render())
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# Generator Methods
# =============================================================================

class INDGenerator:
    """Generator for IND application documents."""

    def __init__(self, metadata: Optional[INDMetadata] = None):
        self.metadata = metadata or INDMetadata()
        self.document = INDDocument(self.metadata)

    def generate_section(self, section_name: str) -> str:
        """
        Generate a specific IND section by name.

        Args:
            section_name: One of 'introduction', 'target_validation',
                         'pharmacology', 'toxicology', 'manufacturing'

        Returns:
            Markdown-formatted section content
        """
        section_map = {
            "introduction": IntroductionSection,
            "target_validation": TargetValidationSection,
            "pharmacology": PreclinicalPharmacologySection,
            "toxicology": ToxicologySection,
            "manufacturing": ManufacturingSection,
        }

        section_name_lower = section_name.lower().replace("-", "_").replace(" ", "_")
        if section_name_lower not in section_map:
            available = ", ".join(section_map.keys())
            raise ValueError(f"Unknown section: '{section_name}'. Available: {available}")

        section = section_map[section_name_lower](self.metadata)
        return section.render()

    def generate_full_ind(self) -> str:
        """
        Generate the complete IND application document.

        Returns:
            Full IND document as markdown string
        """
        return self.document.render()

    def export_markdown(self, filepath: str) -> None:
        """
        Export the complete IND document to a markdown file.

        Args:
            filepath: Output file path
        """
        content = self.generate_full_ind()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


# =============================================================================
# CLI / Main
# =============================================================================

def main():
    """Generate a sample IND document for BrownBioTech."""
    meta = INDMetadata(
        compound_name="BROWN-1",
        generic_name="[GENERIC_NAME_PENDING]",
        indication="Non-Small Cell Lung Cancer (NSCLC)",
        target="DGAT1 (Diacylglycerol O-Acyltransferase 1)",
        target_pathway="Lipid Metabolism / Triglyceride Synthesis",
        sponsor="BrownBioTech",
        founder="Dr. Chang-Myung Oh",
        institution="Gwangju Institute of Science and Technology (GIST)",
        ind_target="Q3 2026",
        version="1.0",
        status="Draft",
    )

    generator = INDGenerator(meta)

    # Generate full IND
    full_ind = generator.generate_full_ind()
    print(full_ind)

    # Export to file
    output_path = "/Users/ocm/.openclaw/workspace/brownbiotech/IND_BROWN-1_Draft.md"
    generator.export_markdown(output_path)
    print(f"\n\n[Exported to: {output_path}]")


if __name__ == "__main__":
    main()
