# CausalPol

**Extracting and Validating Causal Claims in Policy Documents Using Large Language Models**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)]()

CausalPol is a modular Python package for extracting, classifying, validating, and detecting inconsistencies in causal claims embedded in multilingual policy documents—legislation, regulatory impact assessments, central bank communications, and international agreement texts.

---

## Overview

Policy documents are dense with causal claims that embed and justify consequential decisions. CausalPol addresses three tasks in a sequential pipeline:

| Stage | Task | Method | Paper Performance |
|-------|------|--------|-------------------|
| 1 | **Causal span extraction** | Fine-tuned mDeBERTa-v3 with nominalization detection + hedge-scope parsing | F1 = 0.79 overall; up to 0.83 (central bank comms) |
| 2 | **Claim validation** | GPT-4o plausibility assessment with uncertainty-aware triage | κ = 0.68 vs. expert panel |
| 3 | **Cross-claim consistency detection** | Semantic similarity + LLM resolution | 6.3% inconsistency rate in EU regulatory docs |

The package also provides `PolCausal-50K`-compatible evaluation metrics (span F1, Cohen's κ, classification reports) for benchmarking.

---

## Installation

### Minimal installation (extraction only, no GPU required)

```bash
pip install causalpol
```

### Full installation (all features)

```bash
pip install "causalpol[full]"
```

This adds `sentence-transformers` (for semantic similarity), `spaCy` (sentence segmentation), and `langdetect` (automatic language detection).

### Development installation

```bash
git clone https://github.com/causalpol/causalpol.git
cd causalpol
pip install -e ".[full,dev]"
```

---

## Quick Start

### Full pipeline (requires OpenAI API key)

```python
from causalpol import CausalPolPipeline
from causalpol.taxonomy.schema import PolicyDomain, Language

pipeline = CausalPolPipeline(openai_api_key="sk-...")

text = """
The elevated level of inflation reflects a combination of supply-side disruptions
and demand-side pressures amplified by energy price pass-through effects.
Higher interest rates reduce inflation through demand contraction, as households
and firms decrease borrowing and spending. However, supply-side price pressures
may cause inflation to persist even as monetary policy tightens.
"""

result = pipeline.run(
    text=text,
    domain=PolicyDomain.MONETARY_POLICY,
    language=Language.ENGLISH,
)

print(result.summary())
```

**Output:**
```
=== CausalPol Pipeline Result ===
Document ID : a3f2b1...
Domain      : monetary_policy
Language    : en
Claims      : 3 extracted, 1 flagged
Inconsist.  : 1 genuine / 2 candidate pairs

[1] [MECHANISTIC / established] CAUSE: 'Higher interest rates' → EFFECT: 'lower inflation' [via: demand contraction] | plausibility=4/5
[2] [MECHANISTIC / speculative] CAUSE: 'supply-side price pressures' → EFFECT: 'inflation to persist' | plausibility=3/5
[3] [CORRELATIONAL / established] CAUSE: 'supply-side disruptions' → EFFECT: 'elevated inflation' | plausibility=4/5

--- Inconsistencies ---
[DIRECTIONAL inconsistency — GENUINE]
  Claim A: [MECHANISTIC / established] CAUSE: 'Higher interest rates' → ...
  Claim B: [MECHANISTIC / speculative] CAUSE: 'supply-side price pressures' → ...
```

### Extraction only (no API key required)

```python
from causalpol import CausalPolPipeline
from causalpol.pipeline import PipelineConfig

config = PipelineConfig(run_validation=False, run_consistency=False)
pipeline = CausalPolPipeline(config=config)

claims = pipeline.run_extraction_only(
    text="Carbon taxes reduce emissions because they raise the cost of fossil fuels.",
    domain=PolicyDomain.ENVIRONMENTAL_REGULATION,
)

for claim in claims:
    print(claim)
```

---

## Architecture

```
causalpol/
├── pipeline.py              # CausalPolPipeline orchestrator
├── taxonomy/
│   └── schema.py            # CausalClaim, CausalType, EpistemicStatus, ...
├── extraction/
│   └── extractor.py         # CausalSpanExtractor (mDeBERTa-v3) + RuleBasedExtractor
├── validation/
│   └── validator.py         # ClaimValidator (GPT-4o, two-stage triage)
├── consistency/
│   └── detector.py          # InconsistencyDetector (similarity + LLM resolution)
├── evaluation/
│   └── __init__.py          # span_f1, cohens_kappa, classification_report, ...
└── utils/
    └── text.py              # Preprocessing, nominalization detection, hedge parsing
```

### Stage 1: Causal Span Extraction

The extractor fine-tunes `mDeBERTa-v3-base` on PolCausal-50K with a BIO tagging scheme (`B-CAUSE`, `I-CAUSE`, `B-EFFECT`, `I-EFFECT`, `B-MECH`, `I-MECH`, `O`). Three policy-specific mechanisms augment the base model:

1. **Nominalization detector** — identifies deverbal nouns (*reduction*, *disruption*, *contraction*) combined with causal prepositions (*stemming from*, *resulting in*).
2. **Hedge scope parser** — detects modal verbs and conditional markers and assigns `SPECULATIVE` epistemic status to claims within their scope.
3. **Domain terminology encoder** — prepends a domain label to each input to activate domain-adaptive vocabulary learned during fine-tuning.

A rule-based extractor (`RuleBasedExtractor`) is provided as a fallback and lower-bound baseline.

### Stage 2: Claim Validation

Each extracted claim is formatted as a structured proposition and submitted to GPT-4o with a domain-specific prompt. The **uncertainty-aware triage** reduces expert workload by ~60%:

- Score < 2 or > 4 → treated as reliable without expert review
- Score 2–4 → flagged for human expert review (`claim.flagged_for_review = True`)

> **Known bias**: GPT-4o systematically over-rates claims that invoke widely cited mechanisms (*higher rates → lower inflation via demand contraction*) regardless of whether the mechanism applies in the specific context (zero lower bound, supply-side inflation). The validation prompt includes an explicit warning. Monitor this failure mode in new domains.

### Stage 3: Cross-Claim Consistency Detection

1. Build pairwise semantic similarity matrix (sentence-transformers or TF-IDF fallback).
2. Flag candidate pairs above `similarity_threshold` with detected directional opposition.
3. Submit each candidate pair to GPT-4o to resolve genuine inconsistency vs. contextual variation (short-run/long-run, different populations, explicit caveats).

---

## Taxonomy

### Causal Types

| Type | Description | Example |
|------|-------------|---------|
| `MECHANISTIC` | Intermediate mechanism specified | "rates → reduced borrowing → lower demand → lower inflation" |
| `CORRELATIONAL` | Co-occurrence asserted, no mechanism | "high unemployment coincides with lower inflation" |
| `COUNTERFACTUAL` | Conditional assertion about the absence of cause | "without the subsidy, deployment would have stalled" |
| `DEFINITIONAL` | Causal language constitutes a concept | "a recession is caused by two negative GDP quarters" |

### Epistemic Status

| Status | Description |
|--------|-------------|
| `ESTABLISHED` | Presented as settled or evidence-supported |
| `CONTESTED` | Acknowledged uncertainty or expert disagreement |
| `SPECULATIVE` | Forward-looking or explicitly uncertain |

### Policy Domains

`MONETARY_POLICY`, `FISCAL_POLICY`, `PHARMACEUTICAL_REGULATION`, `ENVIRONMENTAL_REGULATION`, `LABOR_MARKET_POLICY`, `TRADE_POLICY`, `TECHNOLOGY_REGULATION`

### Languages

English, French, German, Spanish, Italian, Polish (in order of extraction performance; Section 5.2)

---

## Evaluation

```python
from causalpol.evaluation import span_f1, cohens_kappa, classification_report

# Span-level F1 (Table 1 protocol)
result = span_f1(predicted_spans, gold_spans, overlap_threshold=0.5)
print(f"F1: {result['f1']:.3f}")

# Cohen's κ for validation agreement (Section 6.1)
kappa = cohens_kappa(llm_scores, expert_labels)
print(f"κ: {kappa:.3f}")  # Paper: κ = 0.68

# Causal type classification
report = classification_report(pred_claims, gold_claims, label_field="causal_type")
print(f"Macro-F1: {report['macro_f1']:.3f}")  # Paper: 0.74
```

### Reported Benchmarks (Table 1)

| Domain | CausalPol F1 | Prior Best | Δ |
|--------|-------------|------------|---|
| Central Bank Comms | 0.83 | 0.74 | +8.7 |
| Legislative Text | 0.81 | 0.69 | +11.3 |
| Regulatory Impact | 0.79 | 0.68 | +10.2 |
| Environmental Reg | 0.77 | 0.71 | +6.1 |
| Pharmaceutical Reg | 0.75 | 0.69 | +5.7 |
| Trade Policy | 0.73 | 0.65 | +8.3 |
| Technology Reg | 0.80 | 0.71 | +9.0 |
| **Overall** | **0.79** | **0.71** | **+8.0** |

---

## Configuration

```python
from causalpol.pipeline import PipelineConfig

config = PipelineConfig(
    # Stage toggles
    run_extraction=True,
    run_validation=True,
    run_consistency=True,

    # Extraction
    extraction_model="path/to/causalpol-mdeberta-v3",  # fine-tuned checkpoint
    extraction_device="cuda",
    extraction_confidence_threshold=0.5,
    use_rule_based_fallback=True,

    # Validation (Section 4.2)
    validation_model="gpt-4o",
    triage_low=2,    # Below this: reliably implausible
    triage_high=4,   # Above this: reliably plausible

    # Consistency (Section 4.3)
    consistency_similarity_threshold=0.60,

    # Preprocessing
    auto_detect_language=True,
)
```

---

## Running the Example

```bash
# Extraction only
python examples/example_basic_usage.py --domain monetary_policy

# All domains
python examples/example_basic_usage.py --all-domains

# Full pipeline with validation (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python examples/example_basic_usage.py --validate
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ -v --cov=causalpol --cov-report=term-missing
```

---

## Using a Fine-Tuned Checkpoint

The fine-tuned mDeBERTa-v3 checkpoint trained on PolCausal-50K will be released alongside the paper. To use it:

```python
from causalpol import CausalPolPipeline
from causalpol.pipeline import PipelineConfig

config = PipelineConfig(
    extraction_model="causalpol/mdeberta-v3-polcausal50k",  # HuggingFace Hub path
    extraction_device="cuda",
)
pipeline = CausalPolPipeline(openai_api_key="sk-...", config=config)
```

Until the checkpoint is released, `use_rule_based_fallback=True` (the default) provides functional extraction using the heuristic baseline.

---

## Citation

If you use CausalPol or PolCausal-50K in your research, please cite:

```bibtex
@inproceedings{causalpol2024,
  title     = {{CausalPol}: Extracting and Validating Causal Claims in Policy Documents
               Using Large Language Models},
  author    = {Anonymous},
  booktitle = {Proceedings of [Venue]},
  year      = {2024},
  note      = {Under review}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Known Limitations

- **Domain coverage**: PolCausal-50K covers 7 policy domains and 6 languages. Performance may differ in health policy, social policy, or less-resourced languages.
- **LLM bias**: GPT-4o over-rates mechanistically familiar claims in non-canonical contexts. Monitor false negative rate (~11.3%) for contested claims in new domains.
- **Multilingual performance gradient**: English F1 = 0.83, Polish F1 = 0.71. Polish and other morphologically rich languages are disadvantaged.
- **Proprietary validation model**: Stage 2 requires GPT-4o API access. Reproducibility depends on API availability.
- **Annotation subjectivity**: Inter-annotator agreement reflects genuine expert disagreement (κ = 0.68–0.73); this disagreement is propagated into the benchmark.
