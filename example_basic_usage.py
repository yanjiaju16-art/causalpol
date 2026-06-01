"""
examples/example_basic_usage.py
================================
Basic usage walkthrough for researchers adopting CausalPol.

This script demonstrates the three pipeline stages on a short monetary
policy excerpt.  It is designed to run without GPU access and without an
OpenAI API key (using rule-based extraction and mock validation).

To run with full LLM validation, set the OPENAI_API_KEY environment
variable and pass it to CausalPolPipeline.

Usage::

    python examples/example_basic_usage.py

    # With API key for full validation:
    OPENAI_API_KEY=sk-... python examples/example_basic_usage.py --validate
"""

import argparse
import json
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from causalpol import CausalPolPipeline
from causalpol.pipeline import PipelineConfig
from causalpol.taxonomy.schema import PolicyDomain, Language


# ---------------------------------------------------------------------------
# Sample policy texts from different domains
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = {
    "monetary_policy": {
        "text": (
            "The elevated level of inflation reflects a combination of supply-side "
            "disruptions and demand-side pressures amplified by energy price pass-through "
            "effects. Higher interest rates reduce inflation through demand contraction, "
            "as households and firms decrease borrowing and spending. However, supply-side "
            "price pressures may cause inflation to persist even as monetary policy "
            "tightens, because the transmission mechanism operates primarily through the "
            "demand channel. Without the tightening measures, inflation could have risen "
            "further, leading to a de-anchoring of inflation expectations. The elevated "
            "level of inflation causes real income losses for households, particularly "
            "those with fixed nominal incomes."
        ),
        "domain": PolicyDomain.MONETARY_POLICY,
        "language": Language.ENGLISH,
        "description": "ECB-style central bank communication"
    },
    "environmental_regulation": {
        "text": (
            "Carbon pricing mechanisms reduce greenhouse gas emissions by increasing "
            "the cost of carbon-intensive production, thereby incentivising firms to "
            "invest in cleaner technologies. However, in the absence of border carbon "
            "adjustments, carbon pricing may lead to carbon leakage as production "
            "migrates to jurisdictions with less stringent environmental standards. "
            "The regulatory framework is expected to generate co-benefits for air "
            "quality, as reduced fossil fuel combustion causes lower concentrations "
            "of particulate matter and nitrogen oxides."
        ),
        "domain": PolicyDomain.ENVIRONMENTAL_REGULATION,
        "language": Language.ENGLISH,
        "description": "EU environmental regulatory impact assessment"
    },
    "trade_policy": {
        "text": (
            "Tariff increases on imported steel raise domestic production costs for "
            "downstream manufacturing sectors, because steel is a key input in "
            "automobiles, machinery, and construction. The higher input costs lead to "
            "reduced competitiveness of domestic manufacturers in export markets. "
            "Trade liberalization, by contrast, promotes specialization according to "
            "comparative advantage, which results in efficiency gains and higher "
            "aggregate output. Without preferential trade agreements, bilateral trade "
            "volumes would decline, causing economic welfare losses for both parties."
        ),
        "domain": PolicyDomain.TRADE_POLICY,
        "language": Language.ENGLISH,
        "description": "WTO-style trade policy analysis"
    },
}


def run_example(document_key: str, validate: bool = False, api_key: str = None):
    """Run CausalPol on a sample document."""
    doc = SAMPLE_DOCUMENTS[document_key]
    print(f"\n{'=' * 70}")
    print(f"Document: {doc['description']}")
    print(f"Domain:   {doc['domain'].value}")
    print(f"Language: {doc['language'].value}")
    print(f"{'=' * 70}")
    print(f"\nText:\n{doc['text']}\n")

    # Configure pipeline
    config = PipelineConfig(
        run_validation=validate and api_key is not None,
        run_consistency=validate and api_key is not None,
        use_rule_based_fallback=True,
    )

    if validate and api_key is None:
        print("[WARNING] No OpenAI API key provided. Running extraction only.")

    pipeline = CausalPolPipeline(openai_api_key=api_key, config=config)

    # Run pipeline
    result = pipeline.run(
        text=doc["text"],
        domain=doc["domain"],
        language=doc["language"],
        document_id=f"example-{document_key}",
    )

    # Print results
    print(result.summary())

    # Show causal type distribution
    if result.claims:
        print("\n--- Causal Type Distribution ---")
        from collections import Counter
        type_counts = Counter(c.causal_type.value for c in result.claims)
        for ctype, count in sorted(type_counts.items()):
            print(f"  {ctype:20s}: {count}")

        print("\n--- Epistemic Status Distribution ---")
        status_counts = Counter(c.epistemic_status.value for c in result.claims)
        for status, count in sorted(status_counts.items()):
            print(f"  {status:20s}: {count}")

    return result


def demonstrate_utility_functions():
    """Demonstrate individual utility functions for researchers."""
    from causalpol.utils.text import (
        extract_nominalizations, detect_hedge_markers,
        has_causal_signal, chunk_document,
    )

    print(f"\n{'=' * 70}")
    print("Utility Function Demonstrations")
    print(f"{'=' * 70}")

    # Nominalization detection
    text = ("The reduction in employment stemming from automation is "
            "projected to affect low-skilled workers most severely.")
    print(f"\n[Nominalization Detection]\nText: {text}")
    noms = extract_nominalizations(text)
    for n in noms:
        print(f"  Found: '{n['span'][:60]}...' (nominalization: {n['nominalization']})")

    # Hedge detection
    text2 = ("Higher interest rates may reduce inflation, but this effect "
             "could be offset if supply constraints persist.")
    print(f"\n[Hedge Marker Detection]\nText: {text2}")
    hedges = detect_hedge_markers(text2)
    for h in hedges:
        print(f"  Hedge: '{h['marker']}' at position {h['start']}-{h['end']}")

    # Causal signal check
    texts_to_check = [
        "The council met on Thursday to discuss the agenda.",
        "Carbon taxes reduce emissions because they raise the cost of fossil fuels.",
        "Article 5(2) of Regulation EU/2021/123 shall apply.",
    ]
    print("\n[Causal Signal Detection]")
    for t in texts_to_check:
        has_sig = has_causal_signal(t)
        print(f"  {'✓' if has_sig else '✗'} {t[:70]}")


def demonstrate_evaluation_metrics():
    """Demonstrate the evaluation metrics module."""
    from causalpol.evaluation import (
        span_f1, cohens_kappa, classification_report
    )

    print(f"\n{'=' * 70}")
    print("Evaluation Metrics Demonstrations")
    print(f"{'=' * 70}")

    # Span F1
    pred_spans = [
        {"source_char_start": 0, "source_char_end": 45},
        {"source_char_start": 120, "source_char_end": 165},
    ]
    gold_spans = [
        {"source_char_start": 5, "source_char_end": 48},  # partial overlap
        {"source_char_start": 118, "source_char_end": 162},
    ]
    f1_result = span_f1(pred_spans, gold_spans, overlap_threshold=0.5)
    print(f"\n[Span-Level F1 (Section 5.1)]")
    print(f"  Precision: {f1_result['precision']:.3f}")
    print(f"  Recall:    {f1_result['recall']:.3f}")
    print(f"  F1:        {f1_result['f1']:.3f}")

    # Cohen's kappa
    print(f"\n[Cohen's κ (Section 6.1)]")
    # Simulate GPT-4o scores vs. expert labels (simplified)
    llm_scores = ["4", "5", "3", "2", "4", "5", "3", "4", "2", "5"]
    expert_labels = ["4", "4", "3", "2", "4", "5", "2", "4", "3", "5"]
    kappa = cohens_kappa(llm_scores, expert_labels)
    print(f"  Simulated GPT-4o vs. expert κ: {kappa:.3f}")
    print(f"  (Paper reports κ = 0.68 on PolCausal-50K held-out set)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CausalPol basic usage example"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run LLM validation (requires OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--domain", default="monetary_policy",
        choices=list(SAMPLE_DOCUMENTS.keys()),
        help="Which sample domain to analyze"
    )
    parser.add_argument(
        "--all-domains", action="store_true",
        help="Run on all sample domains"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") if args.validate else None

    if args.all_domains:
        for key in SAMPLE_DOCUMENTS:
            run_example(key, validate=args.validate, api_key=api_key)
    else:
        run_example(args.domain, validate=args.validate, api_key=api_key)

    demonstrate_utility_functions()
    demonstrate_evaluation_metrics()
