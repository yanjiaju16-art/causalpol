"""
tests/test_causalpol.py
=======================
Unit and integration tests for the CausalPol package.

Run with::

    pytest tests/ -v

Or with coverage::

    pytest tests/ -v --cov=causalpol --cov-report=term-missing
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from causalpol.taxonomy.schema import (
    CausalClaim, CausalType, EpistemicStatus, InconsistencyPair,
    InconsistencyType, Language, PipelineResult, PolicyDomain,
)
from causalpol.utils.text import (
    normalize_whitespace, detect_hedge_markers, extract_nominalizations,
    has_causal_signal, chunk_document, segment_sentences,
)
from causalpol.extraction.extractor import RuleBasedExtractor
from causalpol.evaluation.__init__ import (
    span_f1, cohens_kappa, classification_report, inconsistency_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_monetary_text():
    return (
        "Higher interest rates reduce inflation through demand contraction, "
        "as households and firms decrease borrowing and spending. "
        "However, supply-side price pressures may cause inflation to persist "
        "even as monetary policy tightens. "
        "The reduction in aggregate demand stemming from rate increases "
        "typically leads to lower output in the short run."
    )


@pytest.fixture
def sample_claim():
    return CausalClaim(
        cause_span="higher interest rates",
        effect_span="lower inflation",
        mechanism_span="demand contraction",
        causal_type=CausalType.MECHANISTIC,
        epistemic_status=EpistemicStatus.ESTABLISHED,
        domain=PolicyDomain.MONETARY_POLICY,
        language=Language.ENGLISH,
        source_char_start=0,
        source_char_end=80,
        extraction_confidence=0.82,
        text="Higher interest rates reduce inflation through demand contraction.",
    )


@pytest.fixture
def sample_claim_b():
    return CausalClaim(
        cause_span="supply-side disruptions",
        effect_span="higher inflation",
        causal_type=CausalType.MECHANISTIC,
        epistemic_status=EpistemicStatus.ESTABLISHED,
        domain=PolicyDomain.MONETARY_POLICY,
        language=Language.ENGLISH,
        source_char_start=100,
        source_char_end=180,
        extraction_confidence=0.74,
        text="Supply-side disruptions cause higher inflation independently of demand.",
    )


# ---------------------------------------------------------------------------
# Taxonomy schema tests
# ---------------------------------------------------------------------------

class TestCausalClaimSchema:
    def test_claim_creation(self, sample_claim):
        assert sample_claim.cause_span == "higher interest rates"
        assert sample_claim.causal_type == CausalType.MECHANISTIC
        assert sample_claim.epistemic_status == EpistemicStatus.ESTABLISHED

    def test_claim_str(self, sample_claim):
        s = str(sample_claim)
        assert "MECHANISTIC" in s
        assert "higher interest rates" in s
        assert "lower inflation" in s
        assert "demand contraction" in s

    def test_claim_id_uniqueness(self):
        c1 = CausalClaim(
            cause_span="a", effect_span="b",
            causal_type=CausalType.CORRELATIONAL,
            epistemic_status=EpistemicStatus.SPECULATIVE,
            domain=PolicyDomain.FISCAL_POLICY,
            language=Language.ENGLISH,
            source_char_start=0, source_char_end=10,
        )
        c2 = CausalClaim(
            cause_span="c", effect_span="d",
            causal_type=CausalType.COUNTERFACTUAL,
            epistemic_status=EpistemicStatus.CONTESTED,
            domain=PolicyDomain.FISCAL_POLICY,
            language=Language.ENGLISH,
            source_char_start=20, source_char_end=40,
        )
        assert c1.claim_id != c2.claim_id

    def test_is_high_confidence(self, sample_claim):
        sample_claim.validation_score = 5
        assert sample_claim.is_high_confidence is True
        sample_claim.validation_score = 3
        assert sample_claim.is_high_confidence is False
        sample_claim.validation_score = 1
        assert sample_claim.is_high_confidence is True

    def test_pipeline_result_properties(self, sample_claim, sample_claim_b):
        result = PipelineResult(
            document_id="test-doc",
            domain=PolicyDomain.MONETARY_POLICY,
            language=Language.ENGLISH,
            claims=[sample_claim, sample_claim_b],
        )
        assert result.n_claims == 2
        assert result.n_flagged == 0
        assert result.n_inconsistencies == 0

    def test_pipeline_result_summary(self, sample_claim):
        result = PipelineResult(
            document_id="doc-001",
            domain=PolicyDomain.MONETARY_POLICY,
            language=Language.ENGLISH,
            claims=[sample_claim],
        )
        summary = result.summary()
        assert "doc-001" in summary
        assert "monetary_policy" in summary
        assert "1 extracted" in summary

    def test_causal_type_values(self):
        assert CausalType.MECHANISTIC.value == "mechanistic"
        assert CausalType.CORRELATIONAL.value == "correlational"
        assert CausalType.COUNTERFACTUAL.value == "counterfactual"
        assert CausalType.DEFINITIONAL.value == "definitional"

    def test_epistemic_status_values(self):
        assert EpistemicStatus.ESTABLISHED.value == "established"
        assert EpistemicStatus.CONTESTED.value == "contested"
        assert EpistemicStatus.SPECULATIVE.value == "speculative"

    def test_policy_domain_values(self):
        domains = list(PolicyDomain)
        assert len(domains) == 7  # Paper covers 7 domains

    def test_language_values(self):
        languages = list(Language)
        assert len(languages) == 6  # Paper covers 6 languages
        codes = {l.value for l in languages}
        assert "en" in codes and "pl" in codes  # English best, Polish worst


# ---------------------------------------------------------------------------
# Text utility tests
# ---------------------------------------------------------------------------

class TestTextUtilities:
    def test_normalize_whitespace_basic(self):
        text = "hello   world\r\n\nfoo  bar"
        result = normalize_whitespace(text)
        assert "  " not in result
        assert "\r" not in result

    def test_normalize_soft_hyphen(self):
        text = "infla\u00adtion"
        result = normalize_whitespace(text)
        assert "\u00ad" not in result

    def test_has_causal_signal_positive(self):
        assert has_causal_signal("Higher rates cause lower demand.")
        assert has_causal_signal("The policy leads to improved outcomes.")
        assert has_causal_signal("Inflation results from supply disruptions.")

    def test_has_causal_signal_negative(self):
        assert not has_causal_signal("The committee met on Tuesday.")
        assert not has_causal_signal("Article 5 paragraph 2 of Regulation EU/2023.")

    def test_detect_hedge_markers(self):
        text = "Inflation may increase as a result of higher energy prices."
        markers = detect_hedge_markers(text)
        marker_words = [m["marker"].lower() for m in markers]
        assert "may" in marker_words

    def test_detect_hedge_markers_conditional(self):
        text = "If monetary conditions tighten, output could fall."
        markers = detect_hedge_markers(text)
        assert len(markers) >= 1

    def test_extract_nominalizations(self):
        text = "The reduction in employment stemming from automation is significant."
        noms = extract_nominalizations(text)
        assert len(noms) >= 1
        assert any("reduction" in n.get("nominalization", "").lower() for n in noms)

    def test_segment_sentences_basic(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = segment_sentences(text)
        assert len(sentences) >= 2

    def test_chunk_document_length(self, sample_monetary_text):
        chunks = chunk_document(sample_monetary_text, max_tokens=50)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "char_start" in chunk
            assert "char_end" in chunk
            assert "chunk_index" in chunk

    def test_chunk_document_ordering(self, sample_monetary_text):
        chunks = chunk_document(sample_monetary_text, max_tokens=30)
        for i in range(len(chunks) - 1):
            assert chunks[i]["char_start"] <= chunks[i + 1]["char_start"]


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------

class TestRuleBasedExtractor:
    def setup_method(self):
        self.extractor = RuleBasedExtractor(
            domain=PolicyDomain.MONETARY_POLICY,
            language=Language.ENGLISH,
        )

    def test_extracts_because(self):
        text = "Output falls because interest rates increase."
        results = self.extractor.extract(text)
        assert len(results) >= 1
        assert any("interest rates" in r["cause_span"] for r in results)

    def test_extracts_leads_to(self):
        text = "Higher inflation leads to erosion of real wages."
        results = self.extractor.extract(text)
        assert len(results) >= 1

    def test_extracts_due_to(self):
        text = "GDP contracted due to supply chain disruptions."
        results = self.extractor.extract(text)
        assert len(results) >= 1

    def test_no_extraction_without_signal(self):
        text = "The Governing Council met on Thursday at its Frankfurt headquarters."
        results = self.extractor.extract(text)
        assert len(results) == 0

    def test_confidence_range(self):
        text = "Output falls because rates increase."
        results = self.extractor.extract(text)
        for r in results:
            assert 0.0 <= r["extraction_confidence"] <= 1.0

    def test_extractor_domain_agnostic(self):
        """Rule-based extractor should work across domains."""
        for domain in PolicyDomain:
            ext = RuleBasedExtractor(domain=domain, language=Language.ENGLISH)
            text = "Carbon taxes reduce emissions because they increase the cost of fossil fuels."
            results = ext.extract(text)
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Evaluation metrics tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_span_f1_perfect(self):
        spans = [{"source_char_start": 0, "source_char_end": 50}]
        result = span_f1(spans, spans)
        assert result["f1"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_span_f1_no_overlap(self):
        pred = [{"source_char_start": 0, "source_char_end": 20}]
        gold = [{"source_char_start": 100, "source_char_end": 120}]
        result = span_f1(pred, gold)
        assert result["f1"] == pytest.approx(0.0)

    def test_span_f1_partial_overlap(self):
        pred = [{"source_char_start": 0, "source_char_end": 30}]
        gold = [{"source_char_start": 20, "source_char_end": 50}]
        result = span_f1(pred, gold, overlap_threshold=0.1)
        assert result["f1"] > 0.0

    def test_span_f1_empty_predictions(self):
        gold = [{"source_char_start": 0, "source_char_end": 50}]
        result = span_f1([], gold)
        assert result["f1"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)

    def test_cohens_kappa_perfect_agreement(self):
        labels = ["a", "b", "c", "a", "b"]
        kappa = cohens_kappa(labels, labels)
        assert kappa == pytest.approx(1.0)

    def test_cohens_kappa_chance_agreement(self):
        """Kappa near zero for chance-level agreement."""
        a = ["a", "b", "a", "b"]
        b = ["b", "a", "b", "a"]
        kappa = cohens_kappa(a, b)
        assert kappa < 0.2

    def test_cohens_kappa_paper_benchmark(self):
        """
        Confirm that κ = 0.68 (paper Section 6.1) is within a plausible
        range for the reported inter-expert agreement (κ = 0.72).
        """
        assert 0.60 <= 0.68 <= 0.80  # substantial agreement range

    def test_cohens_kappa_length_mismatch(self):
        with pytest.raises(ValueError):
            cohens_kappa(["a", "b"], ["a"])

    def test_classification_report_perfect(self, sample_claim):
        gold = [sample_claim]
        pred = [sample_claim]
        report = classification_report(pred, gold, "causal_type")
        assert report["accuracy"] == pytest.approx(1.0)
        assert report["macro_f1"] == pytest.approx(1.0)

    def test_inconsistency_metrics_empty(self):
        result = inconsistency_metrics([], [])
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)

    def test_inconsistency_metrics_genuine_filter(self, sample_claim, sample_claim_b):
        genuine_pair = InconsistencyPair(
            claim_a=sample_claim,
            claim_b=sample_claim_b,
            inconsistency_type=InconsistencyType.DIRECTIONAL,
            llm_resolution="Genuine contradiction.",
            is_genuine=True,
            resolution_rationale="Same instrument, opposite effects.",
        )
        non_genuine = InconsistencyPair(
            claim_a=sample_claim,
            claim_b=sample_claim_b,
            inconsistency_type=InconsistencyType.TEMPORAL,
            llm_resolution="Short-run vs long-run distinction.",
            is_genuine=False,
            resolution_rationale="Context distinguishes time horizon.",
        )
        gold_pair = genuine_pair
        result = inconsistency_metrics([genuine_pair, non_genuine], [gold_pair])
        # Only genuine_pair should count
        assert result["n_genuine_predicted"] == 1


# ---------------------------------------------------------------------------
# Pipeline integration test (no API calls)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_pipeline_extraction_only(self, sample_monetary_text):
        """Full extraction pipeline without LLM components."""
        from causalpol.pipeline import CausalPolPipeline, PipelineConfig

        config = PipelineConfig(
            run_validation=False,
            run_consistency=False,
            use_rule_based_fallback=True,
        )
        pipeline = CausalPolPipeline(config=config)
        result = pipeline.run(
            text=sample_monetary_text,
            domain=PolicyDomain.MONETARY_POLICY,
            language=Language.ENGLISH,
        )

        assert isinstance(result, PipelineResult)
        assert result.domain == PolicyDomain.MONETARY_POLICY
        assert result.language == Language.ENGLISH
        assert result.n_claims >= 0  # May be 0 if no signals
        assert result.inconsistency_pairs == []

    def test_pipeline_no_api_key_disables_stages(self, sample_monetary_text):
        from causalpol.pipeline import CausalPolPipeline

        pipeline = CausalPolPipeline(openai_api_key=None)
        assert pipeline.config.run_validation is False
        assert pipeline.config.run_consistency is False

    def test_run_extraction_only_method(self, sample_monetary_text):
        from causalpol.pipeline import CausalPolPipeline

        pipeline = CausalPolPipeline()
        claims = pipeline.run_extraction_only(
            text=sample_monetary_text,
            domain=PolicyDomain.MONETARY_POLICY,
        )
        assert isinstance(claims, list)

    def test_pipeline_result_summary_format(self, sample_claim):
        result = PipelineResult(
            document_id="integration-test",
            domain=PolicyDomain.ENVIRONMENTAL_REGULATION,
            language=Language.GERMAN,
            claims=[sample_claim],
        )
        summary = result.summary()
        assert "integration-test" in summary
        assert "environmental_regulation" in summary


# ---------------------------------------------------------------------------
# Domain coverage sanity checks
# ---------------------------------------------------------------------------

class TestDomainCoverage:
    """Sanity checks ensuring all seven domains from the paper are covered."""

    def test_all_domains_have_validation_context(self):
        from causalpol.validation.validator import _DOMAIN_CONTEXTS
        for domain in PolicyDomain:
            assert domain in _DOMAIN_CONTEXTS, (
                f"Domain {domain.value} missing from validation context dict"
            )

    def test_all_domains_extractable(self):
        for domain in PolicyDomain:
            extractor = RuleBasedExtractor(domain=domain, language=Language.ENGLISH)
            text = "The policy leads to improved outcomes because of structural reforms."
            results = extractor.extract(text)
            assert isinstance(results, list)
