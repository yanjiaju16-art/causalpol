"""
causalpol.pipeline
==================
Top-level CausalPol pipeline orchestrator.

CausalPolPipeline orchestrates the three sequential tasks described in
Section 2 of the paper:

1. **Causal span extraction** (Section 4.1) — using fine-tuned mDeBERTa-v3.
2. **Claim validation** (Section 4.2) — GPT-4o plausibility assessment with
   uncertainty-aware triage.
3. **Cross-claim consistency detection** (Section 4.3) — pairwise semantic
   similarity and LLM-based resolution.

Each stage can be run independently or as part of the full pipeline.

Classes
-------
CausalPolPipeline
    Main pipeline class.  Accepts configuration for all three stages and
    exposes a single ``run()`` method for end-to-end analysis.

PipelineConfig
    Dataclass for pipeline hyperparameters.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from causalpol.consistency.detector import InconsistencyDetector
from causalpol.extraction.extractor import CausalSpanExtractor
from causalpol.taxonomy.schema import Language, PipelineResult, PolicyDomain
from causalpol.utils.text import detect_language, normalize_whitespace
from causalpol.validation.validator import ClaimValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Configuration dataclass for CausalPolPipeline.

    Parameters
    ----------
    run_extraction : bool
        Whether to run Stage 1 (causal span extraction). Default True.
    run_validation : bool
        Whether to run Stage 2 (claim validation). Default True.
        Requires ``openai_api_key``.
    run_consistency : bool
        Whether to run Stage 3 (cross-claim consistency detection).
        Default True.  Requires ``openai_api_key``.
    extraction_model : str
        HuggingFace model path for the span extractor.  Use the fine-tuned
        PolCausal-50K checkpoint when available.
    extraction_device : str
        PyTorch device string.
    extraction_confidence_threshold : float
        Minimum per-span confidence for inclusion (Section 4.1).
    use_rule_based_fallback : bool
        Use the heuristic extractor when the transformer model is
        unavailable.
    validation_model : str
        OpenAI model for claim validation.
    triage_low : int
        Lower triage bound for expert review flagging (default 2).
    triage_high : int
        Upper triage bound for expert review flagging (default 4).
    consistency_similarity_threshold : float
        Minimum semantic similarity to flag a claim pair as a candidate
        inconsistency (default 0.60).
    auto_detect_language : bool
        Automatically detect document language when not specified.
    """

    run_extraction: bool = True
    run_validation: bool = True
    run_consistency: bool = True

    # Extraction
    extraction_model: str = "microsoft/mdeberta-v3-base"
    extraction_device: str = "cpu"
    extraction_confidence_threshold: float = 0.5
    use_rule_based_fallback: bool = True

    # Validation
    validation_model: str = "gpt-4o"
    triage_low: int = 2
    triage_high: int = 4

    # Consistency
    consistency_similarity_threshold: float = 0.60

    # Preprocessing
    auto_detect_language: bool = True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CausalPolPipeline:
    """
    End-to-end CausalPol pipeline for extracting and validating causal claims
    in policy documents.

    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key.  Required for validation and consistency detection.
        If not provided, only extraction is run.
    config : PipelineConfig, optional
        Pipeline configuration.  Uses defaults when not specified.

    Examples
    --------
    **Full pipeline (requires OpenAI key):**

    >>> from causalpol import CausalPolPipeline
    >>> from causalpol.taxonomy.schema import PolicyDomain, Language
    >>>
    >>> pipeline = CausalPolPipeline(openai_api_key="sk-...")
    >>> result = pipeline.run(
    ...     text="Higher interest rates reduce inflation through demand "
    ...          "contraction, but supply constraints cause inflation "
    ...          "independent of demand conditions.",
    ...     domain=PolicyDomain.MONETARY_POLICY,
    ...     language=Language.ENGLISH,
    ... )
    >>> print(result.summary())

    **Extraction only (no API key needed):**

    >>> from causalpol import CausalPolPipeline
    >>> from causalpol.pipeline import PipelineConfig
    >>>
    >>> cfg = PipelineConfig(run_validation=False, run_consistency=False)
    >>> pipeline = CausalPolPipeline(config=cfg)
    >>> result = pipeline.run(text="...", domain=...)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.openai_api_key = openai_api_key
        self.config = config or PipelineConfig()

        if openai_api_key is None:
            if self.config.run_validation:
                logger.warning(
                    "openai_api_key not provided; disabling validation stage."
                )
                self.config.run_validation = False
            if self.config.run_consistency:
                logger.warning(
                    "openai_api_key not provided; disabling consistency detection."
                )
                self.config.run_consistency = False

        # Lazy-initialize stage components (per domain/language)
        self._extractors: dict[tuple, CausalSpanExtractor] = {}
        self._validators: dict[PolicyDomain, ClaimValidator] = {}
        self._detector: Optional[InconsistencyDetector] = None

    def _get_extractor(
        self, domain: PolicyDomain, language: Language
    ) -> CausalSpanExtractor:
        key = (domain, language)
        if key not in self._extractors:
            self._extractors[key] = CausalSpanExtractor(
                model_name_or_path=self.config.extraction_model,
                domain=domain,
                language=language,
                device=self.config.extraction_device,
                use_rule_based_fallback=self.config.use_rule_based_fallback,
                confidence_threshold=self.config.extraction_confidence_threshold,
            )
        return self._extractors[key]

    def _get_validator(self, domain: PolicyDomain) -> ClaimValidator:
        if domain not in self._validators:
            self._validators[domain] = ClaimValidator(
                openai_api_key=self.openai_api_key,
                domain=domain,
                model=self.config.validation_model,
                triage_low=self.config.triage_low,
                triage_high=self.config.triage_high,
            )
        return self._validators[domain]

    def _get_detector(self) -> InconsistencyDetector:
        if self._detector is None:
            self._detector = InconsistencyDetector(
                openai_api_key=self.openai_api_key,
                similarity_threshold=self.config.consistency_similarity_threshold,
            )
        return self._detector

    def run(
        self,
        text: str,
        domain: PolicyDomain = PolicyDomain.MONETARY_POLICY,
        language: Optional[Language] = None,
        document_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full CausalPol pipeline on a policy document.

        Parameters
        ----------
        text : str
            Raw document text.  May be a full document, a section, or a
            paragraph.  The pipeline handles chunking internally.
        domain : PolicyDomain
            Policy domain of the document.
        language : Language, optional
            Source language.  If None and ``config.auto_detect_language`` is
            True, language is detected automatically.
        document_id : str, optional
            Identifier for tracking.  Generated automatically if not provided.

        Returns
        -------
        PipelineResult
            Container with all extracted claims, inconsistency pairs, and
            summary statistics.
        """
        doc_id = document_id or str(uuid.uuid4())
        text = normalize_whitespace(text)

        # Language detection
        if language is None and self.config.auto_detect_language:
            language = detect_language(text)
            logger.info(f"Auto-detected language: {language.value}")
        elif language is None:
            language = Language.ENGLISH

        result = PipelineResult(
            document_id=doc_id,
            domain=domain,
            language=language,
            raw_text=text,
        )

        # Stage 1: Extraction
        if self.config.run_extraction:
            logger.info(f"[Stage 1] Extracting causal claims from document {doc_id}...")
            extractor = self._get_extractor(domain, language)
            result.claims = extractor.extract_from_text(text)
            logger.info(f"[Stage 1] Extracted {result.n_claims} claims.")
        else:
            logger.info("[Stage 1] Skipped (run_extraction=False).")

        # Stage 2: Validation
        if self.config.run_validation and result.claims:
            logger.info(f"[Stage 2] Validating {result.n_claims} claims...")
            validator = self._get_validator(domain)
            result.claims = validator.validate_claims(result.claims)
            stats = validator.compute_validation_statistics(result.claims)
            logger.info(
                f"[Stage 2] Validation complete. "
                f"Mean score={stats['mean_score']:.2f}, "
                f"flagged={stats['n_flagged']}/{stats['n_validated']}"
            )
        elif self.config.run_validation and not result.claims:
            logger.info("[Stage 2] Skipped (no claims to validate).")
        else:
            logger.info("[Stage 2] Skipped (run_validation=False).")

        # Stage 3: Consistency detection
        if self.config.run_consistency and len(result.claims) >= 2:
            logger.info(f"[Stage 3] Running consistency detection on {result.n_claims} claims...")
            detector = self._get_detector()
            result.inconsistency_pairs = detector.detect(result.claims)
            n_genuine = result.n_inconsistencies
            logger.info(
                f"[Stage 3] Found {n_genuine} genuine inconsistencies "
                f"({len(result.inconsistency_pairs)} candidates)."
            )
        elif self.config.run_consistency:
            logger.info("[Stage 3] Skipped (fewer than 2 claims).")
        else:
            logger.info("[Stage 3] Skipped (run_consistency=False).")

        return result

    def run_extraction_only(
        self,
        text: str,
        domain: PolicyDomain = PolicyDomain.MONETARY_POLICY,
        language: Optional[Language] = None,
    ):
        """
        Run only the extraction stage.  Convenience method that does not
        require an OpenAI API key.

        Parameters
        ----------
        text : str
        domain : PolicyDomain
        language : Language, optional

        Returns
        -------
        list[CausalClaim]
        """
        text = normalize_whitespace(text)
        if language is None:
            language = detect_language(text) if self.config.auto_detect_language \
                else Language.ENGLISH
        extractor = self._get_extractor(domain, language)
        return extractor.extract_from_text(text)

    def run_validation_only(self, claims, domain: PolicyDomain = PolicyDomain.MONETARY_POLICY):
        """
        Run only the validation stage on pre-extracted claims.

        Parameters
        ----------
        claims : list[CausalClaim]
        domain : PolicyDomain

        Returns
        -------
        list[CausalClaim]
        """
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required for validation.")
        validator = self._get_validator(domain)
        return validator.validate_claims(claims)

    def run_consistency_only(self, claims):
        """
        Run only the consistency detection stage on pre-extracted claims.

        Parameters
        ----------
        claims : list[CausalClaim]

        Returns
        -------
        list[InconsistencyPair]
        """
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required for consistency detection.")
        return self._get_detector().detect(claims)
