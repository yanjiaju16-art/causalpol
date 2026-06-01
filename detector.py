"""
causalpol.consistency.detector
===============================
Cross-claim inconsistency detection (Section 4.3 of the paper).

The detector identifies pairs of causal claims within the same document
that express logically contradictory causal mechanisms.  The pipeline
operates in two steps:

1. **Candidate identification** — a pairwise semantic similarity matrix
   is constructed over all claims.  Pairs above a similarity threshold
   (claims about the same cause-effect topic) are flagged as candidates.
   Directional opposition is then detected using antonym matching and
   directional language patterns.

2. **LLM resolution** — each candidate pair is submitted to GPT-4o to
   determine whether the apparent inconsistency is:

   * **Genuine** — the same policy instrument is claimed to produce
     contradictory outcomes with no contextual justification.
   * **Contextual variation** — the contradiction is explained by
     appropriately distinguished contexts (e.g., short-run vs. long-run,
     different populations, conditional effects).

The four inconsistency types from Table 2 of the paper are:
``DIRECTIONAL`` (2.8%), ``MAGNITUDE`` (1.9%), ``SCOPE`` (1.1%),
``TEMPORAL`` (0.5%).

Classes
-------
InconsistencyDetector
    Main entry point.  Accepts a list of CausalClaim objects and returns
    InconsistencyPair objects for all detected inconsistencies.

SimilarityIndex
    Lightweight semantic similarity index using sentence-transformers
    (or TF-IDF cosine similarity as a fallback).

DirectionalOppositionDetector
    Detects directional contradictions using lexical patterns.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

from causalpol.taxonomy.schema import (
    CausalClaim, InconsistencyPair, InconsistencyType
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directional opposition patterns
# ---------------------------------------------------------------------------

# Pairs of opposing directional verbs/adjectives common in policy text
_OPPOSING_PAIRS = [
    ({"increase", "increases", "increased", "raise", "raises", "raised",
      "higher", "rise", "rises", "rose", "boost", "boosts", "expand",
      "expands", "expanded", "stimulate", "stimulates", "promote", "promotes"},
     {"decrease", "decreases", "decreased", "reduce", "reduces", "reduced",
      "lower", "fall", "falls", "fell", "decline", "declines", "declined",
      "contract", "contracts", "contracted", "dampen", "dampens", "suppress",
      "suppresses", "curtail", "curtails"}),
    ({"prevent", "prevents", "prevented", "avoid", "avoids", "eliminate",
      "eliminates"},
     {"cause", "causes", "caused", "lead", "leads", "produce", "produces",
      "generate", "generates", "create", "creates"}),
    ({"strengthen", "strengthens", "improve", "improves", "improved"},
     {"weaken", "weakens", "worsen", "worsens", "deteriorate", "deteriorates"}),
]


class DirectionalOppositionDetector:
    """
    Detect directional opposition between cause-effect claims using lexical
    patterns.

    Parameters
    ----------
    threshold : float
        Proportion of opposing words required to flag a pair as directionally
        opposed.  Default: at least one opposing token found.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def are_opposed(self, claim_a: CausalClaim, claim_b: CausalClaim) -> tuple[bool, str]:
        """
        Return (is_opposed, direction_description) for a pair of claims.

        Parameters
        ----------
        claim_a, claim_b : CausalClaim

        Returns
        -------
        tuple[bool, str]
            ``(True, description)`` if a directional opposition is found.
        """
        text_a = f"{claim_a.cause_span} {claim_a.effect_span}".lower()
        text_b = f"{claim_b.cause_span} {claim_b.effect_span}".lower()

        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())

        for pos_set, neg_set in _OPPOSING_PAIRS:
            a_positive = bool(tokens_a & pos_set)
            a_negative = bool(tokens_a & neg_set)
            b_positive = bool(tokens_b & pos_set)
            b_negative = bool(tokens_b & neg_set)

            if (a_positive and b_negative) or (a_negative and b_positive):
                direction = (
                    f"Claim A asserts {'increase/positive' if a_positive else 'decrease/negative'} "
                    f"direction; Claim B asserts the opposite."
                )
                return True, direction

        # Check for explicit negation opposition
        negation_pat = re.compile(r"\b(not|no|without|absent|lack of)\b", re.IGNORECASE)
        a_negated = bool(negation_pat.search(text_a))
        b_negated = bool(negation_pat.search(text_b))
        if a_negated != b_negated:
            return True, "One claim uses explicit negation while the other asserts the same relationship positively."

        return False, ""


# ---------------------------------------------------------------------------
# Semantic similarity index
# ---------------------------------------------------------------------------

class SimilarityIndex:
    """
    Semantic similarity index for causal claims.

    Uses ``sentence-transformers`` when available; falls back to TF-IDF
    cosine similarity computed with scikit-learn, then to Jaccard similarity
    as a final fallback.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model name.  Default is a lightweight
        multilingual model suitable for policy text.
    """

    _DEFAULT_SBERT = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = _DEFAULT_SBERT):
        self.model_name = model_name
        self._sbert = None
        self._tfidf = None
        self._backend = self._init_backend()

    def _init_backend(self) -> str:
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.model_name)
            logger.info("Similarity backend: sentence-transformers")
            return "sbert"
        except ImportError:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
            logger.info("Similarity backend: TF-IDF (scikit-learn)")
            return "tfidf"
        except ImportError:
            logger.warning("No similarity library available; using Jaccard fallback.")
            return "jaccard"

    def _claim_text(self, claim: CausalClaim) -> str:
        parts = [claim.cause_span, claim.effect_span]
        if claim.mechanism_span:
            parts.append(claim.mechanism_span)
        return " ".join(parts)

    def _cosine(self, vec_a, vec_b) -> float:
        """Compute cosine similarity between two dense or sparse vectors."""
        try:
            import numpy as np
            a = np.array(vec_a).flatten()
            b = np.array(vec_b).flatten()
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0
        except Exception:
            return 0.0

    def _jaccard(self, text_a: str, text_b: str) -> float:
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def similarity(self, claim_a: CausalClaim, claim_b: CausalClaim) -> float:
        """
        Compute semantic similarity between two claims.

        Parameters
        ----------
        claim_a, claim_b : CausalClaim

        Returns
        -------
        float
            Similarity score in [0, 1].
        """
        text_a = self._claim_text(claim_a)
        text_b = self._claim_text(claim_b)

        if self._backend == "sbert":
            embeddings = self._sbert.encode([text_a, text_b], convert_to_numpy=True)
            return float(self._cosine(embeddings[0], embeddings[1]))

        if self._backend == "tfidf":
            try:
                matrix = self._tfidf.fit_transform([text_a, text_b])
                vec_a = matrix[0].toarray()
                vec_b = matrix[1].toarray()
                return self._cosine(vec_a, vec_b)
            except Exception:
                pass

        return self._jaccard(text_a, text_b)

    def pairwise_matrix(self, claims: list[CausalClaim]) -> list[list[float]]:
        """
        Compute the full pairwise similarity matrix for *claims*.

        Parameters
        ----------
        claims : list[CausalClaim]

        Returns
        -------
        list[list[float]]
            n×n matrix of similarity scores.
        """
        n = len(claims)
        matrix = [[0.0] * n for _ in range(n)]

        if self._backend == "sbert":
            texts = [self._claim_text(c) for c in claims]
            embeddings = self._sbert.encode(texts, convert_to_numpy=True)
            for i in range(n):
                for j in range(i, n):
                    sim = float(self._cosine(embeddings[i], embeddings[j]))
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        else:
            for i in range(n):
                for j in range(i, n):
                    sim = self.similarity(claims[i], claims[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix


# ---------------------------------------------------------------------------
# LLM resolution prompt
# ---------------------------------------------------------------------------

_RESOLUTION_SYSTEM = """You are an expert policy analyst assessing whether two causal
claims extracted from the same policy document are genuinely inconsistent.

A genuine inconsistency means: the same policy instrument is claimed to produce
contradictory outcomes (opposite directions, incompatible magnitudes, or mutually
exclusive scope) with NO adequate contextual justification provided in the document.

Contextual variation is NOT an inconsistency when:
- The document explicitly distinguishes short-run vs. long-run effects
- The claims apply to clearly different populations or geographies
- One claim is explicitly presented as a caveat or exception to the other
- The claims apply under different policy scenarios (baseline vs. alternative)

Respond ONLY with valid JSON (no markdown, no preamble):
{
  "is_genuine_inconsistency": <boolean>,
  "inconsistency_type": "<directional|magnitude|scope|temporal|none>",
  "resolution_rationale": "<2-3 sentence explanation>",
  "contextual_variation_explanation": "<if not genuine, explain the contextual distinction>",
  "severity": "<high|medium|low>"
}
"""


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class InconsistencyDetector:
    """
    Cross-claim inconsistency detector (Section 4.3 of the paper).

    Identifies pairs of causal claims within the same document that assert
    contradictory causal mechanisms, then resolves whether each apparent
    inconsistency is genuine or contextually explained.

    Parameters
    ----------
    openai_api_key : str
        OpenAI API key (required for LLM-based resolution).
    similarity_threshold : float
        Minimum semantic similarity for a pair to be considered a candidate.
        Default 0.60 balances precision and recall on PolCausal-50K.
    model : str
        OpenAI model for inconsistency resolution.
    sbert_model : str
        Sentence-transformers model for semantic similarity.
    max_retries : int
        API call retries.

    Notes
    -----
    The paper reports that 6.3% of EU regulatory impact assessments contain
    at least one genuine causal inconsistency, and that 82% of identified
    inconsistencies are confirmed as genuine upon manual expert review.
    """

    def __init__(
        self,
        openai_api_key: str,
        similarity_threshold: float = 0.60,
        model: str = "gpt-4o",
        sbert_model: str = SimilarityIndex._DEFAULT_SBERT,
        max_retries: int = 3,
    ):
        self.openai_api_key = openai_api_key
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.max_retries = max_retries
        self._similarity_index = SimilarityIndex(sbert_model)
        self._opposition_detector = DirectionalOppositionDetector()
        self._client = self._build_client()

    def _build_client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.openai_api_key)
        except ImportError:
            logger.warning("openai not installed; LLM resolution unavailable.")
            return None

    def _llm_resolve(self, claim_a: CausalClaim, claim_b: CausalClaim) -> dict:
        """Submit a candidate pair to GPT-4o for resolution."""
        prompt = f"""Assess whether these two causal claims from the SAME policy document
are genuinely inconsistent:

CLAIM A:
  Cause: {claim_a.cause_span}
  Effect: {claim_a.effect_span}
  Mechanism: {claim_a.mechanism_span or 'not specified'}
  Context: {claim_a.text[:300] if claim_a.text else '[not provided]'}

CLAIM B:
  Cause: {claim_b.cause_span}
  Effect: {claim_b.effect_span}
  Mechanism: {claim_b.mechanism_span or 'not specified'}
  Context: {claim_b.text[:300] if claim_b.text else '[not provided]'}

Determine whether this is a genuine inconsistency or legitimate contextual variation."""

        if self._client is None:
            return {
                "is_genuine_inconsistency": True,
                "inconsistency_type": "directional",
                "resolution_rationale": "Mock resolution (openai unavailable).",
                "contextual_variation_explanation": "",
                "severity": "medium",
            }

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _RESOLUTION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
            except Exception as exc:
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))

        logger.warning(f"LLM resolution failed: {last_exc}")
        return {
            "is_genuine_inconsistency": False,
            "inconsistency_type": "none",
            "resolution_rationale": f"API failure: {last_exc}",
            "contextual_variation_explanation": "",
            "severity": "low",
        }

    @staticmethod
    def _map_type(type_str: str) -> InconsistencyType:
        mapping = {
            "directional": InconsistencyType.DIRECTIONAL,
            "magnitude": InconsistencyType.MAGNITUDE,
            "scope": InconsistencyType.SCOPE,
            "temporal": InconsistencyType.TEMPORAL,
        }
        return mapping.get(type_str.lower(), InconsistencyType.DIRECTIONAL)

    def detect(self, claims: list[CausalClaim]) -> list[InconsistencyPair]:
        """
        Detect cross-claim inconsistencies in a list of claims from a single
        document.

        Parameters
        ----------
        claims : list[CausalClaim]
            All extracted claims from one document, in document order.

        Returns
        -------
        list[InconsistencyPair]
            All identified inconsistency pairs (genuine and contextual).
            Filter with ``pair.is_genuine`` for confirmed inconsistencies.
        """
        if len(claims) < 2:
            return []

        # Step 1: Build pairwise similarity matrix
        logger.info(f"Computing similarity matrix for {len(claims)} claims...")
        sim_matrix = self._similarity_index.pairwise_matrix(claims)

        # Step 2: Identify candidate pairs
        candidates: list[tuple[int, int, float]] = []
        for i, j in itertools.combinations(range(len(claims)), 2):
            sim = sim_matrix[i][j]
            if sim >= self.similarity_threshold:
                is_opposed, _ = self._opposition_detector.are_opposed(
                    claims[i], claims[j]
                )
                if is_opposed:
                    candidates.append((i, j, sim))

        logger.info(f"Found {len(candidates)} candidate inconsistent pairs.")

        # Step 3: LLM resolution
        pairs: list[InconsistencyPair] = []
        for i, j, sim in candidates:
            resolution = self._llm_resolve(claims[i], claims[j])
            inc_type = self._map_type(resolution.get("inconsistency_type", "directional"))
            pairs.append(InconsistencyPair(
                claim_a=claims[i],
                claim_b=claims[j],
                inconsistency_type=inc_type,
                llm_resolution=resolution.get("resolution_rationale", ""),
                is_genuine=bool(resolution.get("is_genuine_inconsistency", False)),
                resolution_rationale=resolution.get(
                    "resolution_rationale",
                    resolution.get("contextual_variation_explanation", "")
                ),
                similarity_score=sim,
            ))

        return pairs

    def inconsistency_rate(self, pairs: list[InconsistencyPair]) -> dict:
        """
        Compute inconsistency rate statistics matching Table 2 of the paper.

        Parameters
        ----------
        pairs : list[InconsistencyPair]
            Output from detect().

        Returns
        -------
        dict
            Keys: ``total_candidates``, ``genuine_count``, ``genuine_rate``,
            ``by_type`` (dict mapping InconsistencyType → count).
        """
        genuine = [p for p in pairs if p.is_genuine]
        by_type = {t: 0 for t in InconsistencyType}
        for p in genuine:
            by_type[p.inconsistency_type] += 1

        return {
            "total_candidates": len(pairs),
            "genuine_count": len(genuine),
            "genuine_rate": len(genuine) / max(1, len(pairs)),
            "by_type": {t.value: count for t, count in by_type.items()},
        }
