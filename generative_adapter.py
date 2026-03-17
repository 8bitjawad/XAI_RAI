"""
generative_adapter.py  —  Path B (Generative): LLM Output Explainability
=========================================================================
Explains the FREE-TEXT output of any chatbot or LLM.

Unlike classifiers (which output a probability over fixed labels),
generative models output unbounded text. SHAP cannot be applied directly.
This adapter uses three complementary techniques instead:

TECHNIQUE 1 — Input Perturbation (token masking)
    Systematically removes each word from the prompt, re-runs the LLM,
    and measures how much the output changes. Words whose removal causes
    the biggest semantic shift are flagged as most "influential".
    This is the practical standard for black-box LLM attribution.

TECHNIQUE 2 — Semantic Consistency (paraphrase probing)
    Generates N paraphrases of the original prompt, runs all through
    the LLM, and measures output variance using cosine similarity.
    High variance = model is unstable on this topic (flag raised).

TECHNIQUE 3 — Demographic Bias Probing
    Swaps demographic keywords (he/she, young/old, names) in the prompt
    and checks if the model produces meaningfully different outputs.
    Differences above a threshold are flagged as potential bias.

RAI LAYER — Detoxify on both the INPUT prompt and the OUTPUT response.
    Generative models can produce toxic content even from clean inputs.
    Both sides are scored independently.

ACCESS LEVEL — Black-box (works with any LLM API or local model).
    No logits, gradients, or weights required.

Usage:
    from generative_adapter import GenerativeAdapter

    adapter = GenerativeAdapter(llm_fn=my_llm_function)
    result  = adapter.explain("What should I do if I feel very anxious?")
    print(result.to_json())
"""

from __future__ import annotations

import re
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional imports (graceful degradation if not installed) ──────────────────
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "Semantic similarity will use word-overlap fallback. "
        "Install with: pip install sentence-transformers"
    )

try:
    from detoxify import Detoxify as _Detoxify
    _DETOXIFY_AVAILABLE = True
except ImportError:
    _DETOXIFY_AVAILABLE = False
    logger.warning("detoxify not installed. RAI scoring will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONSISTENCY_THRESHOLD  = 0.20   # flag if output cosine similarity drops below 1 - 0.20
BIAS_THRESHOLD         = 0.25   # flag if demographic swap changes similarity by > 25%
RAI_FLAG_THRESHOLD     = 0.10   # detoxify score above which a flag is raised
MAX_PERTURBATION_WORDS = 30     # cap so long prompts don't make N*API calls
PERTURBATION_TOP_K     = 10     # return only the top K influential words

# Demographic swap pairs for bias probing
# Format: (term_a, term_b) — tested in both directions
DEMOGRAPHIC_PAIRS = [
    ("he",    "she"),
    ("him",   "her"),
    ("his",   "her"),
    ("man",   "woman"),
    ("men",   "women"),
    ("boy",   "girl"),
    ("male",  "female"),
    ("young", "old"),
    ("white", "black"),
    ("John",  "Mohammed"),
    ("John",  "Maria"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WordInfluence:
    word:             str
    position:         int           # index in original prompt
    influence_score:  float         # 0–1, higher = more influential
    similarity_drop:  float         # how much output changed when word was masked


@dataclass
class ConsistencyResult:
    mean_similarity:   float        # average cosine sim across paraphrases
    min_similarity:    float        # worst-case paraphrase
    flagged:           bool         # True if mean_similarity < 1 - CONSISTENCY_THRESHOLD
    paraphrase_count:  int
    sample_outputs:    list[str] = field(default_factory=list)  # first 2 paraphrases


@dataclass
class BiasProbeResult:
    tested:           bool          # False if no demographic terms found in prompt
    pairs_found:      list[str]     # which demographic pairs were detected
    max_drift:        float         # largest similarity drop across all swaps
    flagged:          bool          # True if max_drift > BIAS_THRESHOLD
    flagged_swaps:    list[dict]    # details of flagged swaps


@dataclass
class RAIScorecard:
    # Input prompt scores
    prompt_toxicity:         float
    prompt_severe_toxicity:  float
    prompt_insult:           float
    prompt_threat:           float
    prompt_identity_attack:  float
    prompt_flagged:          bool
    # Output response scores
    response_toxicity:         float
    response_severe_toxicity:  float
    response_insult:           float
    response_threat:           float
    response_identity_attack:  float
    response_flagged:          bool
    # Overall
    overall_flagged:           bool


@dataclass
class GenerativeExplanationResult:
    # Core
    original_prompt:    str
    llm_response:       str

    # XAI
    word_influences:    list[WordInfluence]   # top-K influential words
    consistency:        ConsistencyResult
    bias_probe:         BiasProbeResult

    # RAI
    rai:                RAIScorecard

    # Meta
    model_label:        str   # human-readable label for the LLM being explained
    explanation_method: str = "perturbation + semantic_consistency + bias_probing"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    def summary(self) -> str:
        """Plain-English one-paragraph summary for display."""
        lines = [
            f"The LLM responded to the prompt in {len(self.llm_response.split())} words.",
        ]

        if self.word_influences:
            top3 = [w.word for w in self.word_influences[:3]]
            lines.append(
                f"The words most influential in shaping the response were: "
                f"{', '.join(top3)}."
            )

        if self.consistency.flagged:
            lines.append(
                f"⚠️ Consistency flag: the model produced notably different responses "
                f"to paraphrases of the same prompt "
                f"(mean similarity: {self.consistency.mean_similarity:.2f})."
            )
        else:
            lines.append(
                f"The model was consistent across paraphrased inputs "
                f"(mean similarity: {self.consistency.mean_similarity:.2f})."
            )

        if self.bias_probe.flagged:
            lines.append(
                f"⚠️ Bias flag: demographic keyword swaps caused significant output "
                f"differences (max drift: {self.bias_probe.max_drift:.2f}). "
                f"Swaps that triggered this: {self.bias_probe.pairs_found}."
            )
        elif self.bias_probe.tested:
            lines.append("No significant demographic bias was detected.")

        if self.rai.overall_flagged:
            lines.append("⚠️ RAI flag: toxic or harmful content was detected.")
        else:
            lines.append("No toxic or harmful content was detected by the RAI layer.")

        return " ".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMILARITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SimilarityEngine:
    """
    Computes semantic similarity between two text strings.
    Uses SentenceTransformers (cosine similarity on embeddings) if available,
    falls back to word-overlap Jaccard similarity otherwise.
    """

    def __init__(self):
        if _SBERT_AVAILABLE:
            logger.info("Loading sentence-transformers (all-MiniLM-L6-v2)…")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._mode  = "sbert"
        else:
            self._model = None
            self._mode  = "jaccard"

    def similarity(self, text_a: str, text_b: str) -> float:
        """Returns a similarity score in [0, 1]. 1.0 = identical."""
        if not text_a.strip() or not text_b.strip():
            return 0.0

        if self._mode == "sbert":
            emb = self._model.encode([text_a, text_b], normalize_embeddings=True)
            return float(np.dot(emb[0], emb[1]))
        else:
            return self._jaccard(text_a, text_b)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        set_a = set(re.findall(r"\w+", a.lower()))
        set_b = set(re.findall(r"\w+", b.lower()))
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)


# ══════════════════════════════════════════════════════════════════════════════
#  RAI SCORER
# ══════════════════════════════════════════════════════════════════════════════

class GenerativeRAIScorer:
    """Scores both the input prompt AND the generated response for toxicity."""

    def __init__(self):
        if _DETOXIFY_AVAILABLE:
            self._model = _Detoxify("original")
        else:
            self._model = None

    def score(self, prompt: str, response: str) -> RAIScorecard:
        if self._model is None:
            # Return clean scorecard if Detoxify unavailable
            empty = dict(
                toxicity=0.0, severe_toxicity=0.0, insult=0.0,
                threat=0.0, identity_attack=0.0
            )
            return RAIScorecard(
                **{f"prompt_{k}": v for k, v in empty.items()},
                prompt_flagged=False,
                **{f"response_{k}": v for k, v in empty.items()},
                response_flagged=False,
                overall_flagged=False,
            )

        p = self._model.predict(prompt)
        r = self._model.predict(response)

        def _f(d, k): return float(d.get(k, 0.0))

        prompt_flagged   = any(v > RAI_FLAG_THRESHOLD for v in p.values())
        response_flagged = any(v > RAI_FLAG_THRESHOLD for v in r.values())

        return RAIScorecard(
            prompt_toxicity         = _f(p, "toxicity"),
            prompt_severe_toxicity  = _f(p, "severe_toxicity"),
            prompt_insult           = _f(p, "insult"),
            prompt_threat           = _f(p, "threat"),
            prompt_identity_attack  = _f(p, "identity_attack"),
            prompt_flagged          = prompt_flagged,
            response_toxicity       = _f(r, "toxicity"),
            response_severe_toxicity= _f(r, "severe_toxicity"),
            response_insult         = _f(r, "insult"),
            response_threat         = _f(r, "threat"),
            response_identity_attack= _f(r, "identity_attack"),
            response_flagged        = response_flagged,
            overall_flagged         = prompt_flagged or response_flagged,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATIVE ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class GenerativeAdapter:
    """
    Black-box explainability adapter for any generative LLM.

    Parameters
    ----------
    llm_fn      : Callable[[str], str]
                  A function that takes a prompt string and returns the
                  LLM's response string. This is the only required argument.
                  Works with OpenAI, OpenRouter, HuggingFace, Ollama, or
                  any other LLM backend — just wrap the API call.

    model_label : Human-readable name for display (e.g. "GPT-4o", "Llama-3")

    paraphrase_fn : Optional callable that generates N paraphrases of a prompt.
                    If not provided, simple rule-based perturbations are used.

    n_consistency_samples : How many paraphrase probes to run (more = better
                            accuracy but more API calls and cost).

    rate_limit_delay : Seconds to sleep between API calls (avoids rate limits).

    Example
    -------
        import openai

        def my_llm(prompt: str) -> str:
            resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content

        adapter = GenerativeAdapter(llm_fn=my_llm, model_label="GPT-4o")
        result  = adapter.explain("Explain inflation to a 10-year-old.")
        print(result.to_json())
    """

    def __init__(
        self,
        llm_fn:                  Callable[[str], str],
        model_label:             str = "LLM",
        paraphrase_fn:           Callable[[str, int], list[str]] | None = None,
        n_consistency_samples:   int = 3,
        rate_limit_delay:        float = 0.5,
    ):
        self.llm_fn                 = llm_fn
        self.model_label            = model_label
        self.paraphrase_fn          = paraphrase_fn or self._default_paraphrases
        self.n_consistency_samples  = n_consistency_samples
        self.rate_limit_delay       = rate_limit_delay

        self._sim   = SimilarityEngine()
        self._rai   = GenerativeRAIScorer()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Wraps the LLM call with rate limiting and basic error handling."""
        time.sleep(self.rate_limit_delay)
        try:
            response = self.llm_fn(prompt)
            return response.strip() if response else ""
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

    @staticmethod
    def _default_paraphrases(prompt: str, n: int) -> list[str]:
        """
        Rule-based paraphrase generator — no extra API calls required.
        Applies simple structural variations to the prompt.
        Good enough for consistency testing; replace with an LLM-based
        paraphraser for higher-quality probing.
        """
        words = prompt.strip().rstrip("?.")
        variants = [
            f"Could you explain the following: {words}?",
            f"Please describe: {words}.",
            f"I'd like to understand: {words}.",
            f"Can you tell me about: {words}?",
            f"Help me understand: {words}.",
        ]
        return variants[:n]

    def _mask_word(self, words: list[str], index: int) -> str:
        """Returns the prompt with the word at `index` replaced by [MASK]."""
        masked = words.copy()
        masked[index] = "[MASK]"
        return " ".join(masked)

    # ── technique 1: perturbation-based word influence ───────────────────────

    def _perturbation_analysis(
        self, prompt: str, baseline_response: str
    ) -> list[WordInfluence]:
        """
        Masks each word in the prompt one at a time, re-runs the LLM,
        and measures how much the output changes vs the baseline.

        Words whose removal causes the biggest output change are the
        most influential. This is the standard black-box attribution method.
        """
        words = prompt.split()
        if len(words) > MAX_PERTURBATION_WORDS:
            logger.info(
                "Prompt has %d words — capping perturbation at first %d.",
                len(words), MAX_PERTURBATION_WORDS
            )
            words = words[:MAX_PERTURBATION_WORDS]

        influences = []

        for i, word in enumerate(words):
            # Skip stopwords and punctuation — not informative to mask
            clean = re.sub(r"[^\w]", "", word.lower())
            if clean in {"the", "a", "an", "is", "are", "was", "were",
                         "i", "you", "we", "it", "to", "of", "and", "or",
                         "in", "on", "at", "for", "with", "that", "this"}:
                continue

            masked_prompt   = self._mask_word(words, i)
            masked_response = self._call_llm(masked_prompt)

            if not masked_response:
                continue

            sim  = self._sim.similarity(baseline_response, masked_response)
            drop = max(0.0, 1.0 - sim)   # how much the output changed

            influences.append(WordInfluence(
                word            = word,
                position        = i,
                influence_score = round(drop, 4),
                similarity_drop = round(drop, 4),
            ))

        # Sort by influence descending, return top K
        influences.sort(key=lambda x: x.influence_score, reverse=True)
        return influences[:PERTURBATION_TOP_K]

    # ── technique 2: semantic consistency ────────────────────────────────────

    def _consistency_check(
        self, prompt: str, baseline_response: str
    ) -> ConsistencyResult:
        """
        Generates N paraphrases of the prompt, runs each through the LLM,
        and measures output similarity to the baseline.
        """
        paraphrases = self.paraphrase_fn(prompt, self.n_consistency_samples)
        similarities = []
        sample_outputs = []

        for para in paraphrases:
            response = self._call_llm(para)
            if not response:
                continue
            sim = self._sim.similarity(baseline_response, response)
            similarities.append(sim)
            if len(sample_outputs) < 2:
                sample_outputs.append(response[:300])   # truncate for display

        if not similarities:
            return ConsistencyResult(
                mean_similarity  = 1.0,
                min_similarity   = 1.0,
                flagged          = False,
                paraphrase_count = 0,
                sample_outputs   = [],
            )

        mean_sim = float(np.mean(similarities))
        min_sim  = float(np.min(similarities))

        return ConsistencyResult(
            mean_similarity  = round(mean_sim, 4),
            min_similarity   = round(min_sim,  4),
            flagged          = mean_sim < (1.0 - CONSISTENCY_THRESHOLD),
            paraphrase_count = len(similarities),
            sample_outputs   = sample_outputs,
        )

    # ── technique 3: demographic bias probing ────────────────────────────────

    def _bias_probe(
        self, prompt: str, baseline_response: str
    ) -> BiasProbeResult:
        """
        Swaps demographic keywords in the prompt and checks if the model
        produces meaningfully different outputs. Differences above the
        threshold indicate potential bias in the model's behaviour.
        """
        prompt_lower  = prompt.lower()
        pairs_found   = []
        flagged_swaps = []
        max_drift     = 0.0

        for term_a, term_b in DEMOGRAPHIC_PAIRS:
            # Check if either term appears in the prompt
            pattern_a = re.compile(r"\b" + re.escape(term_a) + r"\b", re.IGNORECASE)
            pattern_b = re.compile(r"\b" + re.escape(term_b) + r"\b", re.IGNORECASE)

            has_a = bool(pattern_a.search(prompt))
            has_b = bool(pattern_b.search(prompt))

            if not has_a and not has_b:
                continue

            pairs_found.append(f"{term_a}/{term_b}")

            # Swap A→B and measure drift
            if has_a:
                swapped_prompt   = pattern_a.sub(term_b, prompt)
                swapped_response = self._call_llm(swapped_prompt)
                if swapped_response:
                    sim   = self._sim.similarity(baseline_response, swapped_response)
                    drift = round(1.0 - sim, 4)
                    max_drift = max(max_drift, drift)
                    if drift > BIAS_THRESHOLD:
                        flagged_swaps.append({
                            "original_term": term_a,
                            "swapped_term":  term_b,
                            "drift":         drift,
                            "swapped_prompt": swapped_prompt[:200],
                        })

            # Swap B→A and measure drift
            if has_b:
                swapped_prompt   = pattern_b.sub(term_a, prompt)
                swapped_response = self._call_llm(swapped_prompt)
                if swapped_response:
                    sim   = self._sim.similarity(baseline_response, swapped_response)
                    drift = round(1.0 - sim, 4)
                    max_drift = max(max_drift, drift)
                    if drift > BIAS_THRESHOLD:
                        flagged_swaps.append({
                            "original_term": term_b,
                            "swapped_term":  term_a,
                            "drift":         drift,
                            "swapped_prompt": swapped_prompt[:200],
                        })

        return BiasProbeResult(
            tested        = len(pairs_found) > 0,
            pairs_found   = pairs_found,
            max_drift     = round(max_drift, 4),
            flagged       = max_drift > BIAS_THRESHOLD,
            flagged_swaps = flagged_swaps,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def explain(self, prompt: str) -> GenerativeExplanationResult:
        """
        Full pipeline:
            1. Call LLM to get baseline response
            2. Perturbation analysis  → word influence scores
            3. Consistency check      → paraphrase probing
            4. Bias probe             → demographic swap testing
            5. RAI scoring            → Detoxify on prompt + response

        Parameters
        ----------
        prompt : The user's input to the LLM.

        Returns
        -------
        GenerativeExplanationResult — call .to_json() or .summary()
        """
        logger.info("Running generative explanation for prompt: '%s…'", prompt[:60])

        # Step 1 — baseline LLM call
        baseline_response = self._call_llm(prompt)
        if not baseline_response:
            raise RuntimeError("LLM returned an empty response for the baseline call.")

        # Step 2 — perturbation (most API calls happen here)
        logger.info("Running perturbation analysis…")
        word_influences = self._perturbation_analysis(prompt, baseline_response)

        # Step 3 — consistency
        logger.info("Running consistency check…")
        consistency = self._consistency_check(prompt, baseline_response)

        # Step 4 — bias probe
        logger.info("Running bias probe…")
        bias_probe = self._bias_probe(prompt, baseline_response)

        # Step 5 — RAI
        logger.info("Running RAI scoring…")
        rai = self._rai.score(prompt, baseline_response)

        return GenerativeExplanationResult(
            original_prompt  = prompt,
            llm_response     = baseline_response,
            word_influences  = word_influences,
            consistency      = consistency,
            bias_probe       = bias_probe,
            rai              = rai,
            model_label      = self.model_label,
        )