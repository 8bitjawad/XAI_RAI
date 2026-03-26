"""
xai_layer.py  ─  Universal XAI / RAI Middleware
================================================
Single file.  No framework dependencies.

PATH A  (Tabular)  ─ sklearn / tree models
    wrap(model, X_train, sensitive_column="col")

PATH B  (Text/LLM) ─ HuggingFace classifiers
    wrap("distilbert-base-uncased-finetuned-sst-2-english")

Both paths expose the same interface:
    adapter = wrap(...)
    result  = adapter.predict(sample)
"""

from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONSISTENCY_THRESHOLD = 0.20   # flag if confidence swings > 20 %
RAI_FLAG_THRESHOLD    = 0.10   # flag toxicity if any score   > 10 %
MAX_SHAP_TOKENS       = 512

SYNONYM_SWAP = {
    "good": "great",     "great": "excellent", "bad": "terrible",
    "terrible": "awful", "happy": "glad",      "sad": "unhappy",
    "love": "adore",     "hate": "detest",     "movie": "film",
    "film": "movie",     "fast": "quick",       "quick": "fast",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PATH A  ─  TABULAR  (your existing code, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

import shap
from lime.lime_tabular import LimeTabularExplainer
from fairlearn.metrics import demographic_parity_difference


class ExplainEngine:

    def __init__(self, model, X_train):
        self.model   = model
        self.X_train = X_train

        self.shap_explainer = shap.TreeExplainer(model)

        self.lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=["Down", "Up"],
            mode="classification",
        )

    def shap_explanation(self, sample):
        shap_values = self.shap_explainer(sample)
        values      = np.abs(shap_values.values[0])

        clean_values = {}
        for feature, val in zip(self.X_train.columns, values):
            if isinstance(val, np.ndarray):
                val = float(val[0])
            clean_values[feature] = float(val)

        return clean_values

    def shap_plot(self, sample):
        import matplotlib.pyplot as plt

        shap_values  = self.shap_explainer(sample)
        values       = shap_values.values[0][:, 0]
        feature_names = self.X_train.columns

        shap_series = pd.Series(values, index=feature_names)
        shap_series.abs().sort_values().plot.barh()
        plt.title("Feature Impact on Prediction")
        plt.xlabel("Impact Strength")
        return plt

    def lime_explanation(self, sample):
        exp = self.lime_explainer.explain_instance(
            sample.values[0],
            lambda x: self.model.predict_proba(
                pd.DataFrame(x, columns=self.X_train.columns)
            ),
            num_features=5,
        )
        return exp.as_list()


class ReasonEngine:

    def __init__(self, sensitive_column):
        self.sensitive_column = sensitive_column

    def fairness_check(self, X_test, y_test, y_pred):
        if self.sensitive_column is None:
            return {"fairness_check": "No sensitive feature provided"}

        sensitive_feature = X_test[self.sensitive_column]
        dp_diff  = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_feature,
        )
        bias_flag = abs(dp_diff) > 0.1
        return {
            "demographic_parity_difference": dp_diff,
            "bias_flag": bias_flag,
        }


class RAIWrapper:
    """Path A adapter — tabular models."""

    def __init__(self, model, X_train, sensitive_column):
        self.model         = model
        self.explain_engine = ExplainEngine(model, X_train)
        self.reason_engine  = ReasonEngine(sensitive_column)

    def predict(self, sample):
        prediction  = self.model.predict(sample)[0]
        confidence  = self.model.predict_proba(sample)[0].max()

        shap_values = self.explain_engine.shap_explanation(sample)
        lime_rules  = self.explain_engine.lime_explanation(sample)

        ranked_features = sorted(
            shap_values.items(), key=lambda x: x[1], reverse=True
        )
        top_drivers = ranked_features[:3]

        return {
            "prediction":         prediction,
            "prediction_numeric": int(prediction),
            "confidence":         float(confidence),
            "shap_values":        shap_values,
            "top_factors":        top_drivers,
            "lime_rules":         lime_rules,
        }

    def fairness_report(self, X_test_full, y_test):
        X_test_model = X_test_full.drop(
            columns=[self.reason_engine.sensitive_column]
        )
        y_pred = self.model.predict(X_test_model)
        result = self.reason_engine.fairness_check(X_test_full, y_test, y_pred)

        dp   = float(result["demographic_parity_difference"])
        bias = result["bias_flag"]

        return {
            "Protected Attribute":      self.reason_engine.sensitive_column,
            "Prediction Difference (%)": round(dp * 100, 2),
            "Bias Detected":            "Yes" if bias else "No significant bias found",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PATH B  ─  TEXT / LLM
# ══════════════════════════════════════════════════════════════════════════════

# ── B.1  Data contracts ───────────────────────────────────────────────────────

@dataclass
class Prediction:
    label:      str
    score:      float
    raw_logits: list[float] = field(default_factory=list)


@dataclass
class ConsistencyCheck:
    original_score: float
    variant_score:  float
    delta:          float
    flagged:        bool
    variant_text:   str = ""


@dataclass
class ResponsibilityScorecard:
    toxicity:        float
    severe_toxicity: float
    obscene:         float
    insult:          float
    threat:          float
    identity_attack: float
    bias_flagged:    bool


@dataclass
class TextExplanationResult:
    prediction:      Prediction
    word_importance: dict[str, float]
    responsibility:  ResponsibilityScorecard
    consistency:     ConsistencyCheck
    model_name:      str
    input_text:      str

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


# ── B.2  RAI scorer (Detoxify) ────────────────────────────────────────────────

class RAIScorer:

    def __init__(self, model_type: str = "original"):
        from detoxify import Detoxify
        logger.info("Loading Detoxify (%s)…", model_type)
        self._model = Detoxify(model_type)

    def score(self, text: str) -> ResponsibilityScorecard:
        raw = self._model.predict(text)

        def _f(k):
            return float(raw.get(k, 0.0))

        return ResponsibilityScorecard(
            toxicity        = _f("toxicity"),
            severe_toxicity = _f("severe_toxicity"),
            obscene         = _f("obscene"),
            insult          = _f("insult"),
            threat          = _f("threat"),
            identity_attack = _f("identity_attack"),
            bias_flagged    = any(v > RAI_FLAG_THRESHOLD for v in raw.values()),
        )


# ── B.3  TextAdapter ──────────────────────────────────────────────────────────

class TextAdapter:
    """
    Path B adapter — HuggingFace text classifiers.

    Grey-box: accesses raw model logits (not just the pipeline label string)
    so SHAP gets meaningful gradient-depth signal.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased-finetuned-sst-2-english",
        rai_scorer: RAIScorer | None = None,
        device: str | None = None,
    ):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )

        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        logger.info("Loading %s on %s…", model_name_or_path, self.device)

        # Tokenizer — ensures SHAP token alignment with actual words
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Model loaded directly for logit access (grey-box)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(self.device)
        self.model.eval()
        self.label_map: dict[int, str] = self.model.config.id2label

        # HuggingFace pipeline — used by SHAP's masker
        self._pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            top_k=None,         # replaces deprecated return_all_scores=True
        )

        # SHAP Text explainer — partition strategy for NLP
        self._masker    = shap.maskers.Text(tokenizer=self.tokenizer)
        self._explainer = shap.Explainer(self._pipeline, self._masker)

        self.rai = rai_scorer or RAIScorer()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """Input Adapter: clean + truncate before any model call."""
        text   = text.strip()
        text   = re.sub(r"\s+", " ", text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > MAX_SHAP_TOKENS:
            tokens = tokens[:MAX_SHAP_TOKENS]
            text   = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def _raw_predict(self, text: str) -> tuple[Prediction, list[dict]]:
        """Grey-box forward pass — returns logits alongside the label."""
        torch = self._torch
        enc   = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SHAP_TOKENS + 2,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)

        logits = outputs.logits[0]
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_id = int(np.argmax(probs))

        prediction = Prediction(
            label      = self.label_map[pred_id],
            score      = float(probs[pred_id]),
            raw_logits = logits.cpu().tolist(),
        )
        all_scores = [
            {"label": self.label_map[i], "score": float(probs[i])}
            for i in range(len(probs))
        ]
        return prediction, all_scores

    def _explain(self, text: str) -> dict[str, float]:
        """SHAP token-level attribution map for the predicted class."""
        shap_values   = self._explainer([text], fixed_context=1)
        tokens        = shap_values.data[0]
        _, all_scores = self._raw_predict(text)
        top_idx       = int(np.argmax([s["score"] for s in all_scores]))
        token_shap    = shap_values.values[0][:, top_idx]

        importance: dict[str, float] = {}
        for tok, val in zip(tokens, token_shap):
            tok = tok.strip()
            if tok:
                importance[tok] = importance.get(tok, 0.0) + float(val)

        return dict(
            sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    def _consistency_check(self, text: str, original_score: float) -> ConsistencyCheck:
        """Swap one synonym, re-run, flag if confidence swings > 20 %."""
        words  = text.split()
        variant = words.copy()
        swapped = False

        for i, word in enumerate(words):
            lower = word.lower().strip(".,!?;:")
            if lower in SYNONYM_SWAP:
                rep = SYNONYM_SWAP[lower]
                variant[i] = rep.capitalize() if word[0].isupper() else rep
                swapped = True
                break

        variant_text = " ".join(variant) if swapped else text + " really"
        var_pred, _  = self._raw_predict(self._normalise(variant_text))
        delta        = abs(original_score - var_pred.score)

        return ConsistencyCheck(
            original_score = original_score,
            variant_score  = var_pred.score,
            delta          = round(delta, 4),
            flagged        = delta > CONSISTENCY_THRESHOLD,
            variant_text   = variant_text,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, raw_text: str) -> TextExplanationResult:
        text            = self._normalise(raw_text)
        prediction, _   = self._raw_predict(text)
        word_importance = self._explain(text)
        responsibility  = self.rai.score(text)
        consistency     = self._consistency_check(text, prediction.score)

        return TextExplanationResult(
            prediction      = prediction,
            word_importance = word_importance,
            responsibility  = responsibility,
            consistency     = consistency,
            model_name      = self.model_name,
            input_text      = text,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED wrap()  ─  single entry-point for both paths
# ══════════════════════════════════════════════════════════════════════════════

_text_adapter_cache: dict[str, TextAdapter] = {}


def wrap(model: Any, X_train=None, sensitive_column=None, **kwargs) -> Any:
    """
    Universal middleware factory.

    PATH A — tabular
        adapter = wrap(sklearn_model, X_train=df, sensitive_column="col")
        result  = adapter.predict(sample_df)
        report  = adapter.fairness_report(X_test_full, y_test)

    PATH B — text / LLM
        adapter = wrap("distilbert-base-uncased-finetuned-sst-2-english")
        result  = adapter.predict("This film was fantastic!")
        print(result.to_json())
    """
    if isinstance(model, str):
        # ── Path B: model is a HuggingFace model-id string ───────────────────
        cache_key = model + str(kwargs.get("device", ""))
        if cache_key not in _text_adapter_cache:
            _text_adapter_cache[cache_key] = TextAdapter(
                model_name_or_path=model, **kwargs
            )
        return _text_adapter_cache[cache_key]

    else:
        # ── Path A: model is a fitted sklearn / tree object ───────────────────
        if X_train is None:
            raise ValueError("Path A requires X_train. Usage: wrap(model, X_train=df)")
        return RAIWrapper(
            model=model,
            X_train=X_train,
            sensitive_column=sensitive_column,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST  —  python xai_layer.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    DEMO_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    sentences = [
        "This movie was absolutely fantastic — best film of the year!",
        "I hated every single minute of this terrible, awful film.",
        "It was okay, nothing special but not bad either.",
    ]

    adapter = wrap(DEMO_MODEL)

    for text in sentences:
        print("\n" + "═" * 68)
        print(f"INPUT : {text}")
        result = adapter.predict(text)
        print(result.to_json())