"""
app.py  —  Universal XAI + Responsible AI Layer
================================================
Tab A  →  Model-Agnostic Tabular Demo  (finance / healthcare / real estate)
Tab B  →  Text / LLM  (sentiment + toxicity)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from core import wrap
from llm_engine import generate_explanation

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RAI_FLAG_THRESHOLD = 0.10

# ══════════════════════════════════════════════════════════════════════════════
#  TAB A — domain configs
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_CONFIG = {

    "💳 Finance — Credit Risk": {
        "domain_key":   "finance",
        "description":  "Predicts whether a loan applicant will **default** (1) or **repay** (0).",
        "sensitive_col": "gender",
        "features":     ["credit_score", "annual_income", "debt_to_income",
                         "loan_amount", "employment_years", "missed_payments"],
        "feature_descriptions": {
            "credit_score":     "300 – 850",
            "annual_income":    "USD  e.g. 55000",
            "debt_to_income":   "ratio 0.0 – 1.0  e.g. 0.35",
            "loan_amount":      "USD  e.g. 15000",
            "employment_years": "years at current job",
            "missed_payments":  "count in last 24 months",
        },
        "defaults":   [650, 55000, 0.35, 15000, 4, 1],
        "label_map":  {0: "✅ Will Repay", 1: "⚠️ Likely to Default"},
        "model_cls":  RandomForestClassifier,
        "model_kwargs": {"n_estimators": 50, "random_state": 42},
        "synth": lambda n: pd.DataFrame({
            "credit_score":     np.random.randint(300, 851, n),
            "annual_income":    np.random.randint(20000, 150000, n),
            "debt_to_income":   np.round(np.random.uniform(0.05, 0.95, n), 2),
            "loan_amount":      np.random.randint(1000, 50000, n),
            "employment_years": np.random.randint(0, 30, n),
            "missed_payments":  np.random.randint(0, 10, n),
            "gender":           np.random.choice(["M", "F"], n),
        }),
        "target": lambda df: (
            (df["missed_payments"] > 3).astype(int) |
            (df["debt_to_income"]  > 0.6).astype(int) |
            (df["credit_score"]    < 580).astype(int)
        ).clip(0, 1),
    },

    "🏥 Healthcare — Diabetes Risk": {
        "domain_key":   "healthcare",
        "description":  "Predicts whether a patient is **at risk of diabetes** (1) or not (0).",
        "sensitive_col": "age_group",
        "features":     ["glucose_level", "bmi", "blood_pressure",
                         "insulin", "skin_thickness", "pregnancies"],
        "feature_descriptions": {
            "glucose_level":  "mg/dL  e.g. 120",
            "bmi":            "body mass index  e.g. 28.5",
            "blood_pressure": "mmHg diastolic  e.g. 80",
            "insulin":        "μU/mL  e.g. 85",
            "skin_thickness": "mm triceps fold  e.g. 20",
            "pregnancies":    "count  e.g. 2",
        },
        "defaults":  [120, 28.5, 80, 85, 20, 2],
        "label_map": {0: "✅ Low Risk", 1: "⚠️ Elevated Diabetes Risk"},
        "model_cls": GradientBoostingClassifier,
        "model_kwargs": {"n_estimators": 50, "random_state": 42},
        "synth": lambda n: pd.DataFrame({
            "glucose_level":  np.random.randint(60, 200, n),
            "bmi":            np.round(np.random.uniform(18, 50, n), 1),
            "blood_pressure": np.random.randint(40, 120, n),
            "insulin":        np.random.randint(0, 400, n),
            "skin_thickness": np.random.randint(0, 60, n),
            "pregnancies":    np.random.randint(0, 15, n),
            "age_group":      np.random.choice(["<40", "40-60", ">60"], n),
        }),
        "target": lambda df: (
            (df["glucose_level"] > 140).astype(int) |
            (df["bmi"]           > 35 ).astype(int)
        ).clip(0, 1),
    },

    "🏠 Real Estate — Price Direction": {
        "domain_key":   "realestate",
        "description":  "Predicts whether a property price will go **up** (1) or **down** (0) next quarter.",
        "sensitive_col": "neighbourhood_type",
        "features":     ["sq_footage", "bedrooms", "bathrooms",
                         "age_years", "proximity_to_transit", "school_rating"],
        "feature_descriptions": {
            "sq_footage":           "total area sq ft  e.g. 1800",
            "bedrooms":             "count  e.g. 3",
            "bathrooms":            "count  e.g. 2",
            "age_years":            "years since built  e.g. 15",
            "proximity_to_transit": "minutes walk  e.g. 10",
            "school_rating":        "1–10  e.g. 7",
        },
        "defaults":  [1800, 3, 2, 15, 10, 7],
        "label_map": {0: "📉 Price likely to fall", 1: "📈 Price likely to rise"},
        "model_cls": DecisionTreeClassifier,
        "model_kwargs": {"max_depth": 6, "random_state": 42},
        "synth": lambda n: pd.DataFrame({
            "sq_footage":           np.random.randint(500, 5000, n),
            "bedrooms":             np.random.randint(1, 7, n),
            "bathrooms":            np.random.randint(1, 5, n),
            "age_years":            np.random.randint(0, 80, n),
            "proximity_to_transit": np.random.randint(1, 60, n),
            "school_rating":        np.random.randint(1, 11, n),
            "neighbourhood_type":   np.random.choice(["Urban", "Suburban", "Rural"], n),
        }),
        "target": lambda df: (
            (df["school_rating"]        >= 7).astype(int) &
            (df["proximity_to_transit"] <= 15).astype(int)
        ).clip(0, 1),
    },
}


@st.cache_resource
def build_domain_adapter(domain_name: str):
    cfg      = DOMAIN_CONFIG[domain_name]
    df       = cfg["synth"](500)
    features = cfg["features"]
    X        = df[features]
    y        = cfg["target"](df)

    clf = cfg["model_cls"](**cfg["model_kwargs"])
    clf.fit(X, y)

    adapter = wrap(clf, X_train=X, sensitive_column=cfg["sensitive_col"])
    return adapter, df, y, features


# ══════════════════════════════════════════════════════════════════════════════
#  TAB B — text model registry
#  To add a third model: drop a new entry here. Nothing else changes.
# ══════════════════════════════════════════════════════════════════════════════

TEXT_MODEL_CONFIG = {

    "🎭 Sentiment Analysis — DistilBERT SST-2": {
        "model_id":   "distilbert-base-uncased-finetuned-sst-2-english",
        "domain_key": "text",
        "description": (
            "General-purpose **sentiment classifier** fine-tuned on movie reviews. "
            "Predicts POSITIVE or NEGATIVE and explains which words drove the verdict."
        ),
        "what_to_watch": (
            "Watch how clearly positive words (fantastic, great) push toward POSITIVE "
            "while negative words (terrible, hated) push toward NEGATIVE. "
            "Detoxify runs as a fully independent second audit layer beneath the prediction."
        ),
        "examples": [
            "Select an example…",
            "This market rally was absolutely fantastic — best quarter of the year!",
            "I hated every single minute of this terrible earnings call.",
            "The trade was okay, nothing special but not a total disaster.",
            "You are completely useless and I hate everything about this.",
        ],
    },

    "☣️ Toxic Content Classifier — Unitary ToxicBERT": {
        "model_id":   "unitary/toxic-bert",
        "domain_key": "text",
        "description": (
            "**Toxic content classifier** fine-tuned specifically on toxic language datasets. "
            "Predicts TOXIC / NON_TOXIC. Runs alongside Detoxify — "
            "two fully independent systems auditing the same input from different angles."
        ),
        "what_to_watch": (
            "This is the architectural demo point: the **classifier** predicts toxicity "
            "as a supervised task, while **Detoxify** scores it as a separate RAI audit. "
            "When both agree, the signal is doubly validated. "
            "When they disagree, it surfaces an edge case — exactly what a responsible AI "
            "layer should catch and flag for human review."
        ),
        "examples": [
            "Select an example…",
            "I hope you have a wonderful day, stay safe everyone.",
            "You are a complete idiot and should be ashamed of yourself.",
            "The quarterly results were disappointing but we remain optimistic.",
            "I will find you and make you regret everything you have ever done.",
            "This policy is misguided but I respect the effort behind it.",
        ],
    },
}


@st.cache_resource
def load_text_adapter(model_id: str):
    """Cached per model_id — switching models never reloads the one already in memory."""
    return wrap(model_id)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPER — renders the full XAI result for any text model
#  Keeping this as one function means Tab B never duplicates rendering code
#  no matter how many models are added to TEXT_MODEL_CONFIG.
# ══════════════════════════════════════════════════════════════════════════════

def render_text_result(result, raw_input: str, domain_key: str):

    # ── prediction ────────────────────────────────────────────────────────────
    st.subheader("Prediction")
    label  = result.prediction.label
    score  = result.prediction.score
    colour = "green" if "NON" in label or label == "POSITIVE" else "red"
    st.markdown(f"**:{colour}[{label}]** — confidence `{score:.1%}`")
    st.caption(f"Raw logits: {[round(l, 3) for l in result.prediction.raw_logits]}")

    # ── SHAP word importance ──────────────────────────────────────────────────
    st.subheader("Word Importance (SHAP)")
    importance = result.word_importance
    if importance:
        tokens  = list(importance.keys())[:10]
        values  = [importance[t] for t in tokens]
        colours = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
        fig, ax = plt.subplots(figsize=(7, max(3, len(tokens) * 0.45)))
        ax.barh(tokens[::-1], values[::-1], color=colours[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value  (+ pushes toward predicted label)")
        ax.set_title("Token Attribution Map")
        st.pyplot(fig)
        plt.close(fig)

    # ── RAI scorecard ─────────────────────────────────────────────────────────
    st.subheader("Responsibility Scorecard  *(Detoxify — independent audit)*")
    r = result.responsibility
    rai_scores = {
        "Toxicity":        r.toxicity,
        "Severe Toxicity": r.severe_toxicity,
        "Obscene":         r.obscene,
        "Insult":          r.insult,
        "Threat":          r.threat,
        "Identity Attack": r.identity_attack,
    }
    rai_df = pd.DataFrame(rai_scores.items(), columns=["Category", "Score"])

    def _highlight(row):
        return [
            "background-color: #fde8e8"
            if row["Score"] > RAI_FLAG_THRESHOLD else ""
            for _ in row
        ]

    st.dataframe(
        rai_df.style.apply(_highlight, axis=1).format({"Score": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    flag_colour = "red" if r.bias_flagged else "green"
    st.markdown(
        f"**Detoxify Flag: :{flag_colour}"
        f"[{'⚠ FLAGGED' if r.bias_flagged else '✓ CLEAN'}]**"
    )

    # ── consistency check ─────────────────────────────────────────────────────
    st.subheader("Consistency Check")
    c = result.consistency
    col1, col2, col3 = st.columns(3)
    col1.metric("Original confidence", f"{c.original_score:.1%}")
    col2.metric("Variant confidence",  f"{c.variant_score:.1%}",
                delta=f"{c.delta:+.1%}")
    col3.metric("Flagged", "⚠ Yes" if c.flagged else "✓ No",
                delta_color="inverse")
    with st.expander("Variant text used"):
        st.write(c.variant_text)

    # ── LLM explanation ───────────────────────────────────────────────────────
    st.subheader("AI Explanation")
    text_result_for_llm = {
        "prediction":  label,
        "confidence":  score,
        "top_factors": list(importance.items())[:5],
        "shap_values": importance,
        "lime_rules":  [],
    }
    with st.spinner("Generating explanation…"):
        explanation = generate_explanation(
            text_result_for_llm,
            domain=domain_key,
            sample_values={"input_text": raw_input},
        )
    st.markdown(explanation)

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("Universal XAI + Responsible AI Layer")
st.write("Demo: Model Prediction with Explanation and Fairness Checks")

tab_a, tab_b = st.tabs([
    "📊 Path A — Model-Agnostic Tabular",
    "💬 Path B — Text / LLM",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB A  —  model-agnostic tabular (formerly Tab C, now the main tab)
# ══════════════════════════════════════════════════════════════════════════════

with tab_a:
    st.header("Model-Agnostic XAI Demo")
    st.markdown(
        "Select a domain. The same `wrap(model)` call runs on a different "
        "algorithm each time — the XAI output structure never changes."
    )

    domain = st.selectbox("Select a domain", list(DOMAIN_CONFIG.keys()),
                          key="domain_select")
    cfg        = DOMAIN_CONFIG[domain]
    domain_key = cfg["domain_key"]
    model_name = cfg["model_cls"].__name__

    st.info(cfg["description"])
    st.caption(f"Algorithm: **`{model_name}`**")

    with st.spinner(f"Training `{model_name}` on synthetic data…"):
        adapter, df_full, y_synth, features = build_domain_adapter(domain)

    # ── feature inputs ────────────────────────────────────────────────────────
    st.subheader("Input Features")
    input_vals  = []
    left, right = st.columns(2)

    for i, feat in enumerate(features):
        col = left if i % 2 == 0 else right
        val = col.number_input(
            feat,
            value=float(cfg["defaults"][i]),
            help=cfg["feature_descriptions"][feat],
            key=f"a_{domain}_{feat}",
        )
        input_vals.append(val)

    if st.button("Run Prediction", key="domain_run"):
        sample = pd.DataFrame([input_vals], columns=features)

        # dict of raw input values — passed to LLM so it can reference numbers
        sample_values = dict(zip(features, input_vals))

        with st.spinner("Running XAI layer…"):
            result = adapter.predict(sample)

        # ── prediction ────────────────────────────────────────────────────────
        st.subheader("Prediction")
        pred_int   = int(result["prediction"])
        pred_label = cfg["label_map"][pred_int]
        colour     = "green" if pred_int == 0 else "red"

        st.markdown(f"### :{colour}[{pred_label}]")
        st.metric("Model confidence", f"{result['confidence']:.1%}")

        # ── SHAP bar chart ────────────────────────────────────────────────────
        st.subheader("What drove this prediction? (SHAP)")

        shap_df = (
            pd.DataFrame(result["shap_values"].items(),
                         columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=True)
        )
        bar_colours = ["#2ecc71" if v >= 0 else "#e74c3c"
                       for v in shap_df["Importance"]]
        fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.55)))
        ax.barh(shap_df["Feature"], shap_df["Importance"], color=bar_colours)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Attribution  ·  {model_name}")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Top 3 drivers:**")
        for feat, val in result["top_factors"]:
            direction = "increased" if val > 0 else "reduced"
            st.write(f"- **{feat}** {direction} the score by `{round(val, 4)}`")

        with st.expander("LIME decision rules"):
            for rule, weight in result["lime_rules"]:
                st.write(f"`{rule}` → weight `{round(weight, 4)}`")

        # ── fairness report ───────────────────────────────────────────────────
        st.subheader("Responsible AI — Fairness Report")

        sensitive_col     = cfg["sensitive_col"]
        fairness_input_df = df_full[[sensitive_col] + features].copy()
        fairness          = adapter.fairness_report(fairness_input_df, y_synth)

        f1, f2 = st.columns(2)
        f1.metric("Protected Attribute",   fairness["Protected Attribute"])
        f2.metric("Prediction Difference", f"{fairness['Prediction Difference (%)']}%")

        bias     = fairness["Bias Detected"]
        b_colour = "red" if bias == "Yes" else "green"
        st.markdown(f"**Bias Detected: :{b_colour}[{bias}]**")

        # ── LLM explanation ───────────────────────────────────────────────────
        st.subheader("AI Explanation")
        st.caption(f"Domain context: `{domain_key}` — using domain-specific system prompt")

        with st.spinner("Generating in-depth explanation…"):
            explanation = generate_explanation(
                result,
                domain=domain_key,          # routes to the right system prompt
                sample_values=sample_values, # gives LLM the actual numbers
            )

        st.markdown(explanation)

        # ── proof callout ─────────────────────────────────────────────────────
        st.divider()
        st.success(
            f"✅ **Same `wrap()`. Same output. Different model.**  \n"
            f"Domain: `{domain_key}`  ·  Algorithm: `{model_name}`  ·  "
            f"Features: `{features}`"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB B  —  text / LLM with model dropdown
# ══════════════════════════════════════════════════════════════════════════════

with tab_b:
    st.header("Text / LLM Classifier + XAI / RAI")

    # ── model selector ────────────────────────────────────────────────────────
    model_choice = st.selectbox(
        "Select a model",
        list(TEXT_MODEL_CONFIG.keys()),
        key="text_model_select",
    )
    tcfg = TEXT_MODEL_CONFIG[model_choice]

    st.info(tcfg["description"])
    st.caption(f"HuggingFace model: `{tcfg['model_id']}`")

    with st.expander("💡 What to watch for in this demo"):
        st.markdown(tcfg["what_to_watch"])

    with st.spinner(f"Loading `{tcfg['model_id']}`…"):
        text_adapter = load_text_adapter(tcfg["model_id"])
    st.caption("✅ Model loaded and cached — switching back won't reload it.")

    chosen = st.selectbox("Quick examples", tcfg["examples"], key="text_examples")
    raw_input = st.text_area(
        "Input text",
        value="" if chosen == tcfg["examples"][0] else chosen,
        height=100,
        placeholder="Type any sentence here…",
        key="text_input",
    )

    if st.button("Run Text Analysis", key="text_run"):
        if not raw_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner(f"Running SHAP + Detoxify on `{tcfg['model_id']}`…"):
                result = text_adapter.predict(raw_input)

            # ── agreement callout (only for toxic classifier) ─────────────────
            if "toxic" in tcfg["model_id"].lower():
                classifier_toxic = "NON" not in result.prediction.label
                detoxify_toxic   = result.responsibility.bias_flagged
                if classifier_toxic == detoxify_toxic:
                    agree_colour = "green"
                    agree_text   = (
                        "✅ **Both systems agree** — "
                        + ("doubly validated TOXIC signal." if classifier_toxic
                           else "both give the all-clear.")
                    )
                else:
                    agree_colour = "orange"
                    agree_text   = (
                        f"⚠️ **Systems disagree — edge case detected.**  \n"
                        f"Classifier: **{'TOXIC' if classifier_toxic else 'NON-TOXIC'}**  "
                        f"· Detoxify: **{'FLAGGED' if detoxify_toxic else 'CLEAN'}**  \n"
                        "This input warrants human review."
                    )
                st.markdown(f"### Classifier vs RAI Audit")
                st.markdown(f":{agree_colour}[{agree_text}]")
                st.divider()

            # ── full XAI output (shared renderer) ─────────────────────────────
            render_text_result(result, raw_input, tcfg["domain_key"])

            st.divider()
            st.success(
                f"✅ **Same `wrap()`. Same output structure. Different model.**  \n"
                f"Model: `{tcfg['model_id']}`"
            )