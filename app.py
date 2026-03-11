# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import shap
# from llm_engine import generate_explanation
# from core import RAIWrapper

# model = joblib.load("model.pkl")

# feature_names = joblib.load("features.pkl")
# X_test_full = joblib.load("X_test_full.pkl")
# y_test = joblib.load("y_test.pkl")

# X_train = pd.DataFrame(
#     np.zeros((1, len(feature_names))),
#     columns=feature_names
# )

# from core import wrap

# rai_model = wrap(
#     model,
#     X_train,
#     sensitive_column="trader_group"
# )

# st.title("Universal XAI + Responsible AI Layer")

# st.write("Demo: Model Prediction with Explanation and Fairness Checks")

# st.header("Input Features")

# open_price = st.number_input("Open Price")
# high_price = st.number_input("High Price")
# low_price = st.number_input("Low Price")
# close_price = st.number_input("Close Price")
# volume = st.number_input("Volume")
# run_button = st.button("Run Prediction")

# if run_button:

#     if rai_model is None:
#         st.error("Model not loaded yet.")
#     else:
#         sample = pd.DataFrame([[open_price, high_price, low_price, close_price, volume]],
#         columns=feature_names
# )

#         result = rai_model.predict(sample)

#         st.subheader("Prediction")
#         prediction = result["prediction"]

#         if prediction == 1:
#             prediction_text = "Price is likely to go UP"
#         else:
#             prediction_text = "Price is likely to go DOWN"
#         st.write(prediction_text)

#         st.subheader("Confidence")
#         st.write(result["confidence"])

#         st.subheader("Top Factors")

#         for feature, value in result["top_factors"]:
#             st.write(f"{feature} ({round(value,3)})")

#         plot = rai_model.explain_engine.shap_plot(sample)

#         st.subheader("Feature Impact Visualization")

#         st.pyplot(plot)

#         fairness = rai_model.fairness_report(X_test_full, y_test)

#         st.subheader("Responsible AI Report")

#         st.write(f"Protected Attribute: {fairness['Protected Attribute']}")
#         st.write(f"Prediction Difference: {fairness['Prediction Difference (%)']}%")
#         st.write(f"Bias Detected: {fairness['Bias Detected']}")

#         explanation = generate_explanation(result)

#         st.subheader("AI Explanation")

#         st.write(explanation)

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from llm_engine import generate_explanation

# ── your existing imports ─────────────────────────────────────────────────────
from core import wrap          # replaces "from core import wrap / RAIWrapper"

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS  (Path A — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
RAI_FLAG_THRESHOLD = 0.10
model        = joblib.load("model.pkl")
feature_names = joblib.load("features.pkl")
X_test_full  = joblib.load("X_test_full.pkl")
y_test       = joblib.load("y_test.pkl")

X_train = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names,
)

rai_model = wrap(model, X_train=X_train, sensitive_column="trader_group")

# ══════════════════════════════════════════════════════════════════════════════
#  PATH B  —  text adapter (loaded once, cached by wrap())
# ══════════════════════════════════════════════════════════════════════════════

TEXT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource                  # keeps the 250 MB model in memory across reruns
def load_text_adapter():
    return wrap(TEXT_MODEL)

text_adapter = load_text_adapter()

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("Universal XAI + Responsible AI Layer")
st.write("Demo: Model Prediction with Explanation and Fairness Checks")

tab_a, tab_b = st.tabs(["📊 Path A — Tabular Model", "💬 Path B — Text / LLM"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB A  (your original code, untouched)
# ─────────────────────────────────────────────────────────────────────────────

with tab_a:
    st.header("Input Features")

    open_price  = st.number_input("Open Price")
    high_price  = st.number_input("High Price")
    low_price   = st.number_input("Low Price")
    close_price = st.number_input("Close Price")
    volume      = st.number_input("Volume")
    run_button  = st.button("Run Prediction", key="tabular_run")

    if run_button:
        sample = pd.DataFrame(
            [[open_price, high_price, low_price, close_price, volume]],
            columns=feature_names,
        )

        result = rai_model.predict(sample)

        st.subheader("Prediction")
        prediction_text = (
            "Price is likely to go UP" if result["prediction"] == 1
            else "Price is likely to go DOWN"
        )
        st.write(prediction_text)

        st.subheader("Confidence")
        st.write(result["confidence"])

        st.subheader("Top Factors")
        for feature, value in result["top_factors"]:
            st.write(f"{feature} ({round(value, 3)})")

        st.subheader("Feature Impact Visualization")
        plot = rai_model.explain_engine.shap_plot(sample)
        st.pyplot(plot)

        fairness = rai_model.fairness_report(X_test_full, y_test)

        st.subheader("Responsible AI Report")
        st.write(f"Protected Attribute: {fairness['Protected Attribute']}")
        st.write(f"Prediction Difference: {fairness['Prediction Difference (%)']}%")
        st.write(f"Bias Detected: {fairness['Bias Detected']}")

        explanation = generate_explanation(result)
        st.subheader("AI Explanation")
        st.write(explanation)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB B  (new — Path B text layer)
# ─────────────────────────────────────────────────────────────────────────────

with tab_b:
    st.header("Text Sentiment Classifier + XAI / RAI")

    st.info(f"Model: `{TEXT_MODEL}`", icon="🤖")

    # ── pre-loaded test sentences so you can try it immediately ───────────────
    EXAMPLES = [
        "Select an example or type your own below…",
        "This market rally was absolutely fantastic — best quarter of the year!",
        "I hated every single minute of this terrible earnings call.",
        "The trade was okay, nothing special but not a total disaster.",
        "You are completely useless and I hate everything about this.",   # toxicity demo
    ]

    chosen = st.selectbox("Quick examples", EXAMPLES)

    raw_input = st.text_area(
        "Input text",
        value="" if chosen == EXAMPLES[0] else chosen,
        height=100,
        placeholder="Type any sentence here…",
    )

    run_text = st.button("Run Text Analysis", key="text_run")

    if run_text:
        if not raw_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Running SHAP + Detoxify…"):
                result = text_adapter.predict(raw_input)

            # ── 1. Prediction ─────────────────────────────────────────────────
            st.subheader("Prediction")
            label = result.prediction.label
            score = result.prediction.score
            colour = "green" if label == "POSITIVE" else "red"
            st.markdown(
                f"**:{colour}[{label}]** — confidence `{score:.1%}`"
            )
            st.caption(f"Raw logits: {[round(l,3) for l in result.prediction.raw_logits]}")

            # ── 2. Word Importance (SHAP) ─────────────────────────────────────
            st.subheader("Word Importance (SHAP)")

            importance = result.word_importance
            if importance:
                tokens = list(importance.keys())[:10]        # top 10 tokens
                values = [importance[t] for t in tokens]

                colours = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, max(3, len(tokens) * 0.45)))
                bars = ax.barh(tokens[::-1], values[::-1], color=colours[::-1])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value  (+ → pushes toward predicted label)")
                ax.set_title("Token Attribution Map")
                st.pyplot(fig)
                plt.close(fig)

            # ── 3. Responsibility Scorecard (Detoxify) ────────────────────────
            st.subheader("Responsibility Scorecard")

            r = result.responsibility
            scores = {
                "Toxicity":         r.toxicity,
                "Severe Toxicity":  r.severe_toxicity,
                "Obscene":          r.obscene,
                "Insult":           r.insult,
                "Threat":           r.threat,
                "Identity Attack":  r.identity_attack,
            }

            rai_df = pd.DataFrame(
                scores.items(), columns=["Category", "Score"]
            )

            # colour rows above the flag threshold
            def _highlight(row):
                return [
                    "background-color: #fde8e8" if row["Score"] > RAI_FLAG_THRESHOLD
                    else "" for _ in row
                ]

            st.dataframe(
                rai_df.style
                    .apply(_highlight, axis=1)
                    .format({"Score": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

            flag_colour = "red" if r.bias_flagged else "green"
            st.markdown(
                f"**Bias / Toxicity Flag: :{flag_colour}[{'⚠ FLAGGED' if r.bias_flagged else '✓ CLEAN'}]**"
            )

            # ── 4. Consistency Check ──────────────────────────────────────────
            st.subheader("Consistency Check")

            c = result.consistency
            col1, col2, col3 = st.columns(3)
            col1.metric("Original confidence",  f"{c.original_score:.1%}")
            col2.metric("Variant confidence",   f"{c.variant_score:.1%}",
                        delta=f"{c.delta:+.1%}")
            col3.metric("Flagged",
                        "⚠ Yes" if c.flagged else "✓ No",
                        delta_color="inverse")

            with st.expander("Variant text used"):
                st.write(c.variant_text)


# ── threshold constant needed inside Tab B ────────────────────────────────────
