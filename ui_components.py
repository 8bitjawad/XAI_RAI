"""
ui_components.py  —  Shared Streamlit rendering functions.
Each function is called by multiple tabs — keeping them here means
app.py never repeats the same 20-line chart block four times.
"""

import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from config import RAI_FLAG_THRESHOLD


def render_bar_chart(
    labels: list[str],
    values: list[float],
    xlabel: str,
    title: str,
    thresholds: list[tuple] = None,
) -> None:
    """
    Horizontal bar chart used by both SHAP (tabular/text) and
    word-influence (generative/image) sections.
    Green bars = positive, red = negative.
    thresholds: list of (value, colour, label) for reference lines.
    """
    colours = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(labels) * 0.45)))
    ax.barh(labels[::-1], values[::-1], color=colours[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if thresholds:
        for val, col, lbl in thresholds:
            ax.axvline(val, color=col, linewidth=0.8, linestyle="--", label=lbl)
        ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def render_rai_table(scores: dict[str, float]) -> None:
    """
    Renders the Detoxify scorecard table with red highlighting
    on rows above RAI_FLAG_THRESHOLD.
    """
    df = pd.DataFrame(scores.items(), columns=["Category", "Score"])

    def _highlight(row):
        return [
            "background-color: #fde8e8" if row["Score"] > RAI_FLAG_THRESHOLD else ""
            for _ in row
        ]

    st.dataframe(
        df.style.apply(_highlight, axis=1).format({"Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )


def render_rai_flag(flagged: bool, label: str = "Bias / Toxicity Flag") -> None:
    colour = "red" if flagged else "green"
    text   = "⚠ FLAGGED" if flagged else "✓ CLEAN"
    st.markdown(f"**{label}: :{colour}[{text}]**")


def render_consistency_metrics(c) -> None:
    """Renders the three consistency metric cards + variant expander."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Original confidence", f"{c.original_score:.1%}")
    col2.metric("Variant confidence",  f"{c.variant_score:.1%}",
                delta=f"{c.delta:+.1%}")
    col3.metric("Flagged", "⚠ Yes" if c.flagged else "✓ No",
                delta_color="inverse")
    with st.expander("Variant text used"):
        st.write(c.variant_text)


def render_text_result(result, raw_input: str, domain_key: str) -> None:
    """
    Full XAI output for any text classifier result.
    Called by Tab B for both DistilBERT and ToxicBERT.
    """
    from llm_engine import generate_explanation

    # Prediction
    st.subheader("Prediction")
    label  = result.prediction.label
    score  = result.prediction.score
    colour = "green" if ("NON" in label or label == "POSITIVE") else "red"
    st.markdown(f"**:{colour}[{label}]** — confidence `{score:.1%}`")
    st.caption(f"Raw logits: {[round(l, 3) for l in result.prediction.raw_logits]}")

    # SHAP chart
    st.subheader("Word Importance (SHAP)")
    importance = result.word_importance
    if importance:
        tokens = list(importance.keys())[:10]
        values = [importance[t] for t in tokens]
        render_bar_chart(
            tokens, values,
            xlabel="SHAP value  (+ → pushes toward predicted label)",
            title="Token Attribution Map",
        )

    # RAI scorecard
    st.subheader("Responsibility Scorecard  *(Detoxify)*")
    r = result.responsibility
    render_rai_table({
        "Toxicity": r.toxicity, "Severe Toxicity": r.severe_toxicity,
        "Obscene":  r.obscene,  "Insult": r.insult,
        "Threat":   r.threat,   "Identity Attack": r.identity_attack,
    })
    render_rai_flag(r.bias_flagged)

    # Consistency
    st.subheader("Consistency Check")
    render_consistency_metrics(result.consistency)

    # LLM explanation
    st.subheader("AI Explanation")
    text_result_for_llm = {
        "prediction":  label,
        "confidence":  score,
        "top_factors": list(importance.items())[:5],
        "shap_values": importance,
        "lime_rules":  [],
    }
    with st.spinner("Generating explanation…"):
        st.markdown(generate_explanation(
            text_result_for_llm,
            domain=domain_key,
            sample_values={"input_text": raw_input},
        ))


def render_image_rai(rai) -> None:
    """Renders CLIP alignment + NSFW scorecard for image results."""
    r1, r2, r3 = st.columns(3)
    r1.metric("CLIP Alignment", f"{rai.clip_alignment:.2f}",
              help="Semantic match between image and label/prompt. Low = mismatch.")
    r2.metric("NSFW Score", f"{rai.nsfw_score:.4f}")
    ov_colour = "red" if rai.overall_flagged else "green"
    r3.markdown(
        f"**RAI: :{ov_colour}[{'⚠ FLAGGED' if rai.overall_flagged else '✓ CLEAN'}]**"
    )
    if rai.alignment_flagged:
        st.warning(f"⚠️ Low CLIP alignment ({rai.clip_alignment:.2f}) — label may not match image.")
    if rai.nsfw_flagged:
        st.error(f"⚠️ NSFW content detected (score: {rai.nsfw_score:.4f}).")