"""
app.py  —  Universal XAI + Responsible AI Layer
Tab A  →  Tabular  |  Tab B  →  Text  |  Tab C  →  Generative LLM  |  Tab D  →  Image
"""

import io
import re
import os
import base64
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

from core import wrap
from llm_engine import generate_explanation
from generative_adapter import GenerativeAdapter
from image_classifier_adapter import (ImageClassifierAdapter,
                                       make_hf_pipeline_predict_fn)
from config import (
    RAI_FLAG_THRESHOLD, DOMAIN_CONFIG, TEXT_MODEL_CONFIG,
    CHATBOT_PRESETS, IMAGE_DEMO_URLS, GENERATION_EXAMPLES,
)
from ui_components import (
    render_bar_chart, render_rai_table, render_rai_flag,
    render_consistency_metrics, render_text_result, render_image_rai,
)

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CACHED RESOURCE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_domain_adapter(domain_name: str):
    cfg = DOMAIN_CONFIG[domain_name]
    df  = cfg["synth"](500)
    X   = df[cfg["features"]]
    y   = cfg["target"](df)
    clf = cfg["model_cls"](**cfg["model_kwargs"])
    clf.fit(X, y)
    return wrap(clf, X_train=X, sensitive_column=cfg["sensitive_col"]), df, y


@st.cache_resource
def load_text_adapter(model_id: str):
    return wrap(model_id)


@st.cache_resource
def build_generative_adapter(preset_name: str) -> GenerativeAdapter:
    preset  = CHATBOT_PRESETS[preset_name]
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model   = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")

    def llm_fn(prompt: str) -> str:
        messages = [{"role": "system", "content": preset["system"]},
                    {"role": "user",   "content": prompt}]
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.3},
                timeout=30,
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM error: {e}]"

    return GenerativeAdapter(llm_fn=llm_fn, model_label=preset["label"],
                             n_consistency_samples=3, rate_limit_delay=0.3)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("Universal XAI + Responsible AI Layer")

tab_a, tab_b, tab_c, tab_d = st.tabs([
    "📊 Path A — Tabular",
    "💬 Path B — Text",
    "🤖 Path C — Generative LLM",
    "🖼️ Path D — Image Classifier",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB A  —  Tabular / model-agnostic
# ══════════════════════════════════════════════════════════════════════════════

with tab_a:
    st.header("Model-Agnostic Tabular XAI")

    domain     = st.selectbox("Domain", list(DOMAIN_CONFIG.keys()), key="a_domain")
    cfg        = DOMAIN_CONFIG[domain]
    model_name = cfg["model_cls"].__name__

    st.info(cfg["description"])
    st.caption(f"Algorithm: `{model_name}`")

    with st.spinner(f"Training {model_name}…"):
        adapter, df_full, y_synth = build_domain_adapter(domain)
    features = cfg["features"]

    # Feature inputs
    st.subheader("Input Features")
    input_vals, cols = [], st.columns(2)
    for i, feat in enumerate(features):
        val = cols[i % 2].number_input(feat, value=float(cfg["defaults"][i]),
                                        help=cfg["feature_descriptions"][feat],
                                        key=f"a_{domain}_{feat}")
        input_vals.append(val)

    if st.button("Run Prediction", key="a_run"):
        sample        = pd.DataFrame([input_vals], columns=features)
        sample_values = dict(zip(features, input_vals))

        with st.spinner("Running XAI…"):
            result = adapter.predict(sample)

        # Prediction
        st.subheader("Prediction")
        pred_int = int(result["prediction"])
        colour   = "green" if pred_int == 0 else "red"
        st.markdown(f"### :{colour}[{cfg['label_map'][pred_int]}]")
        st.metric("Confidence", f"{result['confidence']:.1%}")

        # SHAP chart
        st.subheader("Feature Attribution (SHAP)")
        shap_items = sorted(result["shap_values"].items(), key=lambda x: x[1])
        render_bar_chart(
            [x[0] for x in shap_items], [x[1] for x in shap_items],
            xlabel="Mean |SHAP value|",
            title=f"Feature Attribution  ·  {model_name}",
        )
        st.markdown("**Top 3 drivers:**")
        for feat, val in result["top_factors"]:
            st.write(f"- **{feat}** {'increased' if val > 0 else 'reduced'} score by `{round(val,4)}`")

        with st.expander("LIME rules"):
            for rule, w in result["lime_rules"]:
                st.write(f"`{rule}` → `{round(w,4)}`")

        # Fairness
        st.subheader("Fairness Report")
        sensitive_col     = cfg["sensitive_col"]
        fairness_input_df = df_full[[sensitive_col] + features].copy()
        fairness          = adapter.fairness_report(fairness_input_df, y_synth)
        f1, f2 = st.columns(2)
        f1.metric("Protected Attribute",   fairness["Protected Attribute"])
        f2.metric("Prediction Difference", f"{fairness['Prediction Difference (%)']}%")
        bias = fairness["Bias Detected"]
        st.markdown(f"**Bias: :{'red' if bias == 'Yes' else 'green'}[{bias}]**")

        # LLM explanation
        st.subheader("AI Explanation")
        with st.spinner("Generating…"):
            st.markdown(generate_explanation(result, domain=cfg["domain_key"],
                                             sample_values=sample_values))

        st.success(f"✅ `wrap({model_name})` · features: `{features}`")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB B  —  Text classifiers
# ══════════════════════════════════════════════════════════════════════════════

with tab_b:
    st.header("Text Classifier XAI / RAI")

    model_choice = st.selectbox("Model", list(TEXT_MODEL_CONFIG.keys()), key="b_model")
    tcfg         = TEXT_MODEL_CONFIG[model_choice]

    st.info(tcfg["description"])
    with st.expander("💡 What to watch"):
        st.markdown(tcfg["what_to_watch"])

    with st.spinner(f"Loading `{tcfg['model_id']}`…"):
        text_adapter = load_text_adapter(tcfg["model_id"])

    chosen    = st.selectbox("Examples", tcfg["examples"], key="b_examples")
    raw_input = st.text_area("Input text",
                              value="" if chosen == tcfg["examples"][0] else chosen,
                              height=100, key="b_input")

    if st.button("Run Analysis", key="b_run"):
        if not raw_input.strip():
            st.warning("Enter some text first.")
        else:
            with st.spinner("Running SHAP + Detoxify…"):
                result = text_adapter.predict(raw_input)

            # Agreement callout for toxic model
            if "toxic" in tcfg["model_id"].lower():
                clf_toxic = "NON" not in result.prediction.label
                det_toxic = result.responsibility.bias_flagged
                if clf_toxic == det_toxic:
                    colour = "green"
                    msg    = f"✅ Both agree — {'TOXIC (double validated)' if clf_toxic else 'clean'}"
                else:
                    colour = "orange"
                    msg    = (f"⚠️ Disagreement: Classifier={'TOXIC' if clf_toxic else 'CLEAN'} "
                              f"· Detoxify={'FLAGGED' if det_toxic else 'CLEAN'} — review needed")
                st.markdown(f":{colour}[{msg}]")
                st.divider()

            render_text_result(result, raw_input, tcfg["domain_key"])
            st.success(f"✅ `wrap('{tcfg['model_id']}')`")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB C  —  Generative LLM
# ══════════════════════════════════════════════════════════════════════════════

with tab_c:
    st.header("Generative LLM Explainability")
    st.markdown("Explains free-text LLM output via perturbation, consistency probing, and bias testing.")

    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("OPENROUTER_API_KEY not in .env — required for this tab.")
        st.stop()

    preset_name = st.selectbox("Chatbot persona", list(CHATBOT_PRESETS.keys()), key="c_preset")
    preset      = CHATBOT_PRESETS[preset_name]
    st.info(f"**System prompt:** _{preset['system']}_")
    with st.expander("💡 What to watch"):
        st.markdown(preset["what_to_watch"])

    st.warning("~11–16 LLM calls per run · est. cost < $0.01 on free model", icon="💰")

    chosen_ex = st.selectbox("Examples", preset["examples"], key="c_examples")
    gen_input = st.text_area("Prompt",
                              value="" if chosen_ex == preset["examples"][0] else chosen_ex,
                              height=100, key="c_input")

    if st.button("Run Explanation", key="c_run"):
        if not gen_input.strip():
            st.warning("Enter a prompt first.")
        else:
            adapter = build_generative_adapter(preset_name)
            with st.spinner(f"Running ~{min(len(gen_input.split()),15)+4} LLM calls…"):
                result = adapter.explain(gen_input)

            st.subheader("LLM Response")
            st.markdown(f"> {result.llm_response}")

            st.subheader("Explanation Summary")
            st.info(result.summary())

            # Word influence chart
            st.subheader("Word Influence (Perturbation)")
            if result.word_influences:
                words  = [w.word for w in result.word_influences]
                scores = [w.influence_score for w in result.word_influences]
                render_bar_chart(
                    words, scores,
                    xlabel="Influence score (output change when word masked)",
                    title="Prompt Word Attribution",
                    thresholds=[(0.3, "red", "High"), (0.15, "orange", "Medium")],
                )

            # Consistency
            st.subheader("Consistency")
            c = result.consistency
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean similarity",  f"{c.mean_similarity:.2f}")
            m2.metric("Min similarity",   f"{c.min_similarity:.2f}")
            colour = "red" if c.flagged else "green"
            m3.markdown(f"**:{ colour}[{'⚠ INCONSISTENT' if c.flagged else '✓ CONSISTENT'}]**")

            # Bias probe
            st.subheader("Demographic Bias Probe")
            if not result.bias_probe.tested:
                st.info("No demographic keywords detected in prompt.")
            else:
                b1, b2 = st.columns(2)
                b1.metric("Max drift", f"{result.bias_probe.max_drift:.2f}")
                bc = "red" if result.bias_probe.flagged else "green"
                b2.markdown(f"**:{bc}[{'⚠ BIAS DETECTED' if result.bias_probe.flagged else '✓ NONE'}]**")
                if result.bias_probe.flagged_swaps:
                    with st.expander("Flagged swaps"):
                        for s in result.bias_probe.flagged_swaps:
                            st.write(f"`{s['original_term']}` → `{s['swapped_term']}` drift: `{s['drift']}`")

            # RAI
            st.subheader("RAI Scorecard")
            rai = result.rai
            render_rai_table({
                "Toxicity (prompt)":   rai.prompt_toxicity,
                "Insult (prompt)":     rai.prompt_insult,
                "Toxicity (response)": rai.response_toxicity,
                "Insult (response)":   rai.response_insult,
                "Threat (response)":   rai.response_threat,
            })
            render_rai_flag(rai.overall_flagged)

            with st.expander("Full JSON audit trail"):
                st.code(result.to_json(), language="json")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB D  —  Image Classifier XAI + RAI
# ══════════════════════════════════════════════════════════════════════════════

# ONLY showing modified TAB D section — rest of your file remains unchanged

from skimage.segmentation import mark_boundaries

# ══════════════════════════════════════════════════════════════════════════════
#  TAB D  —  Image Classifier XAI + RAI (UPDATED)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  TAB D  —  Image Classifier XAI + RAI
#  Drop-in replacement for the tab_d block in app.py
# ══════════════════════════════════════════════════════════════════════════════

from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with tab_d:
    st.header("Image Classification XAI + RAI")

    MODEL_OPTIONS = {
        "microsoft/resnet-50":          "microsoft/resnet-50",
        "google/efficientnet-b7":        "google/efficientnet-b7",
        "microsoft/resnet-18 (fast)":    "microsoft/resnet-18",
    }

    model_choice   = st.selectbox("Model", list(MODEL_OPTIONS.keys()), key="d_model")
    selected_model = MODEL_OPTIONS[model_choice]

    method = st.selectbox(
        "Explanation method",
        ["both", "occlusion", "lime"],
        index=0,
        key="d_method",
        help=(
            "occlusion = fast (~2-3 s, one batched call). "
            "lime = detailed superpixel map (~20-30 s). "
            "both = show both."
        ),
    )

    DEMO_URLS = {
        "Cat":         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "Dog":         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
        "Sports car":  "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2012_Aston_Martin_Rapide_%28cropped%29.JPG/320px-2012_Aston_Martin_Rapide_%28cropped%29.JPG",
    }

    demo_choice   = st.selectbox("Demo image", ["Upload"] + list(DEMO_URLS.keys()), key="d_demo")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg", "webp"], key="d_upload")

    if st.button("Run Explanation", key="d_run"):
        # ── Load image ───────────────────────────────────────────────────────
        pil_image = None
        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
        elif demo_choice != "Upload":
            try:
                pil_image = Image.open(
                    io.BytesIO(requests.get(DEMO_URLS[demo_choice], timeout=10).content)
                ).convert("RGB")
            except Exception as e:
                st.error(f"Could not load demo image: {e}")

        if pil_image is None:
            st.warning("Please upload an image or select a demo.")
            st.stop()

        st.image(pil_image, caption="Input image", width=300)

        # ── Build adapter ────────────────────────────────────────────────────
        with st.spinner(f"Loading {selected_model}…"):
            predict_fn, class_names = make_hf_pipeline_predict_fn(selected_model)

        adapter = ImageClassifierAdapter(predict_fn=predict_fn, method=method)

        # ── Run explanation ──────────────────────────────────────────────────
        spinner_msg = {
            "occlusion": "Running occlusion (batched, ~3 s)…",
            "lime":      "Running LIME (~20–30 s)…",
            "both":      "Running occlusion + LIME (~25–35 s)…",
        }[method]

        with st.spinner(spinner_msg):
            result = adapter.explain(pil_image)

        # ── Occlusion output ─────────────────────────────────────────────────
        if "occlusion_fig" in result:
            st.subheader("⚡ Fast Explanation — Occlusion Sensitivity")
            st.pyplot(result["occlusion_fig"])
            st.caption(
                "Each region is systematically occluded. "
                "Bright areas caused the largest drop in confidence — "
                "the model relied on them most."
            )

        # ── LIME output ──────────────────────────────────────────────────────
        if "lime" in result:
            st.subheader("🔍 Detailed Explanation — LIME Superpixels")

            lime_exp = result["lime"]
            top_idx  = result["top_idx"]   # ← use the real baseline top_idx

            # Safely fetch explanation — fall back to whatever LIME ranked first
            # if top_idx isn't in LIME's results (shouldn't happen with top_labels=5)
            label_to_explain = top_idx if top_idx in lime_exp.local_exp else lime_exp.top_labels[0]

            temp, mask = lime_exp.get_image_and_mask(
                label_to_explain,
                positive_only=False,
                num_features=8,
                hide_rest=False,
            )

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(np.array(pil_image.resize((224, 224))))
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(mark_boundaries(temp / 255.0, mask))
            axes[1].set_title("LIME — green supports, red opposes")
            axes[1].axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Show top prediction label + score from LIME's own probs
            top_score = float(lime_exp.local_pred[0]) if hasattr(lime_exp, "local_pred") else None
            label_name = class_names[label_to_explain] if label_to_explain < len(class_names) else f"class_{label_to_explain}"
            conf_str   = f" ({top_score:.1%} local confidence)" if top_score is not None else ""
            st.caption(f"Explaining prediction: **{label_name}**{conf_str}")

        st.success("Done ✅")