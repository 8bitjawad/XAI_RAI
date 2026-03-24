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
from image_adapter import ImageClassificationAdapter, ImageGenerationAdapter
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
    "🖼️ Path D — Image",
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

    st.warning("~11–16 LLM calls per run", icon="💰")

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
#  TAB D  —  Image models
# ══════════════════════════════════════════════════════════════════════════════

with tab_d:
    st.header("Image Model XAI + RAI")

    mode = st.radio("Mode", ["🔍 C1 — Classification", "🎨 C2 — Generation"],
                    horizontal=True, key="d_mode")
    st.divider()

    # ── C1: Classification ────────────────────────────────────────────────────
    if mode == "🔍 C1 — Classification":
        st.subheader("Image Classification + LIME")
        st.info("**Model:** `google/vit-base-patch16-224`  ·  **XAI:** LIME superpixel masking (black-box Grad-CAM substitute)")

        with st.expander("⚙️ API calls: ~10 (8 LIME + 1 classify + 1 NSFW)"):
            st.markdown(
                "- Reduced from 22 → **10 calls** (8 LIME samples — fast, demo-accurate)\n"
                "- CLIP alignment runs **locally** — no API call\n"
                "- Free HF Inference API — no GPU needed, no token required"
            )

        demo_url = st.selectbox("Demo image URL", IMAGE_DEMO_URLS, key="d_url")
        uploaded = st.file_uploader("Or upload an image", type=["jpg","jpeg","png","webp"], key="d_upload")

        if st.button("Run Classification + Explanation", key="d_clf_run"):
            pil_image = None
            if uploaded:
                pil_image = Image.open(uploaded).convert("RGB")
            elif demo_url != IMAGE_DEMO_URLS[0]:
                try:
                    pil_image = Image.open(
                        io.BytesIO(requests.get(demo_url, timeout=10).content)
                    ).convert("RGB")
                except Exception as e:
                    st.error(f"Could not load image: {e}")

            if pil_image is None:
                st.warning("Select a demo URL or upload an image.")
            else:
                left, right = st.columns(2)
                left.image(pil_image, caption="Input", use_column_width=True)

                with st.spinner("Classifying + LIME masking (~10 API calls, ~20s)…"):
                    try:
                        result = ImageClassificationAdapter().explain(pil_image)
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        st.stop()

                # Prediction
                st.subheader("Prediction")
                p1, p2 = st.columns(2)
                p1.metric("Label",      result.top_label)
                p2.metric("Confidence", f"{result.top_score:.1%}")
                with st.expander("All top-5 labels"):
                    st.dataframe(pd.DataFrame(result.all_labels)
                                   .style.format({"score":"{:.4f}"}),
                                 use_container_width=True, hide_index=True)

                # Heatmap
                st.subheader("LIME Heatmap")
                st.caption("Green = supports prediction · Red = opposes")
                heatmap = Image.open(io.BytesIO(base64.b64decode(result.heatmap_b64)))
                right.image(heatmap, caption="LIME heatmap", use_column_width=True)

                # RAI
                st.subheader("Responsibility Scorecard")
                render_image_rai(result.rai)
                st.success(f"✅ `{result.model_id}`  ·  `{result.method}`")

    # ── C2: Generation ────────────────────────────────────────────────────────
    else:
        st.subheader("Image Generation + Prompt Attribution")
        st.info("**Model:** `stabilityai/stable-diffusion-2-1`  ·  **XAI:** Prompt perturbation + CLIP similarity")

        with st.expander("⚙️ API calls: ~10–14 · ~2–4 min total"):
            st.markdown(
                "- 1 baseline generation + 1 per content word (max 12) + 1 NSFW\n"
                "- Each generation call ~10–20s on free HF tier\n"
                "- Add `HF_API_TOKEN` to `.env` for faster inference\n"
                "- CLIP similarity runs **locally** — no API call"
            )

        gen_ex = st.selectbox("Examples", GENERATION_EXAMPLES, key="d_gen_ex")
        gen_prompt = st.text_input("Prompt",
                                    value="" if gen_ex == GENERATION_EXAMPLES[0] else gen_ex,
                                    placeholder="Describe the image…", key="d_gen_prompt")
        st.warning("Generation is slow on free HF tier (~10–20s per image).", icon="⏳")

        if st.button("Generate + Explain", key="d_gen_run"):
            if not gen_prompt.strip():
                st.warning("Enter a prompt first.")
            else:
                with st.spinner("Generating + perturbing prompt words…"):
                    try:
                        result = ImageGenerationAdapter().explain(gen_prompt.strip())
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        st.stop()

                # Generated image
                st.subheader("Generated Image")
                gen_img = Image.open(io.BytesIO(base64.b64decode(result.generated_image_b64)))
                st.image(gen_img, width=400, caption=f'"{result.prompt}"')

                # Word attribution chart
                st.subheader("Prompt Word Attribution")
                if result.word_influences:
                    words  = [w.word for w in result.word_influences]
                    scores = [w.influence_score for w in result.word_influences]
                    render_bar_chart(
                        words, scores,
                        xlabel="Visual drift (CLIP similarity drop when word masked)",
                        title="Prompt Word Attribution",
                        thresholds=[(0.25, "red", ">0.25 high"),
                                    (0.10, "orange", ">0.10 medium")],
                    )
                    st.markdown(f"**Most influential:** {', '.join(f'`{w}`' for w in result.top_words)}")

                # RAI
                st.subheader("Responsibility Scorecard")
                render_image_rai(result.rai)

                with st.expander("JSON audit trail"):
                    st.code(result.to_json(), language="json")

                st.success(f"✅ `{result.model_id}`  ·  top words: `{result.top_words}`")