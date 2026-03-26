# """
# image_adapter.py  —  Path C: Image XAI + RAI Middleware
# =========================================================
# Covers two sub-paths:

# SUB-PATH C1 — Image Classification  (e.g. ResNet via HF Inference API)
#     XAI: LIME superpixel masking — divides image into regions, masks each,
#          sends to API, measures which regions drove the label prediction.
#          This is the correct black-box substitute for Grad-CAM when you
#          have no access to model gradients over an API.
#     RAI: CLIP alignment score (does the label match the image semantically?)
#          + NSFW detection via a dedicated HF classifier.

# SUB-PATH C2 — Image Generation  (e.g. Stable Diffusion via HF Inference API)
#     XAI: Prompt token perturbation — masks each word in the prompt,
#          regenerates the image, measures CLIP similarity between original
#          and perturbed image. Words that cause the most visual drift are
#          most influential.
#     RAI: CLIP alignment (does the generated image match the prompt intent?)
#          + NSFW detection on the generated image.

# CLIP runs locally (all-MiniLM style via open_clip / transformers).
# It is small (~400 MB) and free — no API needed.

# All HuggingFace Inference API calls use the free tier by default.
# Set HF_API_TOKEN in .env to access gated models or higher rate limits.

# ARCHITECTURE — MODEL-AGNOSTIC DESIGN:
# Both adapters accept an injectable function instead of a hardcoded model.
# The XAI techniques (LIME, perturbation, CLIP) are model-agnostic — they
# only need input/output pairs, not model internals. The injected function
# is the only model-specific piece, keeping the XAI/RAI layer universal.

#     classify_fn : Callable[[PIL.Image], list[dict]]
#                   Takes an image, returns [{"label": str, "score": float}, ...]
#                   Works with: HF API, OpenAI Vision, Azure CV, local PyTorch, anything.

#     generate_fn : Callable[[str], PIL.Image]
#                   Takes a prompt string, returns a PIL Image.
#                   Works with: HF API, DALL-E, Midjourney API, local SD, anything.

# Usage:
#     from image_adapter import ImageClassificationAdapter, ImageGenerationAdapter
#     from image_adapter import make_hf_classify_fn, make_hf_generate_fn

#     # ── Plug in HF Inference API (built-in helper) ──
#     clf = ImageClassificationAdapter(classify_fn=make_hf_classify_fn())
#     result = clf.explain(pil_image)

#     # ── Plug in any custom model ──
#     def my_classifier(image: Image) -> list[dict]:
#         return my_pytorch_model.predict(image)   # your own model

#     clf = ImageClassificationAdapter(classify_fn=my_classifier)
#     result = clf.explain(pil_image)

#     # ── Plug in DALL-E instead of SDXL ──
#     def dalle_fn(prompt: str) -> Image:
#         resp = openai.images.generate(prompt=prompt, ...)
#         return download_image(resp.data[0].url)

#     gen = ImageGenerationAdapter(generate_fn=dalle_fn)
#     result = gen.explain("a red fox in snow")
# """

# from __future__ import annotations

# import io
# import os
# import re
# import json
# import time
# import base64
# import logging
# from dataclasses import dataclass, field, asdict
# from typing import Any

# import numpy as np
# from PIL import Image
# from huggingface_hub import InferenceClient

# logger = logging.getLogger(__name__)

# # ── Optional imports ──────────────────────────────────────────────────────────
# try:
#     from skimage.segmentation import slic
#     from skimage.util import img_as_float
#     _SKIMAGE_AVAILABLE = True
# except ImportError:
#     _SKIMAGE_AVAILABLE = False
#     logger.warning("scikit-image not installed. Install: pip install scikit-image")

# try:
#     import torch
#     import open_clip
#     _CLIP_AVAILABLE = True
# except ImportError:
#     _CLIP_AVAILABLE = False
#     logger.warning(
#         "open_clip not installed. CLIP alignment scoring disabled. "
#         "Install: pip install open-clip-torch"
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  CONFIG
# # ══════════════════════════════════════════════════════════════════════════════

# # ── HuggingFace token (used for both classification and generation) ───────────
# # Set HF_TOKEN in your .env file.  Free tier works for demos.
# HF_TOKEN             = os.getenv("HF_TOKEN", os.getenv("HF_API_TOKEN", ""))

# # ── Default models (used only by the built-in factory functions) ──────────────
# # These are defaults — you can pass any model to the factory functions.
# # The adapters themselves never reference these constants directly.
# CLF_MODEL            = "microsoft/resnet-50"
# GEN_MODEL            = "stabilityai/stable-diffusion-xl-base-1.0"
# NSFW_MODEL           = "Falconsai/nsfw_image_detection"

# # ── Generation settings ───────────────────────────────────────────────────────
# # GEN_SEED is critical for XAI correctness.
# # With a fixed seed, random noise is identical across all perturbation runs.
# # The only variable between baseline and masked runs is the missing word.
# # CLIP similarity differences therefore measure word influence, not noise.
# GEN_SEED             = 42
# GEN_STEPS            = 20       # lower = faster; 20 is good enough for XAI demo
# GEN_GUIDANCE         = 7.5

# # ── CLIP (runs locally — no API call) ────────────────────────────────────────
# CLIP_MODEL_NAME      = "ViT-B-32"
# CLIP_PRETRAINED      = "openai"

# # ── LIME settings ─────────────────────────────────────────────────────────────
# N_SUPERPIXELS        = 30      # image regions to segment into
# N_LIME_SAMPLES       = 8       # masked variants per explanation run
# LIME_TOP_K           = 8       # top regions to surface

# # ── Perturbation settings ─────────────────────────────────────────────────────
# MAX_PROMPT_WORDS     = 12      # max content words to perturb
# GEN_PERTURB_TOP_K    = 8

# # ── RAI thresholds ────────────────────────────────────────────────────────────
# CLIP_ALIGNMENT_WARN  = 0.20
# NSFW_THRESHOLD       = 0.70


# # ══════════════════════════════════════════════════════════════════════════════
# #  DATA CONTRACTS
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class RegionInfluence:
#     region_id:        int
#     influence_score:  float    # how much masking this region changed the prediction
#     positive:         bool     # True = region supports predicted label


# @dataclass
# class WordInfluence:
#     word:             str
#     position:         int
#     influence_score:  float    # CLIP similarity drop when word was masked


# @dataclass
# class RAIImageScorecard:
#     clip_alignment:       float   # 0–1, how well image matches label/prompt
#     alignment_flagged:    bool    # True if alignment < CLIP_ALIGNMENT_WARN
#     nsfw_score:           float   # 0–1, NSFW probability
#     nsfw_flagged:         bool    # True if nsfw_score > NSFW_THRESHOLD
#     overall_flagged:      bool


# @dataclass
# class ClassificationResult:
#     # Prediction
#     top_label:         str
#     top_score:         float
#     all_labels:        list[dict]   # [{"label": str, "score": float}, ...]

#     # XAI
#     region_influences: list[RegionInfluence]
#     heatmap_b64:       str          # base64 PNG of the LIME heatmap overlay

#     # RAI
#     rai:               RAIImageScorecard

#     # Meta
#     model_id:          str
#     method:            str = "LIME superpixel masking (black-box)"

#     def to_json(self, indent: int = 2) -> str:
#         d = asdict(self)
#         d.pop("heatmap_b64", None)   # exclude binary from JSON log
#         return json.dumps(d, indent=indent, default=str)


# @dataclass
# class GenerationResult:
#     # Output
#     prompt:            str
#     generated_image_b64: str        # base64 PNG of the generated image

#     # XAI
#     word_influences:   list[WordInfluence]
#     top_words:         list[str]    # most visually influential prompt words

#     # RAI
#     rai:               RAIImageScorecard

#     # Meta
#     model_id:          str
#     method:            str = "prompt token perturbation + CLIP similarity"

#     def to_json(self, indent: int = 2) -> str:
#         d = asdict(self)
#         d.pop("generated_image_b64", None)
#         return json.dumps(d, indent=indent, default=str)


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLIP ALIGNMENT ENGINE  (runs locally)
# # ══════════════════════════════════════════════════════════════════════════════

# class CLIPEngine:
#     """
#     Local CLIP model for:
#     1. Image–text alignment scoring (does this image match this label/prompt?)
#     2. Image–image similarity (how much did generation change after prompt perturbation?)
#     """

#     def __init__(self):
#         if not _CLIP_AVAILABLE:
#             self._model = None
#             logger.warning("CLIP unavailable — alignment scores will be skipped.")
#             return

#         logger.info("Loading CLIP %s/%s locally…", CLIP_MODEL_NAME, CLIP_PRETRAINED)
#         self._model, _, self._preprocess = open_clip.create_model_and_transforms(
#             CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
#         )
#         self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
#         self._model.eval()

#     def image_text_similarity(self, image: Image.Image, text: str) -> float:
#         """Returns cosine similarity between image and text embeddings."""
#         if not _CLIP_AVAILABLE or self._model is None:
#             return 0.5   # neutral fallback

#         img_tensor  = self._preprocess(image).unsqueeze(0)
#         text_tokens = self._tokenizer([text])

#         with torch.no_grad():
#             img_emb  = self._model.encode_image(img_tensor)
#             txt_emb  = self._model.encode_text(text_tokens)
#             img_emb  /= img_emb.norm(dim=-1, keepdim=True)
#             txt_emb  /= txt_emb.norm(dim=-1, keepdim=True)
#             sim = (img_emb @ txt_emb.T).item()

#         return round(float(sim), 4)

#     def image_image_similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
#         """Returns cosine similarity between two image embeddings."""
#         if not _CLIP_AVAILABLE or self._model is None:
#             return 0.8

#         def _embed(img):
#             t = self._preprocess(img).unsqueeze(0)
#             with torch.no_grad():
#                 e = self._model.encode_image(t)
#                 e /= e.norm(dim=-1, keepdim=True)
#             return e

#         sim = (_embed(img_a) @ _embed(img_b).T).item()
#         return round(float(sim), 4)


# # ══════════════════════════════════════════════════════════════════════════════
# #  HF INFERENCE API HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def _pil_to_b64(image: Image.Image) -> str:
#     buf = io.BytesIO()
#     image.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode()


# def _classify_image_api(image: Image.Image, model_id: str = CLF_MODEL) -> list[dict]:
#     """
#     Internal HF call using InferenceClient.
#     InferenceClient handles retries, queuing, and provider routing automatically.
#     Returns list of {"label": str, "score": float} sorted by score desc.
#     """
#     client = InferenceClient(api_key=HF_TOKEN or None)
#     results = client.image_classification(image, model=model_id)
#     # Normalise to consistent dict format regardless of HF SDK version
#     return [
#         {"label": r.label, "score": round(float(r.score), 4)}
#         for r in sorted(results, key=lambda x: x.score, reverse=True)
#     ]


# def make_hf_classify_fn(model_id: str = CLF_MODEL):
#     """
#     Factory — returns a classify_fn for HuggingFace Inference API.
#     Plug this into ImageClassificationAdapter(classify_fn=make_hf_classify_fn()).

#     To use a different model: make_hf_classify_fn("google/efficientnet-b7")
#     To use your own model entirely: write any function with this signature:
#         def my_fn(image: PIL.Image) -> list[dict[str, float]]:
#             ...returns [{"label": "cat", "score": 0.95}, ...]
#     """
#     def classify_fn(image: Image.Image) -> list[dict]:
#         return _classify_image_api(image, model_id)
#     classify_fn.__name__ = f"hf_classify({model_id})"
#     return classify_fn


# def _generate_image_api(prompt: str, model_id: str = GEN_MODEL) -> Image.Image:
#     """
#     Internal HF generation call using InferenceClient.
#     Uses a fixed seed — see GEN_SEED comment above for why this matters.
#     """
#     client = InferenceClient(api_key=HF_TOKEN or None)
#     image  = client.text_to_image(
#         prompt,
#         model=model_id,
#         seed=GEN_SEED,
#         num_inference_steps=GEN_STEPS,
#         guidance_scale=GEN_GUIDANCE,
#     )
#     return image  # InferenceClient already returns a PIL Image


# def make_hf_generate_fn(model_id: str = GEN_MODEL, seed: int = GEN_SEED):
#     """
#     Factory — returns a generate_fn for HuggingFace Inference API (SDXL).
#     Plug into ImageGenerationAdapter(generate_fn=make_hf_generate_fn()).

#     The seed parameter is critical for XAI — see _generate_image_api docstring.
#     To use DALL-E, Midjourney, or any other generator, write a function:
#         def my_fn(prompt: str) -> PIL.Image:
#             ...generate and return a PIL Image
#     and pass it directly: ImageGenerationAdapter(generate_fn=my_fn)
#     """
#     def generate_fn(prompt: str) -> Image.Image:
#         return _generate_image_api(prompt, model_id)
#     generate_fn.__name__ = f"hf_generate({model_id})"
#     return generate_fn


# def _classify_nsfw_api(image: Image.Image) -> float:
#     """
#     NSFW probability score via InferenceClient.
#     Returns 0.0 gracefully if the model is unavailable.
#     """
#     try:
#         client  = InferenceClient(api_key=HF_TOKEN or None)
#         results = client.image_classification(image, model=NSFW_MODEL)
#         for r in results:
#             if any(k in r.label.lower() for k in ("nsfw", "unsafe", "explicit")):
#                 return round(float(r.score), 4)
#         return 0.0
#     except Exception as e:
#         logger.warning("NSFW check failed: %s", e)
#         return 0.0


# # ══════════════════════════════════════════════════════════════════════════════
# #  SUB-PATH C1 — IMAGE CLASSIFICATION ADAPTER
# # ══════════════════════════════════════════════════════════════════════════════

# class ImageClassificationAdapter:
#     """
#     Black-box XAI for image classifiers via HuggingFace Inference API.

#     XAI method: LIME superpixel masking
#         - Segments image into N_SUPERPIXELS regions using SLIC algorithm
#         - Generates N_LIME_SAMPLES random masks (subsets of regions hidden)
#         - Sends each masked image to the classification API
#         - Measures how much each region's presence/absence affected the
#           top label's score → produces region importance ranking
#         - Overlays a heatmap on the original image

#     This is the standard black-box substitute for Grad-CAM.
#     Grad-CAM requires gradient access; LIME only needs input/output pairs.

#     Parameters
#     ----------
#     model_id : HuggingFace model ID for the classifier
#     """

#     def __init__(self, classify_fn=None, model_id: str = CLF_MODEL):
#         """
#         Parameters
#         ----------
#         classify_fn : Callable[[PIL.Image], list[dict]], optional
#             Function that takes a PIL Image and returns classification results
#             as [{"label": str, "score": float}, ...].
#             Defaults to HuggingFace Inference API with microsoft/resnet-50.
#             Pass any function here to use a completely different model.

#         model_id : str
#             Only used when classify_fn is None (builds the HF default).
#             Ignored if you supply your own classify_fn.
#         """
#         self._classify  = classify_fn or make_hf_classify_fn(model_id)
#         self.model_id   = getattr(self._classify, "__name__", model_id)
#         self._clip      = CLIPEngine()

#     def _segment_image(self, image: Image.Image) -> np.ndarray:
#         """
#         Segments image into superpixels using SLIC.
#         Returns integer array where each pixel's value is its region ID.
#         """
#         if not _SKIMAGE_AVAILABLE:
#             # Fallback: simple grid segmentation
#             arr    = np.array(image)
#             h, w   = arr.shape[:2]
#             grid_h = max(1, h // 8)
#             grid_w = max(1, w // 8)
#             segs   = np.zeros((h, w), dtype=int)
#             seg_id = 0
#             for i in range(0, h, grid_h):
#                 for j in range(0, w, grid_w):
#                     segs[i:i+grid_h, j:j+grid_w] = seg_id
#                     seg_id += 1
#             return segs

#         arr  = img_as_float(np.array(image.convert("RGB")))
#         segs = slic(arr, n_segments=N_SUPERPIXELS, compactness=10, sigma=1)
#         return segs

#     def _mask_image(
#         self, image: Image.Image, segments: np.ndarray,
#         active_segments: set[int], fill_color: tuple = (127, 127, 127)
#     ) -> Image.Image:
#         """Returns image with non-active segments filled with grey."""
#         arr    = np.array(image.convert("RGB")).copy()
#         mask   = np.isin(segments, list(active_segments))
#         arr[~mask] = fill_color
#         return Image.fromarray(arr)

#     def _build_heatmap(
#         self,
#         image: Image.Image,
#         segments: np.ndarray,
#         region_scores: dict[int, float],
#     ) -> Image.Image:
#         """
#         Overlays a colour heatmap on the original image.
#         Green = supports prediction, Red = opposes prediction.
#         """
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         import matplotlib.cm as cm

#         arr    = np.array(image.convert("RGB"))
#         heatmap = np.zeros(segments.shape, dtype=float)

#         scores = np.array(list(region_scores.values()))
#         if scores.max() != scores.min():
#             for seg_id, score in region_scores.items():
#                 norm = (score - scores.min()) / (scores.max() - scores.min())
#                 heatmap[segments == seg_id] = norm
#         else:
#             heatmap[:] = 0.5

#         fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#         ax.imshow(arr)
#         ax.imshow(heatmap, cmap="RdYlGn", alpha=0.5,
#                   vmin=0, vmax=1, interpolation="nearest")
#         ax.axis("off")
#         ax.set_title("LIME Region Importance\n(green = supports label, red = opposes)")

#         buf = io.BytesIO()
#         plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=100)
#         plt.close(fig)
#         buf.seek(0)
#         return Image.open(buf).convert("RGB")

#     def explain(self, image: Image.Image) -> ClassificationResult:
#         """
#         Full pipeline for one image:
#         1. Classify original → get baseline label + score
#         2. LIME masking loop → region influence scores
#         3. Build heatmap overlay
#         4. RAI: CLIP alignment + NSFW check
#         """
#         # Resize for API consistency
#         image = image.convert("RGB").resize((224, 224))

#         # ── Step 1: baseline classification ──────────────────────────────────
#         logger.info("Classifying original image…")
#         raw_labels = self._classify(image)
#         if not raw_labels:
#             raise RuntimeError("Classification API returned empty response.")

#         top_label = raw_labels[0]["label"]
#         top_score = float(raw_labels[0]["score"])
#         all_labels = [{"label": d["label"], "score": round(float(d["score"]), 4)}
#                       for d in raw_labels[:5]]

#         # ── Step 2: LIME superpixel masking ───────────────────────────────────
#         logger.info("Running LIME masking (%d samples)…", N_LIME_SAMPLES)
#         segments    = self._segment_image(image)
#         n_segments  = segments.max() + 1
#         all_seg_ids = set(range(n_segments))

#         # Accumulate weighted scores per region
#         region_scores: dict[int, float] = {i: 0.0 for i in range(n_segments)}
#         region_counts: dict[int, int]   = {i: 0   for i in range(n_segments)}

#         rng = np.random.default_rng(42)

#         for sample_idx in range(N_LIME_SAMPLES):
#             # Randomly choose which regions to show (50% on average)
#             active = set(
#                 i for i in range(n_segments)
#                 if rng.random() > 0.5
#             )
#             if not active:
#                 active = {0}

#             masked_img = self._mask_image(image, segments, active)

#             try:
#                 result     = self._classify(masked_img)
#                 # Score = probability of the TOP label in this masked version
#                 label_score = next(
#                     (d["score"] for d in result if d["label"] == top_label),
#                     0.0
#                 )
#             except Exception as e:
#                 logger.warning("LIME sample %d failed: %s", sample_idx, e)
#                 continue

#             # Credit active regions: higher label score = regions support label
#             for seg_id in active:
#                 region_scores[seg_id] += float(label_score)
#                 region_counts[seg_id] += 1

#         # Normalise by appearance count
#         influence_map: dict[int, float] = {}
#         for seg_id in range(n_segments):
#             if region_counts[seg_id] > 0:
#                 influence_map[seg_id] = region_scores[seg_id] / region_counts[seg_id]
#             else:
#                 influence_map[seg_id] = 0.0

#         # Normalise to [0, 1]
#         max_inf = max(influence_map.values()) or 1.0
#         min_inf = min(influence_map.values())
#         for k in influence_map:
#             influence_map[k] = (influence_map[k] - min_inf) / (max_inf - min_inf + 1e-9)

#         # Build RegionInfluence objects
#         sorted_regions = sorted(influence_map.items(), key=lambda x: x[1], reverse=True)
#         median_score   = float(np.median(list(influence_map.values())))
#         region_influences = [
#             RegionInfluence(
#                 region_id       = rid,
#                 influence_score = round(score, 4),
#                 positive        = score >= median_score,
#             )
#             for rid, score in sorted_regions[:LIME_TOP_K]
#         ]

#         # ── Step 3: build heatmap ─────────────────────────────────────────────
#         logger.info("Building heatmap…")
#         heatmap_img = self._build_heatmap(image, segments, influence_map)
#         heatmap_b64 = _pil_to_b64(heatmap_img)

#         # ── Step 4: RAI ───────────────────────────────────────────────────────
#         logger.info("Running RAI checks…")
#         clip_sim  = self._clip.image_text_similarity(image, top_label)
#         nsfw_score = _classify_nsfw_api(image)

#         rai = RAIImageScorecard(
#             clip_alignment    = clip_sim,
#             alignment_flagged = clip_sim < CLIP_ALIGNMENT_WARN,
#             nsfw_score        = nsfw_score,
#             nsfw_flagged      = nsfw_score > NSFW_THRESHOLD,
#             overall_flagged   = (clip_sim < CLIP_ALIGNMENT_WARN
#                                  or nsfw_score > NSFW_THRESHOLD),
#         )

#         return ClassificationResult(
#             top_label         = top_label,
#             top_score         = round(top_score, 4),
#             all_labels        = all_labels,
#             region_influences = region_influences,
#             heatmap_b64       = heatmap_b64,
#             rai               = rai,
#             model_id          = self.model_id,
#         )


# # ══════════════════════════════════════════════════════════════════════════════
# #  SUB-PATH C2 — IMAGE GENERATION ADAPTER
# # ══════════════════════════════════════════════════════════════════════════════

# class ImageGenerationAdapter:
#     """
#     Black-box XAI for SDXL image generation via HuggingFace Inference API.

#     Model: stabilityai/stable-diffusion-xl-base-1.0 (free HF tier)
#     No new API key — uses the same HF_API_TOKEN as classification.

#     XAI method: Prompt token perturbation + CLIP image similarity
#     ---------------------------------------------------------------
#     1. Generate baseline image from full prompt (fixed seed=42)
#     2. For each content word: mask it, regenerate with SAME seed
#     3. Measure CLIP image-image similarity: baseline vs perturbed
#     4. Words whose removal causes most visual drift = most influential

#     WHY THIS WORKS:
#     The fixed seed means random noise is identical across all runs.
#     The only variable between baseline and each perturbed run is the
#     missing word. CLIP similarity therefore measures purely how much
#     that word shaped the visual output — not noise variation.

#     RAI layer:
#     - CLIP text-image alignment: does the image match the prompt intent?
#     - NSFW detection: is the generated image safe?

#     Parameters
#     ----------
#     model_id : HuggingFace model ID (default: SDXL)
#     """

#     def __init__(self, generate_fn=None, model_id: str = GEN_MODEL):
#         """
#         Parameters
#         ----------
#         generate_fn : Callable[[str], PIL.Image], optional
#             Function that takes a prompt string and returns a PIL Image.
#             Defaults to HuggingFace SDXL Inference API.
#             Pass any function here to use DALL-E, Midjourney, local SD, etc.

#         model_id : str
#             Only used when generate_fn is None (builds the HF default).
#             Ignored if you supply your own generate_fn.

#         NOTE ON SEEDS: If you supply a custom generate_fn, make sure it uses
#         a fixed seed internally. Without this, perturbation results will be
#         noisy because image differences reflect randomness, not word influence.
#         """
#         self._generate = generate_fn or make_hf_generate_fn(model_id)
#         self.model_id  = getattr(self._generate, "__name__", model_id)
#         self._clip     = CLIPEngine()

#     @staticmethod
#     def _is_stopword(word: str) -> bool:
#         clean = re.sub(r"[^\w]", "", word.lower())
#         return clean in {
#             "a", "an", "the", "of", "in", "on", "at", "to", "and",
#             "or", "is", "are", "with", "by", "for", "from", "as",
#             "its", "it", "this", "that", "be", "was", "were",
#         }

#     def explain(self, prompt: str) -> GenerationResult:
#         """
#         Full pipeline for one prompt:
#         1. Generate baseline image
#         2. Perturb each content word → regenerate → measure CLIP drift
#         3. RAI: CLIP text-image alignment + NSFW check
#         """
#         # ── Step 1: baseline generation ───────────────────────────────────────
#         logger.info("Generating baseline image for prompt: '%s'", prompt[:60])
#         baseline_image = self._generate(prompt)

#         # ── Step 2: prompt word perturbation ─────────────────────────────────
#         words = prompt.split()
#         content_indices = [
#             i for i, w in enumerate(words)
#             if not self._is_stopword(w)
#         ][:MAX_PROMPT_WORDS]

#         logger.info(
#             "Perturbing %d content words in prompt…", len(content_indices)
#         )

#         word_influences: list[WordInfluence] = []

#         for idx in content_indices:
#             masked_words    = words.copy()
#             masked_words[idx] = "[MASK]"
#             masked_prompt   = " ".join(masked_words)

#             try:
#                 perturbed_image = self._generate(masked_prompt)
#                 sim  = self._clip.image_image_similarity(baseline_image, perturbed_image)
#                 drop = round(max(0.0, 1.0 - sim), 4)
#             except Exception as e:
#                 logger.warning("Generation failed for masked prompt: %s", e)
#                 continue

#             word_influences.append(WordInfluence(
#                 word            = words[idx],
#                 position        = idx,
#                 influence_score = drop,
#             ))

#         word_influences.sort(key=lambda x: x.influence_score, reverse=True)
#         top_words = [w.word for w in word_influences[:5]]

#         # ── Step 3: RAI ───────────────────────────────────────────────────────
#         logger.info("Running RAI checks…")
#         clip_align = self._clip.image_text_similarity(baseline_image, prompt)
#         nsfw_score = _classify_nsfw_api(baseline_image)

#         rai = RAIImageScorecard(
#             clip_alignment    = clip_align,
#             alignment_flagged = clip_align < CLIP_ALIGNMENT_WARN,
#             nsfw_score        = nsfw_score,
#             nsfw_flagged      = nsfw_score > NSFW_THRESHOLD,
#             overall_flagged   = (clip_align < CLIP_ALIGNMENT_WARN
#                                  or nsfw_score > NSFW_THRESHOLD),
#         )

#         return GenerationResult(
#             prompt               = prompt,
#             generated_image_b64  = _pil_to_b64(baseline_image),
#             word_influences      = word_influences[:GEN_PERTURB_TOP_K],
#             top_words            = top_words,
#             rai                  = rai,
#             model_id             = self.model_id,
#         )
from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Callable

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ================= CONFIG =================
LIME_NUM_SAMPLES = 200
LIME_BATCH_SIZE = 64
LIME_IMAGE_SIZE = (96, 96)
IMAGE_SIZE = (224, 224)

# ================= OCCLUSION (FIXED) =================
def occlusion_explain(image: np.ndarray, predict_fn, patch_size=48):
    h, w, _ = image.shape
    heatmap = np.zeros((h, w))

    baseline = predict_fn(image[np.newaxis])[0]
    top_class = np.argmax(baseline)
    baseline_prob = baseline[top_class]

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            occluded = image.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 0

            prob = predict_fn(occluded[np.newaxis])[0][top_class]
            impact = baseline_prob - prob

            heatmap[y:y+patch_size, x:x+patch_size] = impact

    # ✅ Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # ✅ Smooth
    heatmap = gaussian_filter(heatmap, sigma=3)

    # ✅ Threshold (remove noise)
    threshold = 0.4
    heatmap[heatmap < threshold] = 0

    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray):
    fig, ax = plt.subplots()

    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)

    ax.set_title("Occlusion Explanation (Red = Important)")
    ax.axis("off")

    return fig


# ================= ADAPTER =================
class ImageClassifierAdapter:

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], method="both"):
        self.predict_fn = predict_fn
        self.method = method
        self.batch_size = LIME_BATCH_SIZE
        self.explainer = LimeImageExplainer()

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(IMAGE_SIZE)
        return np.array(img).astype(np.float32) / 255.0

    def _predict_batch(self, images: np.ndarray) -> np.ndarray:
        return self.predict_fn(images)

    def _lime_predict(self, images_small: np.ndarray) -> np.ndarray:
        images_uint8 = (images_small * 255).astype(np.uint8)

        pil_batch = [Image.fromarray(img).resize(IMAGE_SIZE) for img in images_uint8]

        upscaled = np.stack([
            np.array(pil).astype(np.float32) / 255.0
            for pil in pil_batch
        ])

        return self._predict_batch(upscaled)

    def explain(self, image: Image.Image):
        img_arr = self._preprocess(image)

        results = {}

        # ================= OCCLUSION =================
        if self.method in ["occlusion", "both"]:
            occ_map = occlusion_explain(img_arr, self.predict_fn)
            occ_fig = overlay_heatmap(img_arr, occ_map)

            results["occlusion_map"] = occ_map
            results["occlusion_fig"] = occ_fig

        # ================= LIME =================
        if self.method in ["lime", "both"]:
            image_uint8 = (img_arr * 255).astype(np.uint8)

            lime_arr = np.array(
                Image.fromarray(image_uint8).resize(LIME_IMAGE_SIZE)
            ).astype(np.float32) / 255.0

            lime_exp = self.explainer.explain_instance(
                lime_arr,
                classifier_fn=self._lime_predict,
                top_labels=1,
                hide_color=0,
                num_samples=LIME_NUM_SAMPLES,
                batch_size=self.batch_size,
                segmentation_fn=lambda x: slic(x, n_segments=30, compactness=10),
            )

            results["lime"] = lime_exp

        return results


# ================= FIXED HF FACTORY =================
from transformers import pipeline as hf_pipeline

def make_hf_pipeline_predict_fn(model_id: str = "microsoft/resnet-50"):
    pipe = hf_pipeline("image-classification", model=model_id, top_k=None)

    labels = list(pipe.model.config.id2label.values())
    label_idx = {l: i for i, l in enumerate(labels)}

    def predict_fn(images: np.ndarray) -> np.ndarray:
        pil_batch = [
            Image.fromarray((img * 255).astype(np.uint8))
            for img in images
        ]

        batch_preds = pipe(pil_batch)

        results = []
        for preds in batch_preds:
            row = np.zeros(len(labels), dtype=np.float32)
            for p in preds:
                idx = label_idx.get(p["label"])
                if idx is not None:
                    row[idx] = float(p["score"])
            results.append(row)

        return np.array(results)

    return predict_fn, labels