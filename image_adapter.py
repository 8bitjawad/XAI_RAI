"""
image_adapter.py  —  Path C: Image XAI + RAI Middleware
=========================================================
Covers two sub-paths:

SUB-PATH C1 — Image Classification  (e.g. ResNet via HF Inference API)
    XAI: LIME superpixel masking — divides image into regions, masks each,
         sends to API, measures which regions drove the label prediction.
         This is the correct black-box substitute for Grad-CAM when you
         have no access to model gradients over an API.
    RAI: CLIP alignment score (does the label match the image semantically?)
         + NSFW detection via a dedicated HF classifier.

SUB-PATH C2 — Image Generation  (e.g. Stable Diffusion via HF Inference API)
    XAI: Prompt token perturbation — masks each word in the prompt,
         regenerates the image, measures CLIP similarity between original
         and perturbed image. Words that cause the most visual drift are
         most influential.
    RAI: CLIP alignment (does the generated image match the prompt intent?)
         + NSFW detection on the generated image.

CLIP runs locally (all-MiniLM style via open_clip / transformers).
It is small (~400 MB) and free — no API needed.

All HuggingFace Inference API calls use the free tier by default.
Set HF_API_TOKEN in .env to access gated models or higher rate limits.

Usage:
    from image_adapter import ImageClassificationAdapter, ImageGenerationAdapter

    # Classification
    clf = ImageClassificationAdapter()
    result = clf.explain(pil_image)
    print(result.to_json())

    # Generation
    gen = ImageGenerationAdapter()
    result = gen.explain("a red fox sitting in a snowy forest")
    print(result.to_json())
"""

from __future__ import annotations

import io
import os
import re
import json
import time
import base64
import logging
import requests
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from skimage.segmentation import slic
    from skimage.util import img_as_float
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not installed. Install: pip install scikit-image")

try:
    import torch
    import open_clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    logger.warning(
        "open_clip not installed. CLIP alignment scoring disabled. "
        "Install: pip install open-clip-torch"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

HF_API_BASE          = "https://api-inference.huggingface.co/models"
HF_API_TOKEN         = os.getenv("HF_API_TOKEN", "")

# Classification model — publicly available, no token required for free tier
CLF_MODEL            = "google/vit-base-patch16-224"

# Generation model — free tier, no token required (but slow without token)
GEN_MODEL            = "stabilityai/stable-diffusion-2-1"

# NSFW classifier — runs on generated/uploaded images
NSFW_MODEL           = "Falconsai/nsfw_image_detection"

# CLIP model for alignment scoring (runs locally, no API)
CLIP_MODEL_NAME      = "ViT-B-32"
CLIP_PRETRAINED      = "openai"

# LIME superpixel settings
N_SUPERPIXELS        = 50     # number of regions to segment image into
N_LIME_SAMPLES       = 20     # number of masked variants to send to API
                               # higher = more accurate, more API calls
LIME_TOP_K           = 8      # top K influential regions to highlight

# Generation perturbation
MAX_PROMPT_WORDS     = 12     # cap prompt words to perturb (API cost control)
GEN_PERTURB_TOP_K    = 8

# RAI thresholds
CLIP_ALIGNMENT_WARN  = 0.20   # flag if CLIP similarity below this
NSFW_THRESHOLD       = 0.70   # flag if NSFW score above this

# Rate limiting
API_CALL_DELAY       = 0.4    # seconds between HF API calls


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegionInfluence:
    region_id:        int
    influence_score:  float    # how much masking this region changed the prediction
    positive:         bool     # True = region supports predicted label


@dataclass
class WordInfluence:
    word:             str
    position:         int
    influence_score:  float    # CLIP similarity drop when word was masked


@dataclass
class RAIImageScorecard:
    clip_alignment:       float   # 0–1, how well image matches label/prompt
    alignment_flagged:    bool    # True if alignment < CLIP_ALIGNMENT_WARN
    nsfw_score:           float   # 0–1, NSFW probability
    nsfw_flagged:         bool    # True if nsfw_score > NSFW_THRESHOLD
    overall_flagged:      bool


@dataclass
class ClassificationResult:
    # Prediction
    top_label:         str
    top_score:         float
    all_labels:        list[dict]   # [{"label": str, "score": float}, ...]

    # XAI
    region_influences: list[RegionInfluence]
    heatmap_b64:       str          # base64 PNG of the LIME heatmap overlay

    # RAI
    rai:               RAIImageScorecard

    # Meta
    model_id:          str
    method:            str = "LIME superpixel masking (black-box)"

    def to_json(self, indent: int = 2) -> str:
        d = asdict(self)
        d.pop("heatmap_b64", None)   # exclude binary from JSON log
        return json.dumps(d, indent=indent, default=str)


@dataclass
class GenerationResult:
    # Output
    prompt:            str
    generated_image_b64: str        # base64 PNG of the generated image

    # XAI
    word_influences:   list[WordInfluence]
    top_words:         list[str]    # most visually influential prompt words

    # RAI
    rai:               RAIImageScorecard

    # Meta
    model_id:          str
    method:            str = "prompt token perturbation + CLIP similarity"

    def to_json(self, indent: int = 2) -> str:
        d = asdict(self)
        d.pop("generated_image_b64", None)
        return json.dumps(d, indent=indent, default=str)


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP ALIGNMENT ENGINE  (runs locally)
# ══════════════════════════════════════════════════════════════════════════════

class CLIPEngine:
    """
    Local CLIP model for:
    1. Image–text alignment scoring (does this image match this label/prompt?)
    2. Image–image similarity (how much did generation change after prompt perturbation?)
    """

    def __init__(self):
        if not _CLIP_AVAILABLE:
            self._model = None
            logger.warning("CLIP unavailable — alignment scores will be skipped.")
            return

        logger.info("Loading CLIP %s/%s locally…", CLIP_MODEL_NAME, CLIP_PRETRAINED)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self._model.eval()

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Returns cosine similarity between image and text embeddings."""
        if not _CLIP_AVAILABLE or self._model is None:
            return 0.5   # neutral fallback

        img_tensor  = self._preprocess(image).unsqueeze(0)
        text_tokens = self._tokenizer([text])

        with torch.no_grad():
            img_emb  = self._model.encode_image(img_tensor)
            txt_emb  = self._model.encode_text(text_tokens)
            img_emb  /= img_emb.norm(dim=-1, keepdim=True)
            txt_emb  /= txt_emb.norm(dim=-1, keepdim=True)
            sim = (img_emb @ txt_emb.T).item()

        return round(float(sim), 4)

    def image_image_similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
        """Returns cosine similarity between two image embeddings."""
        if not _CLIP_AVAILABLE or self._model is None:
            return 0.8

        def _embed(img):
            t = self._preprocess(img).unsqueeze(0)
            with torch.no_grad():
                e = self._model.encode_image(t)
                e /= e.norm(dim=-1, keepdim=True)
            return e

        sim = (_embed(img_a) @ _embed(img_b).T).item()
        return round(float(sim), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  HF INFERENCE API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hf_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


def _pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _classify_image_api(image: Image.Image, model_id: str = CLF_MODEL) -> list[dict]:
    """
    Calls HF Inference API for image classification.
    Returns list of {"label": str, "score": float} sorted by score desc.
    """
    time.sleep(API_CALL_DELAY)
    url  = f"{HF_API_BASE}/{model_id}"
    resp = requests.post(
        url,
        headers=_hf_headers(),
        data=_pil_to_bytes(image),
        timeout=30,
    )
    if resp.status_code == 503:
        # Model loading — wait and retry once
        logger.info("Model loading on HF, retrying in 10s…")
        time.sleep(10)
        resp = requests.post(url, headers=_hf_headers(),
                             data=_pil_to_bytes(image), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _generate_image_api(prompt: str, model_id: str = GEN_MODEL) -> Image.Image:
    """
    Calls HF Inference API for text-to-image generation.
    Returns a PIL Image.
    """
    time.sleep(API_CALL_DELAY)
    url  = f"{HF_API_BASE}/{model_id}"
    resp = requests.post(
        url,
        headers={**_hf_headers(), "Content-Type": "application/json"},
        json={"inputs": prompt},
        timeout=120,   # generation is slow
    )
    if resp.status_code == 503:
        logger.info("Generation model loading, retrying in 20s…")
        time.sleep(20)
        resp = requests.post(
            url,
            headers={**_hf_headers(), "Content-Type": "application/json"},
            json={"inputs": prompt},
            timeout=120,
        )
    resp.raise_for_status()
    return _bytes_to_pil(resp.content)


def _classify_nsfw_api(image: Image.Image) -> float:
    """Returns NSFW probability score (0–1)."""
    time.sleep(API_CALL_DELAY)
    try:
        url  = f"{HF_API_BASE}/{NSFW_MODEL}"
        resp = requests.post(
            url, headers=_hf_headers(),
            data=_pil_to_bytes(image), timeout=30
        )
        if resp.status_code != 200:
            return 0.0
        results = resp.json()
        # Find the NSFW / unsafe label score
        for item in results:
            if isinstance(item, list):
                item = item[0]
            label = str(item.get("label", "")).lower()
            if any(k in label for k in ("nsfw", "unsafe", "explicit")):
                return float(item.get("score", 0.0))
        return 0.0
    except Exception as e:
        logger.warning("NSFW check failed: %s", e)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SUB-PATH C1 — IMAGE CLASSIFICATION ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class ImageClassificationAdapter:
    """
    Black-box XAI for image classifiers via HuggingFace Inference API.

    XAI method: LIME superpixel masking
        - Segments image into N_SUPERPIXELS regions using SLIC algorithm
        - Generates N_LIME_SAMPLES random masks (subsets of regions hidden)
        - Sends each masked image to the classification API
        - Measures how much each region's presence/absence affected the
          top label's score → produces region importance ranking
        - Overlays a heatmap on the original image

    This is the standard black-box substitute for Grad-CAM.
    Grad-CAM requires gradient access; LIME only needs input/output pairs.

    Parameters
    ----------
    model_id : HuggingFace model ID for the classifier
    """

    def __init__(self, model_id: str = CLF_MODEL):
        self.model_id   = model_id
        self._clip      = CLIPEngine()

    def _segment_image(self, image: Image.Image) -> np.ndarray:
        """
        Segments image into superpixels using SLIC.
        Returns integer array where each pixel's value is its region ID.
        """
        if not _SKIMAGE_AVAILABLE:
            # Fallback: simple grid segmentation
            arr    = np.array(image)
            h, w   = arr.shape[:2]
            grid_h = max(1, h // 8)
            grid_w = max(1, w // 8)
            segs   = np.zeros((h, w), dtype=int)
            seg_id = 0
            for i in range(0, h, grid_h):
                for j in range(0, w, grid_w):
                    segs[i:i+grid_h, j:j+grid_w] = seg_id
                    seg_id += 1
            return segs

        arr  = img_as_float(np.array(image.convert("RGB")))
        segs = slic(arr, n_segments=N_SUPERPIXELS, compactness=10, sigma=1)
        return segs

    def _mask_image(
        self, image: Image.Image, segments: np.ndarray,
        active_segments: set[int], fill_color: tuple = (127, 127, 127)
    ) -> Image.Image:
        """Returns image with non-active segments filled with grey."""
        arr    = np.array(image.convert("RGB")).copy()
        mask   = np.isin(segments, list(active_segments))
        arr[~mask] = fill_color
        return Image.fromarray(arr)

    def _build_heatmap(
        self,
        image: Image.Image,
        segments: np.ndarray,
        region_scores: dict[int, float],
    ) -> Image.Image:
        """
        Overlays a colour heatmap on the original image.
        Green = supports prediction, Red = opposes prediction.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        arr    = np.array(image.convert("RGB"))
        heatmap = np.zeros(segments.shape, dtype=float)

        scores = np.array(list(region_scores.values()))
        if scores.max() != scores.min():
            for seg_id, score in region_scores.items():
                norm = (score - scores.min()) / (scores.max() - scores.min())
                heatmap[segments == seg_id] = norm
        else:
            heatmap[:] = 0.5

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(arr)
        ax.imshow(heatmap, cmap="RdYlGn", alpha=0.5,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        ax.set_title("LIME Region Importance\n(green = supports label, red = opposes)")

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def explain(self, image: Image.Image) -> ClassificationResult:
        """
        Full pipeline for one image:
        1. Classify original → get baseline label + score
        2. LIME masking loop → region influence scores
        3. Build heatmap overlay
        4. RAI: CLIP alignment + NSFW check
        """
        # Resize for API consistency
        image = image.convert("RGB").resize((224, 224))

        # ── Step 1: baseline classification ──────────────────────────────────
        logger.info("Classifying original image…")
        raw_labels = _classify_image_api(image, self.model_id)
        if not raw_labels:
            raise RuntimeError("Classification API returned empty response.")

        top_label = raw_labels[0]["label"]
        top_score = float(raw_labels[0]["score"])
        all_labels = [{"label": d["label"], "score": round(float(d["score"]), 4)}
                      for d in raw_labels[:5]]

        # ── Step 2: LIME superpixel masking ───────────────────────────────────
        logger.info("Running LIME masking (%d samples)…", N_LIME_SAMPLES)
        segments    = self._segment_image(image)
        n_segments  = segments.max() + 1
        all_seg_ids = set(range(n_segments))

        # Accumulate weighted scores per region
        region_scores: dict[int, float] = {i: 0.0 for i in range(n_segments)}
        region_counts: dict[int, int]   = {i: 0   for i in range(n_segments)}

        rng = np.random.default_rng(42)

        for sample_idx in range(N_LIME_SAMPLES):
            # Randomly choose which regions to show (50% on average)
            active = set(
                i for i in range(n_segments)
                if rng.random() > 0.5
            )
            if not active:
                active = {0}

            masked_img = self._mask_image(image, segments, active)

            try:
                result     = _classify_image_api(masked_img, self.model_id)
                # Score = probability of the TOP label in this masked version
                label_score = next(
                    (d["score"] for d in result if d["label"] == top_label),
                    0.0
                )
            except Exception as e:
                logger.warning("LIME sample %d failed: %s", sample_idx, e)
                continue

            # Credit active regions: higher label score = regions support label
            for seg_id in active:
                region_scores[seg_id] += float(label_score)
                region_counts[seg_id] += 1

        # Normalise by appearance count
        influence_map: dict[int, float] = {}
        for seg_id in range(n_segments):
            if region_counts[seg_id] > 0:
                influence_map[seg_id] = region_scores[seg_id] / region_counts[seg_id]
            else:
                influence_map[seg_id] = 0.0

        # Normalise to [0, 1]
        max_inf = max(influence_map.values()) or 1.0
        min_inf = min(influence_map.values())
        for k in influence_map:
            influence_map[k] = (influence_map[k] - min_inf) / (max_inf - min_inf + 1e-9)

        # Build RegionInfluence objects
        sorted_regions = sorted(influence_map.items(), key=lambda x: x[1], reverse=True)
        median_score   = float(np.median(list(influence_map.values())))
        region_influences = [
            RegionInfluence(
                region_id       = rid,
                influence_score = round(score, 4),
                positive        = score >= median_score,
            )
            for rid, score in sorted_regions[:LIME_TOP_K]
        ]

        # ── Step 3: build heatmap ─────────────────────────────────────────────
        logger.info("Building heatmap…")
        heatmap_img = self._build_heatmap(image, segments, influence_map)
        heatmap_b64 = _pil_to_b64(heatmap_img)

        # ── Step 4: RAI ───────────────────────────────────────────────────────
        logger.info("Running RAI checks…")
        clip_sim  = self._clip.image_text_similarity(image, top_label)
        nsfw_score = _classify_nsfw_api(image)

        rai = RAIImageScorecard(
            clip_alignment    = clip_sim,
            alignment_flagged = clip_sim < CLIP_ALIGNMENT_WARN,
            nsfw_score        = nsfw_score,
            nsfw_flagged      = nsfw_score > NSFW_THRESHOLD,
            overall_flagged   = (clip_sim < CLIP_ALIGNMENT_WARN
                                 or nsfw_score > NSFW_THRESHOLD),
        )

        return ClassificationResult(
            top_label         = top_label,
            top_score         = round(top_score, 4),
            all_labels        = all_labels,
            region_influences = region_influences,
            heatmap_b64       = heatmap_b64,
            rai               = rai,
            model_id          = self.model_id,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SUB-PATH C2 — IMAGE GENERATION ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class ImageGenerationAdapter:
    """
    Black-box XAI for text-to-image generation via HuggingFace Inference API.

    XAI method: Prompt token perturbation + CLIP similarity
        - Generates a baseline image from the full prompt
        - Masks each content word in the prompt one at a time
        - Regenerates image for each masked prompt
        - Measures CLIP image-image similarity between baseline and perturbed
        - Words whose removal causes the biggest visual drift are most influential

    This is the generative-image equivalent of the text perturbation in Path B.

    Parameters
    ----------
    model_id : HuggingFace model ID for the image generator
    """

    def __init__(self, model_id: str = GEN_MODEL):
        self.model_id = model_id
        self._clip    = CLIPEngine()

    @staticmethod
    def _is_stopword(word: str) -> bool:
        clean = re.sub(r"[^\w]", "", word.lower())
        return clean in {
            "a", "an", "the", "of", "in", "on", "at", "to", "and",
            "or", "is", "are", "with", "by", "for", "from", "as",
            "its", "it", "this", "that", "be", "was", "were",
        }

    def explain(self, prompt: str) -> GenerationResult:
        """
        Full pipeline for one prompt:
        1. Generate baseline image
        2. Perturb each content word → regenerate → measure CLIP drift
        3. RAI: CLIP text-image alignment + NSFW check
        """
        # ── Step 1: baseline generation ───────────────────────────────────────
        logger.info("Generating baseline image for prompt: '%s'", prompt[:60])
        baseline_image = _generate_image_api(prompt, self.model_id)

        # ── Step 2: prompt word perturbation ─────────────────────────────────
        words = prompt.split()
        content_indices = [
            i for i, w in enumerate(words)
            if not self._is_stopword(w)
        ][:MAX_PROMPT_WORDS]

        logger.info(
            "Perturbing %d content words in prompt…", len(content_indices)
        )

        word_influences: list[WordInfluence] = []

        for idx in content_indices:
            masked_words    = words.copy()
            masked_words[idx] = "[MASK]"
            masked_prompt   = " ".join(masked_words)

            try:
                perturbed_image = _generate_image_api(masked_prompt, self.model_id)
                sim  = self._clip.image_image_similarity(baseline_image, perturbed_image)
                drop = round(max(0.0, 1.0 - sim), 4)
            except Exception as e:
                logger.warning("Generation failed for masked prompt: %s", e)
                continue

            word_influences.append(WordInfluence(
                word            = words[idx],
                position        = idx,
                influence_score = drop,
            ))

        word_influences.sort(key=lambda x: x.influence_score, reverse=True)
        top_words = [w.word for w in word_influences[:5]]

        # ── Step 3: RAI ───────────────────────────────────────────────────────
        logger.info("Running RAI checks…")
        clip_align = self._clip.image_text_similarity(baseline_image, prompt)
        nsfw_score = _classify_nsfw_api(baseline_image)

        rai = RAIImageScorecard(
            clip_alignment    = clip_align,
            alignment_flagged = clip_align < CLIP_ALIGNMENT_WARN,
            nsfw_score        = nsfw_score,
            nsfw_flagged      = nsfw_score > NSFW_THRESHOLD,
            overall_flagged   = (clip_align < CLIP_ALIGNMENT_WARN
                                 or nsfw_score > NSFW_THRESHOLD),
        )

        return GenerationResult(
            prompt               = prompt,
            generated_image_b64  = _pil_to_b64(baseline_image),
            word_influences      = word_influences[:GEN_PERTURB_TOP_K],
            top_words            = top_words,
            rai                  = rai,
            model_id             = self.model_id,
        )