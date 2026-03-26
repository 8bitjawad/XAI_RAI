from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Callable

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic

# ================= CONFIG =================
LIME_NUM_SAMPLES = 150       # was 200 — still clear, faster
LIME_BATCH_SIZE  = 128       # was 64 — fewer model calls
LIME_IMAGE_SIZE  = (112, 112)
IMAGE_SIZE       = (224, 224)
OCCLUSION_PATCH  = 28        # was 32 — finer grid, still fast because batched


# ================= OCCLUSION (BATCHED) =================

def occlusion_explain(image: np.ndarray, predict_fn: Callable, patch_size: int = OCCLUSION_PATCH) -> np.ndarray:
    """
    Batched occlusion sensitivity map.

    Builds ALL occluded variants at once and calls predict_fn ONCE,
    instead of calling it once per patch (the old sequential loop).

    For a 224×224 image with patch_size=28: (224/28)² = 64 patches
    → 64 images in ONE predict_fn call vs 64 separate calls before.
    """
    h, w, _ = image.shape

    # Baseline — single call
    baseline_probs = predict_fn(image[np.newaxis])[0]
    top_class      = int(np.argmax(baseline_probs))
    baseline_score = float(baseline_probs[top_class])

    # Build every occluded variant in one pass
    patches = []
    coords  = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            occluded = image.copy()
            occluded[y : y + patch_size, x : x + patch_size] = 0.0
            patches.append(occluded)
            coords.append((y, x))

    # ONE batched call
    all_probs = predict_fn(np.stack(patches))  # (N, num_classes)
    scores    = all_probs[:, top_class]        # (N,)

    # Fill heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    for (y, x), score in zip(coords, scores):
        heatmap[y : y + patch_size, x : x + patch_size] = baseline_score - score

    return heatmap, top_class


def build_occlusion_fig(image: np.ndarray, heatmap: np.ndarray):
    """
    Renders occlusion heatmap. Normalises heatmap so low-contrast images
    still produce a visible overlay (fixes the invisible-heatmap bug).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Normalise to [0, 1] so the colour range is always meaningful
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        norm_heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        norm_heatmap = np.zeros_like(heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    im = axes[1].imshow(norm_heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("Occlusion Map\n(bright = important region)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ================= ADAPTER =================

class ImageClassifierAdapter:
    """
    Model-agnostic XAI wrapper.

    predict_fn(images: np.ndarray) -> np.ndarray
        Input : (N, H, W, 3) float32 in [0, 1]
        Output: (N, num_classes) probabilities

    method : "occlusion" | "lime" | "both"
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        method: str = "both",
    ):
        self.predict_fn = predict_fn
        self.method     = method
        self.batch_size = LIME_BATCH_SIZE
        self.explainer  = LimeImageExplainer()

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(IMAGE_SIZE)
        return np.array(img).astype(np.float32) / 255.0

    def _predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Calls predict_fn in batches to handle large LIME sample counts."""
        results = []
        for i in range(0, len(images), self.batch_size):
            results.append(self.predict_fn(images[i : i + self.batch_size]))
        return np.vstack(results)

    def _lime_predict(self, images_small: np.ndarray) -> np.ndarray:
        """
        Upscales LIME's downsampled perturbations back to IMAGE_SIZE
        before calling the real model. Done in one vectorised stack.
        """
        upscaled = np.stack([
            np.array(
                Image.fromarray((img * 255).astype(np.uint8)).resize(IMAGE_SIZE)
            ).astype(np.float32) / 255.0
            for img in images_small
        ])
        return self._predict_batch(upscaled)

    def explain(self, image: Image.Image) -> dict:
        """
        Returns a dict with keys depending on `method`:
            "occlusion_fig"  — matplotlib Figure
            "occlusion_map"  — raw float32 heatmap (H, W)
            "occlusion_class"— int index of top class used for occlusion
            "lime"           — LimeImageExplanation object
            "top_idx"        — top predicted class index (for LIME rendering)
        """
        img_arr = self._preprocess(image)
        results: dict = {}

        # ── Occlusion ────────────────────────────────────────────────────────
        if self.method in ("occlusion", "both"):
            occ_map, top_class = occlusion_explain(img_arr, self._predict_batch)
            results["occlusion_map"]   = occ_map
            results["occlusion_class"] = top_class
            results["occlusion_fig"]   = build_occlusion_fig(img_arr, occ_map)

        # ── LIME ─────────────────────────────────────────────────────────────
        if self.method in ("lime", "both"):
            image_uint8 = (img_arr * 255).astype(np.uint8)
            lime_arr    = np.array(
                Image.fromarray(image_uint8).resize(LIME_IMAGE_SIZE)
            ).astype(np.float32) / 255.0

            # Baseline at full res to get the true top_idx
            probs   = self._predict_batch(img_arr[np.newaxis])[0]
            top_idx = int(np.argmax(probs))

            lime_exp = self.explainer.explain_instance(
                lime_arr,
                classifier_fn=self._lime_predict,
                top_labels=5,                      # explain top-5 so top_idx is always present
                hide_color=0,
                num_samples=LIME_NUM_SAMPLES,
                batch_size=self.batch_size,
                random_seed=42,
                segmentation_fn=lambda x: slic(
                    x, n_segments=50, compactness=10, sigma=1  # was 30 — finer, more accurate
                ),
            )

            results["lime"]    = lime_exp
            results["top_idx"] = top_idx  # ← pass the REAL top idx to app.py

        return results


# ================= HF BATCH PREDICT_FN =================

def make_hf_pipeline_predict_fn(model_id: str = "microsoft/resnet-50"):
    """
    Returns (predict_fn, class_names) for any HuggingFace image-classification model.
    predict_fn processes the whole batch in a single pipeline forward pass.
    """
    from transformers import pipeline as hf_pipeline

    pipe      = hf_pipeline("image-classification", model=model_id, top_k=None)
    labels    = list(pipe.model.config.id2label.values())
    label_idx = {l: i for i, l in enumerate(labels)}

    def predict_fn(images: np.ndarray) -> np.ndarray:
        pil_batch   = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        batch_preds = pipe(pil_batch)   # single forward pass

        results = []
        for preds in batch_preds:
            row = np.zeros(len(labels), dtype=np.float32)
            for p in preds:
                idx = label_idx.get(p["label"])
                if idx is not None:
                    row[idx] = float(p["score"])
            results.append(row)
        return np.array(results)

    predict_fn.__name__ = f"hf_pipeline({model_id})"
    return predict_fn, labels