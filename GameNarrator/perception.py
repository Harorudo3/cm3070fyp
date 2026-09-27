"""Perception stage. Loads YAMNet and CLIP and classifies audio and frames."""
import csv
import os

import numpy as np
import torch
import clip
from PIL import Image


def load_yamnet(cache_dir):
    """Load YAMNet and its class list. Returns (model, class_names, name_to_idx)."""
    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    import tensorflow as tf
    import tensorflow_hub as hub
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = [row["display_name"] for row in csv.DictReader(tf.io.gfile.GFile(class_map_path))]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    return yamnet, class_names, name_to_idx


def unknown_classes(events_cfg, name_to_idx):
    """Config class names that do not exist in YAMNet.

    A typo is skipped silently at runtime, so it looks the same as a sound that
    never happened. Checked at startup instead.
    """
    return [f"{ev}: {c}" for ev, spec in events_cfg.items() if isinstance(spec, dict)
            for key in ("any", "require_also", "reject_if_above")
            for c in spec.get(key, []) if c not in name_to_idx]


def classify_audio(scores, name_to_idx, events_cfg, priority, aggregation="mean"):
    """Map YAMNet scores to a game event.

    Returns (event, description, confidence) for the highest-priority match, or
    None. Use aggregation="max" for short impulsive sounds, "mean" for
    continuous ones.
    """
    frame_scores = scores.max(axis=0) if aggregation == "max" else scores.mean(axis=0)

    def score_of(name):
        idx = name_to_idx.get(name)
        return frame_scores[idx] if idx is not None else 0.0

    for event in priority:
        spec = events_cfg[event]
        conf = max((score_of(c) for c in spec["any"]), default=0.0)
        if conf < spec["threshold"]:
            continue
        if "require_also" in spec:
            also = max((score_of(c) for c in spec["require_also"]), default=0.0)
            if also < spec.get("also_threshold", 0.4):
                continue
        # Quiet games leave weak impact classes in windows that are really
        # silence, so a veto class can rule the match out.
        if any(score_of(c) >= limit for c, limit in spec.get("reject_if_above", {}).items()):
            continue
        return event, spec["description"], float(conf)
    return None


def load_clip(scene_prompts, device):
    """Load CLIP and pre-encode the scene prompts, which do not change."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(scene_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features


def scene_probs(model, preprocess, text_features, frame_rgb, device):
    """Probability for every scene prompt. Used when tuning prompts."""
    image = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(image)
        feats /= feats.norm(dim=-1, keepdim=True)
        return (100.0 * feats @ text_features.T).softmax(dim=-1).cpu().numpy()[0]


def classify_scene(model, preprocess, text_features, scene_prompts, frame_rgb, device,
                   min_conf, margin):
    """Top scene match. Returns (scene, confidence, is_confident).

    is_confident requires both a confidence floor and a gap to the runner-up.
    """
    probs = scene_probs(model, preprocess, text_features, frame_rgb, device)
    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]
    confident = bool(probs[top] >= min_conf and (probs[top] - probs[second]) >= margin)
    return scene_prompts[top], float(probs[top]), confident
