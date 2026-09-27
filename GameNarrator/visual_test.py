"""Prints CLIP's live top-5 scene matches, for tuning scene prompts."""
import os
import sys
import time
import json

import mss
import numpy as np
import torch
from PIL import Image

import capture
import perception

CAPTURE_INTERVAL = 1.0
GAME_CONFIG = sys.argv[1] if len(sys.argv) > 1 else "games/cs2.json"

with open(GAME_CONFIG, "r") as f:
    cfg = json.load(f)
scene_prompts = cfg["scene_prompts"]
capture_cfg = cfg.get("capture", {"mode": "monitor", "fallback_monitor": 1})
print(f"Loaded config: {cfg['game']}")
print(f"Loaded {len(scene_prompts)} scene prompts.\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP ViT-B/32 on {device}")
model, preprocess, text_features = perception.load_clip(scene_prompts, device)

sct = mss.mss()

print("Watching screen\n")
try:
    while True:
        region = capture.resolve_region(sct, capture_cfg, cfg["game"])
        frame_rgb = capture.grab_frame(sct, region)

        if not os.path.exists("clip_view.png"):
            # Save what CLIP is fed, once, so the capture region can be checked by eye.
            # A wrong region looks identical to a bad prompt in the numbers.
            Image.fromarray(frame_rgb).resize((224, 224)).save("clip_view.png")
            print(f"Wrote clip_view.png ({region['width']}x{region['height']} source), this is what CLIP sees\n")

        probs = perception.scene_probs(model, preprocess, text_features, frame_rgb, device)

        print("Top scenes:")
        for i in np.argsort(probs)[::-1][:5]:
            print(f"  {scene_prompts[i]:<55} {probs[i]:.3f}")
        print("-" * 40)
        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopped.")
