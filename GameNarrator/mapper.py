import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import numpy as np
import resampy
import soundcard as sc
import tensorflow as tf
import tensorflow_hub as hub
import csv
import warnings
from soundcard import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

# Load YAMNet + class names
print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
class_names = []
with tf.io.gfile.GFile(class_map_path) as f:
    for row in csv.DictReader(f):
        class_names.append(row["display_name"])
name_to_index = {n: i for i, n in enumerate(class_names)}
print(f"Loaded {len(class_names)} classes.")

# Load game config
GAME_CONFIG = "games/cs2.json"
with open(GAME_CONFIG) as f:
    config = json.load(f)
print(f"Loaded config: {config['game']}\n")


def classify_event(mean_scores):
    """Return the highest-priority game event whose rule is satisfied, or None."""
    def score_of(class_name):
        idx = name_to_index.get(class_name)
        return mean_scores[idx] if idx is not None else 0.0

    matched = set()
    for event_name, rule in config["events"].items():
        # 'any' classes: at least one must exceed threshold
        any_hit = max((score_of(c) for c in rule["any"]), default=0.0)
        if any_hit < rule["threshold"]:
            continue
        # optional 'require_also': a second condition must also hold
        if "require_also" in rule:
            also_hit = max((score_of(c) for c in rule["require_also"]), default=0.0)
            if also_hit < rule.get("also_threshold", 0.4):
                continue
        matched.add(event_name)

    # resolve by configured priority
    for event_name in config["priority"]:
        if event_name in matched:
            return event_name
    return None


# --- Audio settings ---
SAMPLE_RATE = 16000
NATIVE_RATE = 48000
WINDOW_SECONDS = 0.96
FRAMES = int(NATIVE_RATE * WINDOW_SECONDS)
LOOPBACK_NAME = "Speakers (Sound BlasterX Katana)"

mic = sc.get_microphone(LOOPBACK_NAME, include_loopback=True)

print("Listening for Sound Events\n")
try:
    # Function that keeps ONE recorder stream open the whole time
    with mic.recorder(samplerate=NATIVE_RATE, channels=1) as rec:
        while True:
            audio = rec.record(numframes=FRAMES)[:, 0]
            audio_16k = resampy.resample(audio, NATIVE_RATE, SAMPLE_RATE).astype(np.float32)

            if np.max(np.abs(audio_16k)) < 0.01:
                continue  # skip silence

            scores, _, _ = yamnet(audio_16k)
            mean_scores = np.mean(scores.numpy(), axis=0)

            event = classify_event(mean_scores)
            if event:
                desc = config["events"][event]["description"]
                print(f"  [EVENT] {event:<12} ({desc})")
except KeyboardInterrupt:
    print("\nStopped.")