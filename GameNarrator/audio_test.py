"""Prints YAMNet's live top-5 classes, for choosing mapping thresholds."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "tfhub_cache")

import sys
import warnings

import numpy as np
import resampy
import soundcard as sc
from soundcard import SoundcardRuntimeWarning

import perception

warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# A 0.96s window gives one frame, so mean equals max. Pass a longer window
# to compare them, e.g. python audio_test.py 1.92
CAPTURE_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.96

print("Loading YAMNet...")
yamnet, class_names, name_to_idx = perception.load_yamnet(
    os.path.join(os.path.dirname(__file__), "tfhub_cache"))
print(f"Loaded {len(class_names)} classes.\n")

SAMPLE_RATE = 16000
NATIVE_RATE = 48000
FRAMES = int(NATIVE_RATE * CAPTURE_SECONDS)
LOOPBACK_NAME = "Speakers (Sound BlasterX Katana)"

mic = sc.get_microphone(LOOPBACK_NAME, include_loopback=True)

print(f"Listening for sound events, {CAPTURE_SECONDS}s window\n")
try:
    with mic.recorder(samplerate=NATIVE_RATE, channels=1) as rec:
        while True:
            audio = rec.record(numframes=FRAMES)[:, 0]
            audio_16k = resampy.resample(audio, NATIVE_RATE, SAMPLE_RATE).astype(np.float32)

            if np.max(np.abs(audio_16k)) < 0.01:
                continue  # skip silence

            scores = yamnet(audio_16k)[0].numpy()
            mean_scores, max_scores = scores.mean(axis=0), scores.max(axis=0)

            print(f"{'class':<30} {'mean':>6} {'max':>6}")
            for i in np.argsort(max_scores)[::-1][:5]:
                print(f"   {class_names[i]:<30} {mean_scores[i]:.3f} {max_scores[i]:.3f}")
            print("-" * 40)
except KeyboardInterrupt:
    print("\nStopped.")
