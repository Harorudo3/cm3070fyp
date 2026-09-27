import os
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "tfhub_cache")

import time
import json
import subprocess
import sys
import threading
import warnings
from datetime import datetime

import numpy as np
import requests
import resampy
import soundcard as sc
import mss
import torch
from PIL import Image
from soundcard import SoundcardRuntimeWarning

import capture
import perception
from trigger import should_narrate

warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
warnings.filterwarnings("ignore", message="data discontinuity in recording")

GAME_CONFIG   = sys.argv[1] if len(sys.argv) > 1 else "games/cs2.json"
# One log per game so the scorer can isolate a single game's sessions.
LOG_PATH      = f"session_log_{os.path.splitext(os.path.basename(GAME_CONFIG))[0]}.jsonl"

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3"
LLM_TIMEOUT   = 60
# Ollama unloads an idle model after 5 minutes, so hold it for the session.
OLLAMA_KEEP_ALIVE = "30m"

SAMPLE_RATE   = 16000
LOOPBACK_NAME = "Speakers (Sound BlasterX Katana)"
NATIVE_RATE   = 48000

with open(GAME_CONFIG, "r") as f:
    cfg = json.load(f)

events_cfg    = cfg["events"]
priority      = cfg["priority"]
scene_prompts = cfg["scene_prompts"]
narration_cfg = cfg["narration"]
visual_cfg    = cfg.get("visual", {})
capture_cfg   = cfg.get("capture", {"mode": "monitor", "fallback_monitor": 1})

# YAMNet scores a fixed 0.96s frame every 0.48s, so a 0.96s window gives one
# frame and max equals mean. Impulsive games need a longer window.
AGGREGATION     = cfg.get("aggregation", "mean")
CAPTURE_SECONDS = cfg.get("window_s", 0.96)
FRAMES          = int(NATIVE_RATE * CAPTURE_SECONDS)

# Scene separability varies by game, so the gates are per-game.
VISUAL_MIN_CONF = visual_cfg.get("min_conf", 0.65)
VISUAL_MARGIN   = visual_cfg.get("margin", 0.10)

MIN_GAP = narration_cfg["min_gap_s"]
MAX_GAP = narration_cfg["max_gap_s"]
print(f"Loaded config: {cfg['game']}")

print("Loading YAMNet")
yamnet, class_names, name_to_idx = perception.load_yamnet(
    os.path.join(os.path.dirname(__file__), "tfhub_cache"))
print(f"Loaded {len(class_names)} classes.")

unknown = perception.unknown_classes(events_cfg, name_to_idx)
if unknown:
    print("  WARNING, these config classes are not YAMNet classes and will never match:")
    for u in unknown:
        print(f"    {u}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP ViT-B/32 on {device}")
clip_model, preprocess, text_features = perception.load_clip(scene_prompts, device)


# Narration → Llama 3 via Ollama
def llm(prompt, num_predict, timeout=LLM_TIMEOUT):
    """Generate text. Returns None if Ollama is unreachable, so a missing model
    server degrades the session to logging instead of ending it."""
    try:
        r = requests.post(OLLAMA_URL, timeout=timeout, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            # Low temperature reduces invented detail.
            "options": {"num_predict": num_predict, "temperature": 0.3},
        })
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.RequestException as e:
        print(f"  [llm unavailable: {e}]")
        return None


def gpu_used_mib():
    """Device-wide VRAM in use. torch's counters only see CLIP, not Ollama."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        return int(out.splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def preload():
    """Load the model before the session so the first narration is not charged
    for it, and a missing model is reported up front."""
    print(f"Preloading {OLLAMA_MODEL} via Ollama (first run reads ~5GB from disk)")
    t0 = time.time()
    # A cold load is much slower than any later generation.
    if llm("Reply with the single word: ready", num_predict=4, timeout=300) is None:
        print("  Ollama unreachable. Session will log events but not narrate.")
        return
    used = gpu_used_mib()
    print(f"  loaded in {time.time() - t0:.1f}s"
          + (f", GPU now {used} MiB in use" if used else ""))


def assemble_context(event, description, scene, recent):
    """Combine the current audio event and visual scene into one prompt."""
    lines = [f"Game: {cfg['game']}", f"On screen: {scene}"]
    # Omit the sound line entirely when no event fired. Saying "nothing distinct"
    # reads as silence and the model narrates the player as standing still.
    if event:
        lines.append(f"Sound: {description}")
    if recent:
        lines.append("Just before this: " + "; ".join(recent))
    return (
        "You are narrating a live video game for someone who cannot see the screen.\n"
        + "\n".join(lines)
        + "\n\nDescribe what is happening in one short, plain sentence.\n"
          "Write in the third person about the player. Never write as 'I', 'we' or 'you'.\n"
          "Cover both the sound and the screen when both are given.\n"
          "Report only what the lines above state. Do not add drama, tension, stakes, "
          "or any detail not listed. Do not invent scores, names, outcomes, directions, "
          "or what players intend. Do not name a specific piece, square, or move "
          "unless it is written above word for word - a vague line stays vague."
    )


def summarise(moments):
    # Spelling out the unit helps but does not fix it. The model still misreads
    # timestamps as durations and miscounts when summarising.
    joined = "\n".join(f"[{m['stamp']} minutes:seconds elapsed] {m['text']}" for m in moments)
    return (
        "Below is a running commentary of one gameplay session of "
        f"{cfg['game']}, in order.\n\n{joined}\n\n"
        "Write a short recap of the session as a connected account, not a list.\n"
        # Repeated here, or the model drifts into first person.
        "Write in the third person about the player. Never write as 'I' or 'we'. "
        "Report only what the lines above state. Do not add drama, tension, stakes, "
        "or any detail not listed."
    )


# Session loop
sct = mss.mss()
mic = sc.get_microphone(LOOPBACK_NAME, include_loopback=True)
log = open(LOG_PATH, "a", encoding="utf-8")

last_narration_t = 0.0
last_narrated_event = None
recent_events = []   # short rolling window of event descriptions for context
moments = []         # every narration produced, for the session summary
peak_vram = 0        # device-wide, sampled at each narration

narrating = threading.Event()   # set while a narration is in flight
log_lock = threading.Lock()     # the narration thread also writes to the log


def write_log(record):
    with log_lock:
        log.write(json.dumps(record) + "\n")
        log.flush()


def narrate_async(stamp, prompt):
    """Generate narration on a background thread.

    Generating inline blocks capture for several seconds, so the system misses
    whatever happens during it. One narration at a time; ticks arriving during
    one are skipped rather than queued, since stale commentary is worse than none.
    """
    def work():
        global peak_vram
        try:
            t0 = time.time()
            text = llm(prompt, num_predict=60)
            latency = round(time.time() - t0, 2)
            if text:
                vram = gpu_used_mib()
                if vram:
                    peak_vram = max(peak_vram, vram)
                print(f"[{stamp}] NARRATE {text}  ({latency}s)")
                moments.append({"stamp": stamp, "text": text, "latency_s": latency})
                write_log({"type": "narration", "stamp": stamp, "text": text,
                           "latency_s": latency, "gpu_used_mib": vram})
        finally:
            narrating.clear()

    narrating.set()
    threading.Thread(target=work, daemon=True).start()


preload()

session_start = time.time()   # after preload, which is not session time
# Marks where this run starts, so the scorer can isolate the latest session.
write_log({"type": "session_start", "wall": datetime.now().isoformat(timespec="seconds"),
           "game": cfg["game"]})
print(f"\nSession started. Logging to {LOG_PATH}\n")
try:
    with mic.recorder(samplerate=NATIVE_RATE, channels=1) as rec:
        while True:
            t = time.time() - session_start
            stamp = f"{int(t//60):02d}:{int(t%60):02d}"

            # Audio
            audio = rec.record(numframes=FRAMES)[:, 0]
            audio_16k = resampy.resample(audio, NATIVE_RATE, SAMPLE_RATE).astype(np.float32)

            audio_event = None
            if np.max(np.abs(audio_16k)) >= 0.01:
                scores, _, _ = yamnet(audio_16k)
                audio_event = perception.classify_audio(
                    scores.numpy(), name_to_idx, events_cfg, priority, AGGREGATION)

            # Visual (sequential, after audio inference)
            region = capture.resolve_region(sct, capture_cfg, cfg["game"])
            frame = capture.grab_frame(sct, region)
            if not os.path.exists("clip_view.png"):
                # Save what CLIP saw once per run, for checking the crop.
                Image.fromarray(frame).resize((224, 224)).save("clip_view.png")
            scene, scene_conf, confident = perception.classify_scene(
                clip_model, preprocess, text_features, scene_prompts, frame, device,
                VISUAL_MIN_CONF, VISUAL_MARGIN)

            # Logging
            entry = {
                "t": round(t, 2),
                "stamp": stamp,
                "wall": datetime.now().isoformat(timespec="seconds"),
                "audio": None,
                "visual": {
                    "scene": scene if confident else "unclear",
                    "confidence": round(scene_conf, 3),
                    "confident": confident,
                },
            }
            if audio_event:
                name, desc, conf = audio_event
                entry["audio"] = {"event": name, "description": desc,
                                  "confidence": round(conf, 3)}
                print(f"[{stamp}] AUDIO  {name:<10} ({desc}) {conf:.2f}")

            vis_label = scene if confident else "unclear"
            print(f"[{stamp}] VISUAL {vis_label[:50]:<50} {scene_conf:.2f}")

            # Coordination: is this a narration moment?
            event_name = audio_event[0] if audio_event else None
            event_desc = audio_event[1] if audio_event else None
            if event_desc:
                recent_events.append(event_desc)
                del recent_events[:-3]

            has_evidence = bool(event_name) or confident
            if not narrating.is_set() and should_narrate(
                    t - last_narration_t, event_name, last_narrated_event,
                    MIN_GAP, MAX_GAP, has_evidence):
                last_narration_t = t
                last_narrated_event = event_name
                narrate_async(stamp, assemble_context(
                    event_name, event_desc, vis_label,
                    recent_events[:-1] if event_desc else recent_events))

            write_log(entry)

except KeyboardInterrupt:
    print("\nSession ended.")
finally:
    if narrating.is_set():
        print("Waiting for narration in flight")
        for _ in range(int(LLM_TIMEOUT)):
            if not narrating.is_set():
                break
            time.sleep(1)

    if moments:
        lat = sorted(m["latency_s"] for m in moments)
        print(f"\nNarrations: {len(moments)}"
              f" | median latency {lat[len(lat)//2]}s"
              + (f" | peak GPU {peak_vram} MiB of 8192" if peak_vram else ""))

        print("\nSession summary\n" + "-" * 40)
        summary = llm(summarise(moments), num_predict=400)
        print(summary or "(unavailable)")
        write_log({
            "type": "session_summary",
            "wall": datetime.now().isoformat(timespec="seconds"),
            "moments": len(moments),
            "text": summary,
        })

    log.close()
