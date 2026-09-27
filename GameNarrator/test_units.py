"""Unit tests for the pipeline logic that needs no GPU, game or model.

Run: python test_units.py

Two are regression tests for bugs found in live sessions, marked below.
"""
import json
import os
import tempfile

import numpy as np

import capture
import eval_score
import perception
from trigger import should_narrate


def test_trigger():
    # max gap breaks a long silence even with nothing detected
    assert should_narrate(30, None, None, min_gap=5, max_gap=25)
    # nothing fires inside the minimum gap
    assert not should_narrate(1, "GUNFIRE", None, min_gap=5, max_gap=25)
    # a changed event past the minimum gap fires
    assert should_narrate(10, "EXPLOSION", "GUNFIRE", min_gap=5, max_gap=25)
    # continuous fire collapses into one juncture until max_gap
    assert not should_narrate(10, "GUNFIRE", "GUNFIRE", min_gap=5, max_gap=25)
    assert should_narrate(25, "GUNFIRE", "GUNFIRE", min_gap=5, max_gap=25)
    # a quiet tick between the gaps is not a narration moment
    assert not should_narrate(10, None, "GUNFIRE", min_gap=5, max_gap=25)
    # with nothing perceived at all, stay silent rather than invent
    assert not should_narrate(30, None, None, min_gap=5, max_gap=25, has_evidence=False)
    print("trigger: ok")


IDX = {"LOUD": 0, "QUIET": 1, "Silence": 2, "ALSO": 3}


def scores(*frames):
    """frames: dicts of class -> score, one per YAMNet frame."""
    a = np.zeros((len(frames), len(IDX)), dtype=np.float32)
    for i, f in enumerate(frames):
        for name, v in f.items():
            a[i, IDX[name]] = v
    return a


def test_classify_audio():
    cfg = {"BANG": {"any": ["LOUD"], "threshold": 0.4, "description": "bang"}}

    # below threshold -> nothing
    assert perception.classify_audio(scores({"LOUD": 0.3}), IDX, cfg, ["BANG"]) is None
    # above threshold -> fires, carrying the confidence
    got = perception.classify_audio(scores({"LOUD": 0.9}), IDX, cfg, ["BANG"])
    assert got[0] == "BANG" and abs(got[2] - 0.9) < 1e-6

    # priority decides when two events both qualify
    two = {
        "BIG":   {"any": ["LOUD"],  "threshold": 0.4, "description": "big"},
        "SMALL": {"any": ["QUIET"], "threshold": 0.4, "description": "small"},
    }
    both = scores({"LOUD": 0.9, "QUIET": 0.9})
    assert perception.classify_audio(both, IDX, two, ["BIG", "SMALL"])[0] == "BIG"
    assert perception.classify_audio(both, IDX, two, ["SMALL", "BIG"])[0] == "SMALL"

    # require_also: the second condition must hold too
    req = {"BANG": {"any": ["LOUD"], "threshold": 0.4, "require_also": ["ALSO"],
                    "also_threshold": 0.5, "description": "bang"}}
    assert perception.classify_audio(scores({"LOUD": 0.9}), IDX, req, ["BANG"]) is None
    assert perception.classify_audio(scores({"LOUD": 0.9, "ALSO": 0.6}), IDX, req, ["BANG"])

    # reject_if_above vetoes an otherwise valid hit
    veto = {"BANG": {"any": ["LOUD"], "threshold": 0.4,
                     "reject_if_above": {"Silence": 0.45}, "description": "bang"}}
    assert perception.classify_audio(scores({"LOUD": 0.9, "Silence": 0.9}), IDX, veto, ["BANG"]) is None
    assert perception.classify_audio(scores({"LOUD": 0.9, "Silence": 0.1}), IDX, veto, ["BANG"])

    # a config comment sitting beside the event specs must not crash the scan
    assert perception.unknown_classes({"_comment": "note", "BANG": {"any": ["LOUD"]}}, IDX) == []
    print("classify_audio: ok")


def test_aggregation_regression():
    """Regression. A brief sound in one frame of a window is diluted by the
    mean but kept by the max. Using the mean lost real chess move events."""
    cfg = {"TAP": {"any": ["LOUD"], "threshold": 0.4, "description": "tap"}}
    window = scores({"LOUD": 0.9}, {"LOUD": 0.0}, {"LOUD": 0.0})  # spike in frame 1 of 3
    assert perception.classify_audio(window, IDX, cfg, ["TAP"], "mean") is None
    assert perception.classify_audio(window, IDX, cfg, ["TAP"], "max") is not None

    # Regression. Max applies to Silence too, so a veto tuned on single-frame
    # data rejected every event: a real sound is still flanked by quiet frames.
    veto = dict(cfg["TAP"], reject_if_above={"Silence": 0.45})
    w = scores({"LOUD": 0.9, "Silence": 0.0}, {"LOUD": 0.0, "Silence": 1.0})
    assert perception.classify_audio(w, IDX, {"TAP": veto}, ["TAP"], "max") is None
    print("aggregation regression: ok")


def test_crop():
    region = {"left": 0, "top": 0, "width": 1000, "height": 500}
    assert capture._apply_crop(region, None) == region
    # left/top default to centring the box, not anchoring it at the corner
    assert capture._apply_crop(region, {"width": 0.5, "height": 1.0}) == {
        "left": 250, "top": 0, "width": 500, "height": 500}
    # explicit offsets are honoured
    assert capture._apply_crop(region, {"left": 0.0, "top": 0.5, "width": 1.0, "height": 0.5}) == {
        "left": 0, "top": 250, "width": 1000, "height": 250}
    print("crop: ok")


def test_eval_scoring():
    assert eval_score.parse_stamp("1:20") == 80.0
    assert eval_score.parse_stamp("12.5") == 12.5

    truth = [{"t": 10.0, "event": "A"}, {"t": 50.0, "event": "A"}, {"t": 90.0, "event": "B"}]
    session = [
        {"t": 11.0, "audio": {"event": "A"}},   # hit, within tolerance
        {"t": 30.0, "audio": {"event": "B"}},   # false positive, no truth nearby
        {"t": 70.0, "audio": {"event": "A"}},   # too far from truth@50 -> false positive
    ]
    hits, misses, fps = eval_score.match(truth, session, tolerance=3.0)
    assert len(hits) == 1
    assert {m["t"] for m in misses} == {50.0, 90.0}
    assert len(fps) == 2

    # one detection cannot satisfy two ground-truth events
    dup_truth = [{"t": 10.0, "event": "A"}, {"t": 11.0, "event": "A"}]
    hits, misses, _ = eval_score.match(dup_truth, [{"t": 10.5, "audio": {"event": "A"}}], 3.0)
    assert len(hits) == 1 and len(misses) == 1
    print("eval scoring: ok")


def test_session_isolation_regression():
    """Regression. Runs append to one file, so only the latest session should
    be scored, or an earlier run inflates the result."""
    records = [
        {"type": "session_start"},
        {"t": 1.0, "audio": {"event": "OLD"}},
        {"type": "session_start"},
        {"t": 2.0, "audio": {"event": "NEW"}},
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        session = eval_score.load_session(path)
        assert [r["audio"]["event"] for r in session] == ["NEW"]
    finally:
        os.unlink(path)
    print("session isolation regression: ok")


if __name__ == "__main__":
    test_trigger()
    test_classify_audio()
    test_aggregation_regression()
    test_crop()
    test_eval_scoring()
    test_session_isolation_regression()
    print("\nall unit tests passed")
