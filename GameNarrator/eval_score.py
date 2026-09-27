"""Scores detected events against an annotated ground-truth log.

Compares two logs, so it needs no models and runs in a second.

Usage:
    python eval_score.py games/cs2.json truth_cs2.csv
    python eval_score.py games/cs2.json truth_cs2.csv --tolerance 2 --log mylog.jsonl
"""
import argparse
import csv
import json
import os


def parse_stamp(s):
    """Parse MM:SS or a plain number of seconds."""
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


def load_truth(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    truth = []
    for row in csv.DictReader(rows):
        t = row.get("t") or row.get("stamp")
        truth.append({"t": parse_stamp(t), "event": row["event"].strip()})
    return sorted(truth, key=lambda r: r["t"])


def load_session(log_path):
    """Entries from the last session_start marker, so repeated runs in one
    file are not scored together."""
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    starts = [i for i, r in enumerate(records) if r.get("type") == "session_start"]
    if starts:
        records = records[starts[-1]:]
    return [r for r in records if "audio" in r]  # per-tick entries only


def match(truth, session, tolerance):
    """Greedy nearest-timestamp matching, same event type, one to one.

    Returns (hits, misses, false_positives).
    """
    detections = [d for d in session if d.get("audio")]
    used = set()
    hits, misses = [], []

    for te in truth:
        best_i, best_dt = None, None
        for i, d in enumerate(detections):
            if i in used or d["audio"]["event"] != te["event"]:
                continue
            dt = abs(d["t"] - te["t"])
            if dt <= tolerance and (best_dt is None or dt < best_dt):
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i)
            hits.append((te, detections[best_i]))
        else:
            misses.append(te)

    false_positives = [d for i, d in enumerate(detections) if i not in used]
    return hits, misses, false_positives


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game_config")
    ap.add_argument("truth_csv")
    ap.add_argument("--log", default=None,
                     help="override the log path (default: derived from game_config, matching orchestrator.py)")
    ap.add_argument("--tolerance", type=float, default=3.0,
                     help="seconds of slack when matching a detection to a ground-truth event (default 3)")
    args = ap.parse_args()

    with open(args.game_config) as f:
        cfg = json.load(f)

    log_path = args.log or f"session_log_{os.path.splitext(os.path.basename(args.game_config))[0]}.jsonl"
    truth = load_truth(args.truth_csv)
    session = load_session(log_path)

    hits, misses, false_positives = match(truth, session, args.tolerance)

    recall = len(hits) / len(truth) if truth else 0.0
    total_detections = sum(1 for r in session if r.get("audio"))
    precision = len(hits) / total_detections if total_detections else 0.0

    print(f"Game: {cfg['game']}")
    print(f"Log: {log_path} ({len(session)} ticks)")
    print(f"Ground truth events: {len(truth)}\n")

    print(f"Recall (the project's headline metric): {len(hits)}/{len(truth)} = {recall:.1%}"
          + ("  -- >= 75% target MET" if recall >= 0.75 else "  -- below 75% target"))
    print(f"Precision: {len(hits)}/{total_detections} = {precision:.1%}\n")

    print("Per-event-type recall:")
    by_type = {}
    for te in truth:
        by_type.setdefault(te["event"], [0, 0])[1] += 1
    for te, _ in hits:
        by_type[te["event"]][0] += 1
    for event, (h, t) in sorted(by_type.items()):
        print(f"  {event:<15} {h}/{t} = {h/t:.0%}")

    if misses:
        print("\nMissed (ground truth, no matching detection within tolerance):")
        for m in misses:
            print(f"  {m['t']:>6.1f}s  {m['event']}")

    if false_positives:
        print("\nFalse positives (detected, no matching ground truth):")
        for d in false_positives:
            print(f"  {d['t']:>6.1f}s  {d['audio']['event']}")


if __name__ == "__main__":
    main()
