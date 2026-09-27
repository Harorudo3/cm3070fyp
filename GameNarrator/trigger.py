"""Decides when a narration moment occurs.

Separate from the orchestrator so it can be tested without loading models.
"""


def should_narrate(since_last, event, last_event, min_gap, max_gap, has_evidence=True):
    """Whether to narrate on this tick.

    since_last   seconds since the last narration
    event        event detected this tick, or None
    last_event   event that triggered the last narration, or None
    has_evidence an audio event fired, or CLIP was confident

    max_gap stops slow games going silent, min_gap stops fast games flooding.
    Between the two, only a change of event narrates, so continuous fire becomes
    one line instead of one per window. Nothing fires without evidence, since an
    empty context makes the model invent rather than stay quiet.
    """
    if not has_evidence:
        return False
    if since_last >= max_gap:
        return True
    if since_last < min_gap:
        return False
    return event is not None and event != last_event


def demo():
    # max_gap breaks a long silence
    assert should_narrate(30, None, None, min_gap=5, max_gap=25)
    # nothing fires inside min_gap
    assert not should_narrate(1, "GUNFIRE", None, min_gap=5, max_gap=25)
    # a changed event past min_gap fires
    assert should_narrate(10, "EXPLOSION", "GUNFIRE", min_gap=5, max_gap=25)
    # the same event does not re-fire until max_gap
    assert not should_narrate(10, "GUNFIRE", "GUNFIRE", min_gap=5, max_gap=25)
    assert should_narrate(25, "GUNFIRE", "GUNFIRE", min_gap=5, max_gap=25)
    # a quiet tick between the gaps is not a narration moment
    assert not should_narrate(10, None, "GUNFIRE", min_gap=5, max_gap=25)
    # nothing perceived means stay silent
    assert not should_narrate(30, None, None, min_gap=5, max_gap=25, has_evidence=False)
    assert should_narrate(30, None, None, min_gap=5, max_gap=25, has_evidence=True)
    print("trigger: ok")


if __name__ == "__main__":
    demo()
