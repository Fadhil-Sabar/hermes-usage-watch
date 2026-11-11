"""Unit tests for usage_watch_core. The usage fetch is always mocked; no network, no Hermes."""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import usage_watch_core as core  # noqa: E402

NOW = datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc)


@dataclass
class W:
    label: str
    used_percent: object = None
    reset_at: object = None
    scope: str = "account"


@dataclass
class Snap:
    windows: tuple = ()
    plan: object = None
    unavailable_reason: object = None

    @property
    def available(self):
        return bool(self.windows) and not self.unavailable_reason


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- settings


def test_defaults_when_empty():
    cfg = core.coerce_settings({})
    assert cfg == {
        "enabled": True, "every_n_turns": 5, "idle_minutes": 5, "providers": ["auto"],
        "warn_above_percent": 80, "critical_above_percent": 95, "only_when_above": True,
        "mode": "repost", "layout": "bar",
    }


def test_bad_values_fall_back_instead_of_raising():
    cfg = core.coerce_settings({"every_n_turns": "abc", "idle_minutes": None, "enabled": "maybe"})
    assert cfg["every_n_turns"] == 5 and cfg["idle_minutes"] == 5 and cfg["enabled"] is True


def test_string_values_from_config_set():
    cfg = core.coerce_settings({"every_n_turns": "1", "only_when_above": "false", "enabled": "off"})
    assert cfg["every_n_turns"] == 1
    assert cfg["only_when_above"] is False
    assert cfg["enabled"] is False


def test_providers_string_and_list():
    assert core.coerce_settings({"providers": "anthropic, openai-codex"})["providers"] == ["anthropic", "openai-codex"]
    assert core.coerce_settings({"providers": ["", " auto "]})["providers"] == ["auto"]


def test_critical_never_below_warn():
    cfg = core.coerce_settings({"warn_above_percent": 90, "critical_above_percent": 50})
    assert cfg["critical_above_percent"] == 90


# ---------------------------------------------------------------- thresholds


@pytest.mark.parametrize("pct,expected", [(None, None), (0, "ok"), (79.9, "ok"), (80, "warning"),
                                          (94.9, "warning"), (95, "critical"), (100, "critical")])
def test_level_for_percent(pct, expected):
    assert core.level_for_percent(pct, 80, 95) == expected


def test_max_percent_ignores_missing_and_nan():
    s = Snap(windows=(W("a", 10), W("b", None), W("c", float("nan")), W("d", 42.5)))
    assert core.max_percent(s) == 42.5
    assert core.max_percent(Snap(windows=(W("x", None),))) is None


def test_worst_level_across_providers():
    assert core.worst_level(["ok", "warning", None]) == "warning"
    assert core.worst_level(["critical", "warning"]) == "critical"
    assert core.worst_level([]) is None
    assert core.worst_level(["ok", "ok"]) == "ok"  # all-ok must not collapse to None
    assert core.worst_level([None, "ok"]) == "ok"


@pytest.mark.parametrize("level,prev,only_above,every_n,turn,expected", [
    (None, None, True, 1, 1, False),          # no data: silent
    ("ok", None, True, 1, 1, False),          # low + only_when_above: silent
    ("ok", None, False, 5, 5, True),          # low, not only_when_above, periodic
    ("ok", None, False, 5, 4, False),         # low, between periods
    ("warning", None, True, 5, 3, True),      # first sighting at warning is an escalation
    ("warning", "ok", True, 5, 3, True),      # escalation shows immediately
    ("critical", "warning", True, 5, 3, True),
    ("warning", "critical", True, 5, 5, True),  # de-escalation on a period still shows
    ("warning", "warning", True, 5, 10, True),  # periodic while warning
    ("warning", "warning", True, 0, 10, False),  # every_n=0 disables periodic display
])
def test_should_notify(level, prev, only_above, every_n, turn, expected):
    assert core.should_notify(turn=turn, every_n=every_n, level=level, prev_level=prev,
                              only_when_above=only_above) is expected


# ---------------------------------------------------------------- cache / backoff


def _cache(fetch, clock):
    return core.UsageCache(fetch, ttl=60, empty_ttl=600, max_backoff=900, clock=clock)


def test_fetch_cached_for_ttl():
    calls = []
    clock = FakeClock()
    snap = Snap(windows=(W("Current session", 50),))

    def fetch(p):
        calls.append(p)
        return snap

    c = _cache(fetch, clock)
    assert c.get("anthropic") == (snap, "fresh")
    clock.t += 59
    assert c.get("anthropic")[1] == "cached"
    assert len(calls) == 1
    clock.t += 2
    assert c.get("anthropic")[1] == "fresh"
    assert len(calls) == 2


def test_force_bypasses_ttl_without_failures():
    calls = []
    clock = FakeClock()
    c = _cache(lambda p: calls.append(p) or Snap(windows=(W("x", 1),)), clock)
    c.get("p")
    c.get("p", force=True)
    assert len(calls) == 2


def test_exception_never_raises_and_backs_off_keeping_last_good():
    clock = FakeClock()
    state = {"n": 0}
    good = Snap(windows=(W("x", 70),))

    def fetch(p):
        state["n"] += 1
        if state["n"] == 1:
            return good
        raise RuntimeError("boom")

    c = _cache(fetch, clock)
    assert c.get("p")[0] is good
    clock.t += 61
    snap, status = c.get("p")
    assert status == "error" and snap is good  # stale data kept, no raise
    # still inside the backoff window: no new fetch
    clock.t += 60
    before = state["n"]
    assert c.get("p")[1] == "backoff"
    assert state["n"] == before


def test_rate_limit_reason_backs_off():
    clock = FakeClock()
    calls = []

    def fetch(p):
        calls.append(1)
        return Snap(unavailable_reason="HTTP 429 Too Many Requests")

    c = _cache(fetch, clock)
    assert c.get("p") == (None, "error")
    clock.t += 61  # first failure waits ttl*2 = 120s
    c.get("p")
    assert len(calls) == 1
    clock.t += 120
    c.get("p")
    assert len(calls) == 2


def test_backoff_is_capped():
    clock = FakeClock()
    c = _cache(lambda p: (_ for _ in ()).throw(RuntimeError("x")), clock)
    for _ in range(12):
        c.get("p")
        clock.t += 10_000
    entry = c._entries["p"]
    assert entry.next_at - clock.t <= 900 + 1e-6 or entry.next_at >= clock.t  # waits never exceed the cap
    assert entry.failures >= 12


def test_none_is_transient_not_cached_for_long():
    """fetch_account_usage returns None on swallowed failures too, so None must not stick for minutes."""
    clock = FakeClock()
    calls = []

    def fetch(p):
        calls.append(1)
        return None

    c = _cache(fetch, clock)
    assert c.get("p") == (None, "error")
    clock.t += 61  # first failure waits ttl * 2 = 120 s
    c.get("p")
    assert len(calls) == 1
    clock.t += 60
    c.get("p")
    assert len(calls) == 2


def test_stable_unavailable_reason_is_retried_slowly():
    clock = FakeClock()
    calls = []

    def fetch(p):
        calls.append(1)
        return Snap(unavailable_reason="Anthropic account limits are only available for OAuth-backed accounts.")

    c = _cache(fetch, clock)
    assert c.get("nope") == (None, "empty")
    clock.t += 300
    assert c.get("nope")[1] == "cached"
    assert len(calls) == 1
    clock.t += 301
    c.get("nope")
    assert len(calls) == 2


# ---------------------------------------------------------------- formatting


@pytest.mark.parametrize("delta,expected", [
    (timedelta(minutes=0), "resets now"),
    (timedelta(minutes=-5), "resets now"),
    (timedelta(seconds=20), "resets in 1m"),
    (timedelta(minutes=45), "resets in 45m"),
    (timedelta(hours=2, minutes=13), "resets in 2h 13m"),
    (timedelta(days=3, hours=4, minutes=30), "resets in 3d 4h"),
])
def test_fmt_reset(delta, expected):
    assert core.fmt_reset(NOW + delta, NOW) == expected


def test_fmt_reset_unknown_and_naive():
    assert core.fmt_reset(None, NOW) == ""
    naive = datetime(2026, 11, 11, 12, 0)  # treated as UTC
    assert core.fmt_reset(naive, NOW) == "resets in 2h 0m"


def test_render_notice_contains_windows_resets_and_thresholds():
    snap = Snap(windows=(
        W("Current session", 96.4, NOW + timedelta(hours=1, minutes=12)),
        W("Opus week", 41, NOW + timedelta(days=3, hours=4), scope="model"),
    ), plan="Max")
    cfg = core.coerce_settings({})
    text = core.render_notice([("anthropic", snap)], level="critical", cfg=cfg, now=NOW,
                              clock_label="17:00 WIB")
    assert text.splitlines()[0] == "Usage CRITICAL: 96% used"
    assert "anthropic (Max)" in text
    assert "  - Current session: 96% · resets in 1h 12m" in text
    assert "(model-only)" in text and "resets in 3d 4h" in text
    assert "warn 80% · critical 95%" in text
    assert text.splitlines()[-1].startswith("checked 17:00 WIB")


def test_render_notice_handles_unknown_percent_and_note():
    snap = Snap(windows=(W("Session", None),))
    text = core.render_notice([("codex", snap)], level=None, cfg=core.coerce_settings({}), now=NOW,
                              note="idle refresh")
    assert "Usage OK" in text.splitlines()[0]
    assert "n/a" in text
    assert text.splitlines()[-1].startswith("idle refresh |")


def test_mode_setting_tolerant():
    assert core.coerce_settings({"mode": "EDIT"})["mode"] == "edit"
    assert core.coerce_settings({"mode": "bogus"})["mode"] == "repost"


def _compact_snap():
    return Snap(windows=(
        W("Current session", 16, NOW + timedelta(hours=3, minutes=36)),
        W("Current week", 44, NOW + timedelta(days=5, hours=11)),
        W("Fable week", 1, NOW + timedelta(days=5), scope="model"),
    ), plan="Max")


def test_render_compact_layouts():
    snap = _compact_snap()
    prov = "claude-subscription-directsdk-experimental"
    r = lambda layout, level="ok": core.render_compact([(prov, snap)], level=level, now=NOW, layout=layout)
    assert r("bar") == "-# 🟢 Claude · session ▰▰▰▰▱ 84% (3h36m) · week ▰▰▰▱▱ 56% (5d)"
    assert r("subtext") == "-# 🟢 Claude · session 84% (3h36m) · week 56% (5d)"
    assert r("quote") == "> Claude · session 84% left 3h36m · week 56% left 5d"
    assert r("bar", "warning").startswith("-# 🟡 ") and r("quote", "critical").startswith("> CRIT ")


def test_render_compact_model_window_shown_when_high_and_multi_provider():
    hot = Snap(windows=(W("Opus week", 90, None, scope="model"), W("Session", None)))
    text = core.render_compact([("anthropic", _compact_snap()), ("openai-codex", hot)], level="warning",
                               now=NOW, layout="subtext", note="idle")
    assert " | Codex · opus week 10% · session n/a · idle" in text
    assert "fable" not in text


def test_fmt_short_and_aliases():
    assert core.fmt_short(NOW + timedelta(minutes=5), NOW) == "5m"
    assert core.fmt_short(NOW - timedelta(minutes=5), NOW) == "now"
    assert core.fmt_short(None, NOW) == ""
    assert core.provider_label("nous") == "nous"
    assert core.coerce_settings({"layout": "QUOTE"})["layout"] == "quote"
    assert core.coerce_settings({"layout": "x"})["layout"] == "bar"


# ---------------------------------------------------------------- turn flow (pure simulation)


def test_turn_flow_every_n_and_escalation():
    """Simulate turn decisions the way __init__._handle_turn does, with a usage sequence."""
    cfg = core.coerce_settings({"every_n_turns": 3, "only_when_above": True})
    usage = [10, 20, 30, 85, 86, 96, 40, 40, 40]  # percent per completed turn
    st = core.SessionWatch()
    shown = []
    for turn, pct in enumerate(usage, start=1):
        st.turns = turn
        level = core.snapshot_level(Snap(windows=(W("s", pct),)), 80, 95)
        prev, st.level = st.level, level
        if core.should_notify(turn=turn, every_n=3, level=level, prev_level=prev,
                              only_when_above=cfg["only_when_above"]):
            shown.append((turn, level))
    # turn 4 escalates to warning, turn 6 escalates to critical, turn 9 is a periodic reminder only
    # if still elevated (40 is OK, so silent), turns 3/6 are periodic but below warning at turn 3.
    assert shown == [(4, "warning"), (6, "critical")]
