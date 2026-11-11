"""Pure logic for usage-watch: settings, thresholds, cache/backoff, turn decisions, rendering.

Nothing here imports Hermes, so the unit tests run without a Hermes install. The host glue
(hooks, timers, post_message, slash command) lives in ``__init__.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Optional

logger = logging.getLogger("hermes_cli.plugins.usage_watch")

OK = "ok"
WARNING = "warning"
CRITICAL = "critical"
_RANK = {None: 0, OK: 0, WARNING: 1, CRITICAL: 2}

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "every_n_turns": 5,
    "idle_minutes": 5,
    "providers": ["auto"],
    "warn_above_percent": 80,
    "critical_above_percent": 95,
    "only_when_above": True,
    "mode": "repost",
    "layout": "bar",
}

MODES = ("repost", "edit")
LAYOUTS = ("bar", "subtext", "quote", "detail")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _as_int(value: Any, default: int, lo: int = 0, hi: int = 10_000) -> int:
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return default


def _as_providers(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(p).strip() for p in value]
    else:
        items = []
    items = [p for p in items if p]
    return items or ["auto"]


def _as_mode(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in MODES else DEFAULTS["mode"]


def _as_layout(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in LAYOUTS else DEFAULTS["layout"]


def coerce_settings(raw: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Tolerant settings parser: bad values fall back to defaults instead of raising."""
    raw = raw or {}
    cfg = {
        "enabled": _as_bool(raw.get("enabled"), DEFAULTS["enabled"]),
        "every_n_turns": _as_int(raw.get("every_n_turns"), DEFAULTS["every_n_turns"], 0, 1000),
        "idle_minutes": _as_int(raw.get("idle_minutes"), DEFAULTS["idle_minutes"], 0, 24 * 60),
        "providers": _as_providers(raw.get("providers", DEFAULTS["providers"])),
        "warn_above_percent": _as_int(raw.get("warn_above_percent"), DEFAULTS["warn_above_percent"], 1, 100),
        "critical_above_percent": _as_int(
            raw.get("critical_above_percent"), DEFAULTS["critical_above_percent"], 1, 100
        ),
        "only_when_above": _as_bool(raw.get("only_when_above"), DEFAULTS["only_when_above"]),
        "mode": _as_mode(raw.get("mode")),
        "layout": _as_layout(raw.get("layout")),
    }
    if cfg["critical_above_percent"] < cfg["warn_above_percent"]:
        cfg["critical_above_percent"] = cfg["warn_above_percent"]
    return cfg


# --------------------------------------------------------------------------- thresholds


def level_for_percent(pct: Optional[float], warn: int, crit: int) -> Optional[str]:
    if pct is None:
        return None
    if pct >= crit:
        return CRITICAL
    if pct >= warn:
        return WARNING
    return OK


def max_percent(snapshot: Any) -> Optional[float]:
    """Highest ``used_percent`` across a snapshot's windows (None when no window reports one)."""
    values = []
    for w in getattr(snapshot, "windows", ()) or ():
        pct = getattr(w, "used_percent", None)
        if isinstance(pct, (int, float)) and pct == pct:  # NaN-safe
            values.append(float(pct))
    return max(values) if values else None


def snapshot_level(snapshot: Any, warn: int, crit: int) -> Optional[str]:
    return level_for_percent(max_percent(snapshot), warn, crit)


def worst_level(levels: Iterable[Optional[str]]) -> Optional[str]:
    """Highest level among the known ones; None only when no provider reported a level at all."""
    worst: Optional[str] = None
    for lv in levels:
        if lv is None:
            continue
        if worst is None or _RANK[lv] > _RANK[worst]:
            worst = lv
    return worst


def should_notify(
    *, turn: int, every_n: int, level: Optional[str], prev_level: Optional[str], only_when_above: bool
) -> bool:
    """Decide whether this turn's result should be shown.

    - No usage data at all: stay silent.
    - Below warning and ``only_when_above``: stay silent.
    - Escalation to a higher level since the previous turn: show now.
    - Otherwise show every ``every_n`` completed turns (0 disables the periodic display).
    """
    if level is None:
        return False
    if level == OK and only_when_above:
        return False
    if _RANK[level] > _RANK.get(prev_level, 0):
        return True
    return every_n > 0 and turn % every_n == 0


# --------------------------------------------------------------------------- cache / backoff


@dataclass
class _Entry:
    snapshot: Any
    next_at: float
    failures: int = 0
    status: str = "fresh"


class UsageCache:
    """Per-provider cache around a fetch callable. Never raises.

    - success: cached for ``ttl`` seconds;
    - no usage data for the provider (None / unavailable): retried after ``empty_ttl``;
    - exception or rate limit (429 text in the reason): exponential backoff, capped at
      ``max_backoff``. The last good snapshot is returned meanwhile, so the notice keeps
      showing data instead of flickering.
    """

    def __init__(
        self,
        fetch: Callable[[str], Any],
        *,
        ttl: float = 60.0,
        empty_ttl: float = 600.0,
        max_backoff: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._ttl = ttl
        self._empty_ttl = empty_ttl
        self._max_backoff = max_backoff
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def get(self, provider: str, *, force: bool = False) -> tuple[Any, str]:
        """Return ``(snapshot_or_None, status)``; status is fresh|cached|backoff|empty|error."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(provider)
            if entry is not None and now < entry.next_at and (not force or entry.failures):
                return entry.snapshot, ("backoff" if entry.failures else "cached")
            previous = entry.snapshot if entry is not None else None
            failures = entry.failures if entry is not None else 0

        try:
            snap = self._fetch(provider)
        except Exception as exc:  # noqa: BLE001 - must never reach the agent loop
            logger.debug("usage-watch fetch failed for %s: %s", provider, exc)
            return self._store(provider, previous, now, failures + 1, "error", self._ttl)

        if snap is None:
            # fetch_account_usage returns None both for "no usage support" and for failures it
            # swallows (timeouts, network). Treat it as transient: short backoff, last good data kept.
            return self._store(provider, previous, now, failures + 1, "error", self._ttl)

        if not getattr(snap, "available", True):
            reason = str(getattr(snap, "unavailable_reason", "") or "")
            if "429" in reason or "rate" in reason.lower():
                return self._store(provider, previous, now, failures + 1, "error", self._ttl)
            # a stable answer (e.g. "only OAuth accounts have limits"): retry rarely
            return self._store(provider, None, now, 0, "empty", self._empty_ttl)

        return self._store(provider, snap, now, 0, "fresh", self._ttl)

    def _store(self, provider, snap, now, failures, status, base_ttl):
        if failures:
            wait = min(self._max_backoff, base_ttl * (2 ** min(failures, 6)))
        else:
            wait = base_ttl
        with self._lock:
            self._entries[provider] = _Entry(snap, now + wait, failures, status)
        return snap, status

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# --------------------------------------------------------------------------- session state


@dataclass
class SessionWatch:
    turns: int = 0
    level: Optional[str] = None  # level seen on the previous completed turn
    handle: Any = None  # PluginMessage from post_message, edited in place afterwards
    timer: Any = None  # threading.Timer for the idle refresh


# --------------------------------------------------------------------------- rendering


def fmt_reset(reset_at: Optional[datetime], now: datetime) -> str:
    """'resets in 2h 13m' / '3d 4h'; empty when the reset time is unknown."""
    if reset_at is None:
        return ""
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    delta = reset_at - now
    if delta <= timedelta(0):
        return "resets now"
    total_min = int(delta.total_seconds() // 60)
    days, rem = divmod(total_min, 24 * 60)
    hours, minutes = divmod(rem, 60)
    if days:
        return f"resets in {days}d {hours}h"
    if hours:
        return f"resets in {hours}h {minutes}m"
    return f"resets in {max(minutes, 1)}m"


_LEVEL_HEADER = {OK: "Usage OK", WARNING: "Usage WARNING", CRITICAL: "Usage CRITICAL"}


def render_notice(
    entries: list[tuple[str, Any]],
    *,
    level: Optional[str],
    cfg: Mapping[str, Any],
    now: datetime,
    clock_label: str = "",
    note: str = "",
) -> str:
    """Plain-text notice (no LLM involved). ``entries`` is ``[(provider, snapshot), ...]``."""
    top = max_over(entries)
    head_level = level or OK
    header = f"{_LEVEL_HEADER.get(head_level, 'Usage')}: {top:.0f}% used" if top is not None else _LEVEL_HEADER[head_level]
    lines = [header]
    for provider, snap in entries:
        plan = getattr(snap, "plan", None)
        label = provider + (f" ({plan})" if plan else "")
        lines.append(label)
        for w in getattr(snap, "windows", ()) or ():
            pct = getattr(w, "used_percent", None)
            pct_txt = f"{pct:.0f}%" if isinstance(pct, (int, float)) else "n/a"
            reset = fmt_reset(getattr(w, "reset_at", None), now)
            tail = f" · {reset}" if reset else ""
            scope = "" if getattr(w, "scope", "account") != "model" else " (model-only)"
            lines.append(f"  - {w.label}: {pct_txt}{scope}{tail}")
    footer = f"warn {cfg['warn_above_percent']}% · critical {cfg['critical_above_percent']}%"
    if clock_label:
        footer = f"checked {clock_label} · {footer}"
    if note:
        footer = f"{note} | {footer}"
    lines.append(footer)
    return "\n".join(lines)


def max_over(entries: list[tuple[str, Any]]) -> Optional[float]:
    vals = [v for v in (max_percent(s) for _, s in entries) if v is not None]
    return max(vals) if vals else None


def fmt_short(reset_at: Optional[datetime], now: datetime) -> str:
    """Compact reset time: '5d' when a day or more away, else '3h36m' / '12m'; empty when unknown."""
    if reset_at is None:
        return ""
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    secs = (reset_at - now).total_seconds()
    if secs <= 0:
        return "now"
    total_min = int(secs // 60)
    days, rem = divmod(total_min, 24 * 60)
    hours, minutes = divmod(rem, 60)
    if days:
        return f"{days}d"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{max(minutes, 1)}m"


_PROVIDER_ALIASES = (("openai-codex", "Codex"), ("codex", "Codex"), ("anthropic", "Claude"), ("claude", "Claude"))
_DOT = {OK: "🟢", WARNING: "🟡", CRITICAL: "🔴"}
_TAG = {WARNING: "WARN ", CRITICAL: "CRIT "}


def provider_label(provider: str) -> str:
    """Short display name: anthropic / claude-* -> Claude, openai-codex -> Codex, else as-is."""
    low = str(provider or "").lower()
    for prefix, label in _PROVIDER_ALIASES:
        if low.startswith(prefix):
            return label
    return str(provider)


def _short_label(label: str) -> str:
    label = str(label or "").strip()
    if label.lower().startswith("current "):
        label = label[8:]
    return label.lower() or "window"


def bar(left_pct: float, cells: int = 5) -> str:
    filled = max(0, min(cells, round(left_pct / (100 / cells))))
    return "▰" * filled + "▱" * (cells - filled)


def _windows(snap: Any, warn: int):
    """(label, used_pct_or_None, reset_at) to show. Secondary windows (model-only, or per-model caps
    next to "Current ..." windows) are hidden until they reach the warning level."""
    out = []
    windows = list(getattr(snap, "windows", ()) or ())
    # Claude reports per-model caps ("Fable week") with account scope next to "Current session/week";
    # when "Current ..." windows exist, the other ones count as secondary.
    has_current = any(str(getattr(w, "label", "")).lower().startswith("current ") for w in windows)
    for w in windows:
        pct = getattr(w, "used_percent", None)
        pct = float(pct) if isinstance(pct, (int, float)) and pct == pct else None
        secondary = getattr(w, "scope", "account") == "model" or (
            has_current and not str(getattr(w, "label", "")).lower().startswith("current "))
        if secondary and (pct is None or pct < warn):
            continue
        out.append((_short_label(w.label), pct, getattr(w, "reset_at", None)))
    return out


def render_compact(
    entries: list[tuple[str, Any]],
    *,
    level: Optional[str],
    now: datetime,
    layout: str = "bar",
    warn: int = 80,
    note: str = "",
    clock_label: str = "",
) -> str:
    """One line showing what is LEFT per window. Layouts (Discord markdown; harmless elsewhere):

    - bar:     ``-# 🟢 Claude · session ▰▰▰▰▱ 84% (3h36m) · week ▰▰▰▱▱ 56% (5d)``
    - subtext: ``-# 🟢 Claude · session 84% (3h36m) · week 56% (5d)``
    - quote:   ``> Claude · session 84% left 3h36m · week 56% left 5d``
    """
    providers = []
    for provider, snap in entries:
        wins = []
        for label, pct, reset_at in _windows(snap, warn):
            reset = fmt_short(reset_at, now)
            if pct is None:
                wins.append(f"{label} n/a")
                continue
            left = max(0.0, 100.0 - pct)
            if layout == "quote":
                wins.append(f"{label} {left:.0f}% left" + (f" {reset}" if reset else ""))
            elif layout == "bar":
                wins.append(f"{label} {bar(left)} {left:.0f}%" + (f" ({reset})" if reset else ""))
            else:
                wins.append(f"{label} {left:.0f}%" + (f" ({reset})" if reset else ""))
        providers.append(" · ".join([provider_label(provider)] + (wins or ["n/a"])))
    body = " | ".join(providers)
    if note:
        body += f" · {note}"
    if layout == "quote":
        return f"> {_TAG.get(level, '')}{body}"
    return f"-# {_DOT.get(level or OK, _DOT[OK])} {body}"
