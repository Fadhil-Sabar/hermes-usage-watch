"""usage-watch: proactive provider account usage notices inside the active session.

Makes no LLM call. Notices go through ``ctx.post_message`` (a keyed message in the bound session,
edited in place on later updates). Usage comes from Hermes' own ``agent.account_usage`` fetchers.
Everything is wrapped so a failure here never reaches the agent loop.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

try:  # Hermes loads this package as a module; the fallback keeps odd loaders working.
    from .usage_watch_core import (
        DEFAULTS,
        SessionWatch,
        UsageCache,
        coerce_settings,
        render_notice,
        should_notify,
        snapshot_level,
        worst_level,
    )
except ImportError:  # pragma: no cover
    from usage_watch_core import (  # type: ignore
        DEFAULTS,
        SessionWatch,
        UsageCache,
        coerce_settings,
        render_notice,
        should_notify,
        snapshot_level,
        worst_level,
    )

logger = logging.getLogger("hermes_cli.plugins.usage_watch")

PLUGIN_NAME = "usage-watch"
_LOCK = threading.Lock()
_SESSIONS: dict[str, SessionWatch] = {}
_CACHE: Optional[UsageCache] = None
_CTX: Any = None


# --------------------------------------------------------------------------- host adapters


def _fetch_provider(provider: str) -> Any:
    """Public usage API. Returns a snapshot or None; raises only on import/IO errors (cache catches)."""
    from agent.account_usage import fetch_account_usage

    return fetch_account_usage(provider)


def _cache() -> UsageCache:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = UsageCache(_fetch_provider, ttl=60.0)
        return _CACHE


def _settings(ctx: Any) -> dict[str, Any]:
    raw = {}
    for key in DEFAULTS:
        try:
            raw[key] = ctx.get_config(key, DEFAULTS[key])
        except Exception:  # noqa: BLE001
            raw[key] = DEFAULTS[key]
    return coerce_settings(raw)


def _override_path(ctx: Any):
    from plugins.plugin_storage import plugin_data_dir  # public plugin API

    return plugin_data_dir(PLUGIN_NAME) / "state.json"


def _enabled(ctx: Any, cfg: dict[str, Any]) -> bool:
    """``/usage-watch on|off`` persists an override; otherwise the config value applies."""
    try:
        path = _override_path(ctx)
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8")).get("enabled")
            if isinstance(value, bool):
                return value
    except Exception:  # noqa: BLE001
        logger.debug("usage-watch: override unreadable", exc_info=True)
    return cfg["enabled"]


def _set_enabled(ctx: Any, value: bool) -> None:
    path = _override_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"enabled": value}), encoding="utf-8")


def _providers(cfg: dict[str, Any]) -> list[str]:
    names = []
    for name in cfg["providers"]:
        if name == "auto":
            name = _active_provider()
        if name and name not in names:
            names.append(name)
    return names


def _active_provider() -> str:
    """Provider from the profile's config.yaml (``model.provider``). Read-only, no LLM call."""
    try:
        from hermes_cli.config import load_config_readonly

        model = (load_config_readonly() or {}).get("model")
        if isinstance(model, dict):
            return str(model.get("provider") or "").strip()
    except Exception:  # noqa: BLE001
        logger.debug("usage-watch: cannot read model.provider", exc_info=True)
    return ""


def _collect(cfg: dict[str, Any], *, force: bool = False) -> list[tuple[str, Any]]:
    """[(provider, snapshot)] for providers with usage data. Providers without data are skipped."""
    out = []
    for provider in _providers(cfg):
        snap, _status = _cache().get(provider, force=force)
        if snap is not None and getattr(snap, "available", True):
            out.append((provider, snap))
    return out


def _levels(entries, cfg) -> Optional[str]:
    return worst_level(snapshot_level(s, cfg["warn_above_percent"], cfg["critical_above_percent"]) for _, s in entries)


def _clock() -> tuple[datetime, str]:
    now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        local = now.astimezone(ZoneInfo("Asia/Jakarta"))
        return now, local.strftime("%H:%M WIB")
    except Exception:  # noqa: BLE001
        return now, now.strftime("%H:%M UTC")


def _render(entries, cfg, *, note: str = "") -> str:
    now, label = _clock()
    return render_notice(
        entries, level=_levels(entries, cfg), cfg=cfg, now=now, clock_label=label, note=note
    )


# --------------------------------------------------------------------------- turn + idle


def _publish(ctx: Any, st: SessionWatch, text: str) -> None:
    """Edit the session's notice in place; post a new one only when there is no live handle."""
    if st.handle is not None:
        try:
            result = st.handle.edit(text)
            if result.get("ok"):
                return
        except Exception:  # noqa: BLE001
            logger.debug("usage-watch: edit failed", exc_info=True)
        st.handle = None  # refused or gone: fall through and post a fresh message
    handle = ctx.post_message(text)
    if getattr(handle, "ok", False):
        st.handle = handle
    else:
        # info, not debug: a refusal here is the one thing a user needs to see when no notice appears
        logger.info("usage-watch: post_message refused: %s %s", getattr(handle, "error", None),
                    getattr(handle, "result", {}).get("detail", ""))


def _arm_idle(ctx: Any, sid: str, st: SessionWatch, cfg: dict[str, Any]) -> None:
    """Replace the session's idle timer (caller holds _LOCK)."""
    if st.timer is not None:
        st.timer.cancel()
        st.timer = None
    if cfg["idle_minutes"] > 0:
        timer = threading.Timer(cfg["idle_minutes"] * 60, _on_idle, args=(ctx, sid))
        timer.daemon = True
        st.timer = timer
        timer.start()


def _handle_turn(ctx: Any, sid: str) -> None:
    cfg = _settings(ctx)
    if not _enabled(ctx, cfg):
        return
    with _LOCK:
        st = _SESSIONS.setdefault(sid, SessionWatch())
        st.turns += 1
        turn = st.turns
        _arm_idle(ctx, sid, st, cfg)
    entries = _collect(cfg)
    level = _levels(entries, cfg) if entries else None
    with _LOCK:
        prev = st.level
        st.level = level
    notify = level is not None and should_notify(
        turn=turn, every_n=cfg["every_n_turns"], level=level, prev_level=prev,
        only_when_above=cfg["only_when_above"],
    )
    logger.info("usage-watch: turn=%d session=%s providers=%d level=%s notify=%s",
                turn, sid[:12], len(entries), level, notify)
    if not notify:
        return
    _publish(ctx, st, _render(entries, cfg))


def _on_idle(ctx: Any, sid: str) -> None:
    """Idle refresh: updates an existing notice only. Without a live handle there is nothing to edit,
    because post_message needs a bound session and a timer thread has none."""
    try:
        with _LOCK:
            st = _SESSIONS.get(sid)
            if st is None:
                return
            st.timer = None
            handle = st.handle
        if handle is None:
            return
        cfg = _settings(ctx)
        if not _enabled(ctx, cfg):
            return
        entries = _collect(cfg)
        if not entries:
            return
        text = _render(entries, cfg, note="idle refresh")
        handle.edit(text)
    except Exception:  # noqa: BLE001
        logger.debug("usage-watch: idle refresh failed", exc_info=True)


# --------------------------------------------------------------------------- hooks + command


def _on_post_llm_call(**kwargs: Any) -> None:
    try:
        sid = str(kwargs.get("session_id") or "default")
        logger.info("usage-watch: post_llm_call session=%s platform=%s", sid[:12], kwargs.get("platform"))
        if _CTX is not None:
            _handle_turn(_CTX, sid)
    except Exception:  # noqa: BLE001 - never raise into the agent loop; warn so it is visible
        logger.warning("usage-watch: post_llm_call failed", exc_info=True)


def _on_session_end(**kwargs: Any) -> None:
    sid = str(kwargs.get("session_id") or "default")
    with _LOCK:
        st = _SESSIONS.pop(sid, None)
    if st is not None and st.timer is not None:
        st.timer.cancel()


def _command(raw_args: str) -> str:
    ctx = _CTX
    if ctx is None:
        return "usage-watch: not loaded"
    arg = (raw_args or "").strip().lower() or "status"
    cfg = _settings(ctx)
    if arg == "on":
        _set_enabled(ctx, True)
        return "usage-watch: on (saved for this profile)"
    if arg == "off":
        _set_enabled(ctx, False)
        return "usage-watch: off (saved for this profile)"
    if arg == "now":
        entries = _collect(cfg, force=True)
        if not entries:
            return "usage-watch: no usage data for providers " + ", ".join(_providers(cfg) or ["(none)"])
        return _render(entries, cfg)
    if arg == "status":
        return "\n".join([
            f"usage-watch: {'on' if _enabled(ctx, cfg) else 'off'}",
            f"providers: {', '.join(_providers(cfg)) or '(none)'}",
            f"every_n_turns={cfg['every_n_turns']} idle_minutes={cfg['idle_minutes']} "
            f"warn={cfg['warn_above_percent']}% critical={cfg['critical_above_percent']}% "
            f"only_when_above={cfg['only_when_above']}",
            "usage: /usage-watch status | now | on | off",
        ])
    return "usage: /usage-watch status | now | on | off"


def register(ctx: Any) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command(
        "usage-watch",
        handler=_command,
        description="Usage warnings: status | now | on | off",
        args_hint="status|now|on|off",
    )
