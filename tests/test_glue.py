"""Glue tests: the hook/timer/command wiring with a fake PluginContext. Fetch and config are mocked."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("usage_watch_glue", ROOT / "__init__.py")
glue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glue)


@dataclass
class W:
    label: str
    used_percent: float
    reset_at: object = None
    scope: str = "account"


@dataclass
class Snap:
    windows: tuple
    plan: object = None
    unavailable_reason: object = None

    @property
    def available(self):
        return bool(self.windows)


class Handle:
    def __init__(self, text, ok=True):
        self.ok = ok
        self.error = None if ok else "no_conversation"
        self.texts = [text]
        self.edit_ok = True

    def edit(self, text):
        if not self.edit_ok:
            return {"ok": False, "error": "gone"}
        self.texts.append(text)
        return {"ok": True}


class FakeCtx:
    def __init__(self, **settings):
        self.settings = settings
        self.posts = []
        self.hooks = {}
        self.commands = {}
        self.refuse_post = False

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def post_message(self, text):
        h = Handle(text, ok=not self.refuse_post)
        self.posts.append(h)
        return h


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    glue._CACHE = None
    glue._SESSIONS.clear()
    monkeypatch.setattr(glue, "_override_path", lambda ctx: tmp_path / "state.json")
    monkeypatch.setattr(glue, "_data_dir", lambda: tmp_path)  # never touch the real hermes home
    monkeypatch.setattr(glue, "_active_provider", lambda: "anthropic")
    state = {"pct": 10.0, "raise": False}
    def fake_fetch(provider):
        if state["raise"]:
            raise RuntimeError("network down")
        return Snap(windows=(W("Current session", state["pct"], None),))
    monkeypatch.setattr(glue, "_fetch_provider", fake_fetch)
    yield state
    for st in glue._SESSIONS.values():
        if st.timer is not None:
            st.timer.cancel()
    glue._SESSIONS.clear()


def _turn(ctx, sid="s1"):
    glue._CTX = ctx
    glue._on_post_llm_call(session_id=sid, model="m", platform="cli")


def test_low_usage_silent_when_only_when_above(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0)
    for _ in range(5):
        _turn(ctx)
    assert ctx.posts == []
    assert glue._SESSIONS["s1"].turns == 5


def test_every_turn_posts_once_then_edits_in_place(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False, mode="edit")
    for _ in range(3):
        _turn(ctx)
    assert len(ctx.posts) == 1
    assert len(ctx.posts[0].texts) == 3  # one post, two in-place edits
    assert ctx.posts[0].texts[-1].startswith("-# 🟢 Claude · ")


def test_repost_mode_is_default_and_posts_one_line_each_time(_isolate):
    ctx = FakeCtx(every_n_turns=2, idle_minutes=0, only_when_above=False)
    for _ in range(4):
        _turn(ctx)
    assert len(ctx.posts) == 2 and all(len(h.texts) == 1 for h in ctx.posts)
    text = ctx.posts[-1].texts[0]
    assert "\n" not in text and text.startswith("-# 🟢 Claude · ") and "▰" in text
    assert glue._SESSIONS["s1"].handle is ctx.posts[-1]  # idle refresh edits the latest one


def test_detail_layout_keeps_multiline_view(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False, layout="detail")
    _turn(ctx)
    assert ctx.posts[0].texts[0].startswith("Usage OK")


def test_every_n_turns_counter(_isolate):
    ctx = FakeCtx(every_n_turns=3, idle_minutes=0, only_when_above=False, mode="edit")
    for _ in range(7):
        _turn(ctx)
    h = ctx.posts[0]
    assert len(ctx.posts) == 1  # first periodic display is turn 3 (no notice on turns 1-2)
    assert len(h.texts) == 1 + 1  # post on turn 3, edit on turn 6


def test_escalation_warning_then_critical_edits_same_message(_isolate):
    ctx = FakeCtx(every_n_turns=0, idle_minutes=0, mode="edit")
    state = _isolate

    def set_pct(p):  # the usage cache is 60s; clear it to simulate a fresh fetch
        state["pct"] = p
        glue._cache().clear()

    set_pct(10)
    _turn(ctx)
    assert ctx.posts == []
    set_pct(85)
    _turn(ctx)
    assert len(ctx.posts) == 1 and "-# 🟡 " in ctx.posts[0].texts[-1]
    set_pct(97)
    _turn(ctx)
    assert len(ctx.posts) == 1
    assert "-# 🔴 " in ctx.posts[0].texts[-1]


def test_refused_edit_falls_back_to_a_new_post(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False, mode="edit")
    _turn(ctx)
    ctx.posts[0].edit_ok = False  # message deleted or edit refused
    _turn(ctx)
    assert len(ctx.posts) == 2


def test_refused_post_is_not_fatal(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False)
    ctx.refuse_post = True
    _turn(ctx)  # no bound session: post refused, must not raise
    assert glue._SESSIONS["s1"].handle is None


def test_fetch_exception_never_raises_and_posts_nothing(_isolate):
    _isolate["raise"] = True
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False)
    _turn(ctx)
    assert ctx.posts == []


def test_disabled_does_nothing(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False, enabled=False)
    _turn(ctx)
    assert ctx.posts == [] and "s1" not in glue._SESSIONS


def test_on_off_command_overrides_config(_isolate, tmp_path):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False)
    glue._CTX = ctx
    assert "off" in glue._command("off")
    _turn(ctx)
    assert ctx.posts == []
    assert "on" in glue._command("on")
    _turn(ctx)
    assert len(ctx.posts) == 1


def test_now_command_renders_without_posting(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0)
    glue._CTX = ctx
    _isolate["pct"] = 50
    out = glue._command("now")
    assert out.startswith("Usage OK") and ctx.posts == []


def test_status_and_help_text(_isolate):
    ctx = FakeCtx(every_n_turns=2, idle_minutes=0)
    glue._CTX = ctx
    assert "every_n_turns=2" in glue._command("status")
    assert glue._command("bogus").startswith("usage: /usage-watch")


def test_idle_refresh_edits_existing_notice_only(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=0, only_when_above=False)
    _turn(ctx)
    h = ctx.posts[0]
    _isolate["pct"] = 60
    glue._on_idle(ctx, "s1")
    assert h.texts[-1].startswith("-# ") and h.texts[-1].endswith(" · idle")
    # no handle yet: idle must not post anything
    glue._SESSIONS["other"] = glue.SessionWatch()
    glue._on_idle(ctx, "other")
    assert len(ctx.posts) == 1


def test_session_finalize_clears_state(_isolate):
    ctx = FakeCtx(every_n_turns=1, idle_minutes=5, only_when_above=False)
    _turn(ctx)
    assert glue._SESSIONS["s1"].timer is not None
    glue._on_session_finalize(session_id=None)  # no id: nothing to drop
    assert "s1" in glue._SESSIONS
    glue._on_session_finalize(session_id="s1")
    assert "s1" not in glue._SESSIONS


def test_state_survives_across_turns_and_edits_in_place(_isolate):
    # Regression: state was dropped every turn (on_session_end fires per run_conversation),
    # so the counter stuck at 1 and every notice was a fresh post.
    ctx = FakeCtx(every_n_turns=2, idle_minutes=0, only_when_above=False, mode="edit")
    for _ in range(4):
        _turn(ctx)
    assert glue._SESSIONS["s1"].turns == 4
    assert len(ctx.posts) == 1 and len(ctx.posts[0].texts) == 2  # posted on turn 2, edited on turn 4


def test_register_wires_hooks_and_command():
    class RegCtx(FakeCtx):
        def register_hook(self, name, cb, **kw):
            self.hooks[name] = cb

        def register_command(self, name, handler, description="", args_hint=""):
            self.commands[name] = handler
    ctx = RegCtx()
    glue.register(ctx)
    assert set(ctx.hooks) == {"post_llm_call", "on_session_finalize"}
    assert "usage-watch" in ctx.commands
