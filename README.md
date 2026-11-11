# usage-watch

A Hermes plugin that shows your provider account usage (Claude subscription, Codex, ...) inside the
active session when it gets high, with the time until the window resets.

- **No LLM calls.** Usage is read through Hermes' own `agent.account_usage` fetchers, and the notice is
  posted with `ctx.post_message`, which writes a message into the session without a model turn.
- **Two modes.** `repost` (default): every update is a new one-line message. `edit`: one notice per
  session, edited in place.
- **Warnings only.** It never switches models or changes anything else.

## What it shows

One quoted line per update, showing what is **left** in each window and when it resets:

```
> usage anthropic: session 14% left (1h12m) · week 59% left (3d4h) · 17:42 WIB
> WARN usage anthropic: session 12% left (58m) · week 57% left (3d3h) · 18:03 WIB
```

`WARN` / `CRIT` appear once usage passes `warn_above_percent` / `critical_above_percent`.
`/usage-watch now` still prints the detailed multi-line view.

## When it shows up

- A notice appears when a level is reached for the first time (escalation from OK to WARNING, or
  WARNING to CRITICAL). Escalations always show.
- Every `every_n_turns` completed turns, a status line is shown or updated. With `only_when_above: true`
  (default) it stays silent while everything is below `warn_above_percent`.
- After `idle_minutes` without a new turn, the session's latest notice is refreshed (edited, in both modes). This only edits an
  existing notice: a timer has no session context, so it cannot create a new message (see limitations).

## Install

```bash
hermes plugins install <your-git-url-or-owner/repo>   # or copy into ~/.hermes/plugins/usage-watch
hermes plugins enable usage-watch
```

## Configure

Settings live under `plugins.entries.usage-watch.settings`:

| Key | Default | Meaning |
|---|---|---|
| `every_n_turns` | `5` | periodic status every N completed turns (0 = threshold crossings only) |
| `idle_minutes` | `5` | refresh the existing notice after N idle minutes (0 = off) |
| `providers` | `["auto"]` | `auto` = `model.provider` from the profile config, or an explicit list |
| `warn_above_percent` | `80` | warning level |
| `critical_above_percent` | `95` | critical level |
| `only_when_above` | `true` | stay silent below the warning level |
| `mode` | `repost` | `repost` = new one-line message per update, `edit` = edit the session's notice in place |
| `enabled` | `true` | master switch (`/usage-watch on|off` overrides it per profile) |

```bash
hermes config set plugins.entries.usage-watch.settings.every_n_turns 3
```

## Slash command

`/usage-watch status` (config and state), `now` (fresh fetch, printed as a reply, not posted),
`on`, `off`.

## Providers

Usage data comes from `agent.account_usage.fetch_account_usage(provider)`. Providers that have a usage
fetcher work: `anthropic` (Claude OAuth), `openai-codex`, plus any provider plugin that implements
`fetch_account_usage`. Providers without usage data are skipped silently.

**Nous is not covered.** Its credit view is only exposed as rendered text by a public helper, not as
structured windows, so there is nothing to threshold.

## Caching and failures

- Usage is cached for 60 seconds per provider.
- "No usage data" results are retried after 10 minutes.
- Exceptions and rate limits (429) back off exponentially, capped at 15 minutes. The last good
  snapshot keeps being shown meanwhile.
- Nothing raises into the agent loop. Hook errors are logged as warnings, state-file errors go to `errors.log`.

## Troubleshooting

- `/usage-watch status` prints the last turn decision (level, whether it notified, the post/edit outcome,
  and what each provider's fetch returned: fresh, cached, backoff, empty or error). The same record is in
  `<hermes home>/plugin-data/usage-watch/last.json`.
- `errors.log` in the same folder keeps tracebacks from the state code.
- After `hermes plugins update`, a running gateway keeps the old code until it restarts:
  `hermes gateway restart` from a shell outside the gateway.

## Where notices appear

- CLI one-shot (`hermes chat -q`) and Desktop/TUI (local sessions): posted as a keyed notification the viewer replaces in place.
- Messaging gateways (Telegram, Discord, ...): posted as a chat message and edited through the adapter.
- API server: not supported. It has no message surface for plugins, so posts are refused (`send_failed`)
  and logged. Slash commands are not routed there either.

## Limitations (honest)

- **Idle can only edit, not create.** `post_message` needs a bound session, and timer threads have none.
  If a session never got a notice before going idle, nothing is shown until the next threshold event.
- **Threshold changes take up to 60 s to appear** because of the cache. `/usage-watch now` forces a fetch.
- **Counters are in memory.** A gateway restart resets turn counts and the "last level" memory.
- **Percent is across all windows**, including model-only windows (e.g. Opus weekly), so a model cap can
  raise the level. The notice marks those windows as `model-only`.
- **Per-session provider is not tracked.** `auto` reads the profile's `model.provider`; a `/model` switch
  mid-session is not followed.

## Disclosure (catalog rule 13)

- Network: calls the provider usage endpoints through Hermes' fetchers (for Anthropic, the OAuth usage
  API). It uses the credentials Hermes already holds. It does not read credential files itself.
- Background: one daemon `threading.Timer` per active session (idle refresh), cancelled on
  `on_session_finalize` (session teardown).
- Reads: `config.yaml` (read-only, to find `model.provider`). Writes: a tiny `state.json` under
  `plugin-data/usage-watch/`, only when you run `/usage-watch on|off`.
- No telemetry.

## Development

```bash
python3 -m pytest -q tests      # fetch is mocked; no network
hermes plugins validate . --install-deps
```

## License

MIT, see [LICENSE](LICENSE).
