# Semantic Auto-Router

OpenAI-compatible HTTP proxy that classifies each incoming prompt into one of nine
**semantic domains**, picks a **tier** from a tier table, walks a **fallback
chain** through that tier, and survives upstream failures with **circuit
breakers**, **multi-layer caching**, and a **per-session escalation judge +
advisor gate**.

The semantic layer (the `smart-prompt-router` module) is the heart: an
embedding model (`all-MiniLM-L6-v2`) scores the prompt against a curated
bank of example sentences for nine domains — `trading_emergency`,
`trading_decision`, `trading_info`, `coding_hard`, `coding_routine`,
`system_admin`, `research`, `analysis`, `creative`, plus `personal` and
two cheap-chat domains. The picked domain is resolved through a
`DOMAIN_TO_TIER` table to a model chain. Every layer below it
(classification cache, response cache, circuit breaker, fallback, stats)
sits on top of the same routing decision and is independently
auditable.

This repository is the **semantic core + shared modules**. It does not
include the per-profile `*-router-proxy.py` wrappers (those are
deployment-specific bindings to a particular user/client) or the
per-profile systemd units. To deploy it you write your own
`my-router-proxy.py` (a thin `BaseHTTPRequestHandler` that calls into
`smart-prompt_router.classify(...)` + `router_escalation.run(...)`),
copy one of the three sibling proxies referenced in the skill as a
template, or run `wire_escalation.py` on your own proxy to add the
stats/escalation/advisor block.

## Why semantic and not keyword?

A keyword router fails on phrasing it has never seen. The semantic
layer recognizes *meaning*: "the momentum is fading, i'm underwater" is
recognized as a trading prompt without the word "stock", and "flatten
everything now" is recognized as `trading_emergency` because its
embedding is close to the curated emergency examples — not because the
word "flatten" is in a keyword list.

Two non-obvious wins:

1. **Panic phrases get the right tier.** Ejection phrasings
   ("flatten everything", "kill all my trades", "get me out") score
   near-zero on the embedding path without example coverage in the
   bank. Belt-and-braces: the library ships panic phrases both in the
   `trading_emergency` keyword list (instant, exact) AND in the
   `DOMAIN_EXAMPLES` (semantic coverage for paraphrases).
2. **Drift detection is a `grep`.** A misclassified prompt shows up as
   a `x-domain` value that doesn't match the prompt's content. Sweep
   a non-specialty sample through the proxy and read `x-domain` for
   clusters that shouldn't be there. Three-way comparison
   (live proxy vs offline `classify()` vs offline `classify_fallback()`)
   catches the silent-degradation failure mode where the proxy serves
   keyword classification while claiming semantic.

## What you get

| Module | Role |
|---|---|
| `smart-prompt-router.py` | Classifier + 9-domain example bank + `classify()` / `classify_fallback()` + tier assignment + session cache + length-gated caps |
| `router_escalation.py` | Stats ring buffer + per-session escalation judge + advisor gate + capability flags + fail-open semantics |
| `router_tiers.py` | Tier table (`tier-reasoning` / `tier-standard` / `tier-fast` / `tier-fallback` / `tier-quota-burn`) with fallback chains per tier |
| `wire_escalation.py` | Idempotent patcher that adds the stats/escalation/advisor block to a `*-router-proxy.py` file. `--check` first, then apply. |
| `anthropic-shim.py` | Anthropic-format → OpenAI-format shim (port 8903 → 8898) for clients that only speak Anthropic's API |
| `switchyard_signals.py` | NVIDIA Switchyard port: shadow-mode routing signals (severity / source / confidence) computed without affecting live routing |
| `config/config.yaml` | Tier catalog + backend reference for upstream OpenAI-compatible gateways |
| `config/envoy.yaml` | Envoy listener config example |
| `test_escalation_advisor.py` | 75-case unit suite for escalation judge + advisor gate + capability flags (stdlib only) |

## Architecture

```
Request (POST /v1/chat/completions)
    │
    ├─> Conversation-keyed response cache
    │     └─> HIT  → re-emit x-domain / x-model-used, return cached body
    │     └─> MISS ↓
    │
    ├─> Classifier (smart_prompt_router.classify)
    │     ├─ last-3 user-message candidates, boilerplate-filtered
    │     ├─ max-confidence embedding match against 9-domain bank
    │     ├─ per-domain cap (0.30 for research/creative/etc.) on short prompts
    │     ├─ trading floor gate (don't demote a trading_ classification below 0.18)
    │     └─ demote-to-fast on borderline general_easy/personal/creative
    │
    ├─> Tier resolution (DOMAIN_TO_TIER)
    │     ├─ tier-reasoning (frontier) — Claude Opus/Fable
    │     ├─ tier-standard — Claude Sonnet / GLM
    │     ├─ tier-fast — Claude Haiku / GLM-flash / qwen-flash
    │     ├─ tier-fallback — chains across all cheap lanes
    │     └─ tier-quota-burn — GLM / MiniMax-M3 / qwen-flash (always cheap)
    │
    ├─> Fallback chain walk (per tier)
    │     ├─ Circuit-breaker check (60s reset per model)
    │     ├─ Primary call (first model in chain)
    │     │     └─ on 4xx/5xx → record failure, try next model
    │     ├─ max_tokens floor (1500) for reasoning models
    │     └─ SSE → JSON collapse (Claude via 9Router sometimes returns SSE for stream:false)
    │
    ├─> Response cleanup
    │     ├─ _strip_think — remove <think>...</think> blocks
    │     └─ _strip_preamble — remove "sorry hit send too early" preambles
    │
    ├─> Escalation judge (cheap lane, opt-in domains)
    │     └─ per-session consecutive-streak; verdict None → HOLD, escalate → latch
    │
    ├─> Advisor gate (opt-in domains, terminal turns)
    │     └─ APPROVE → replay draft; REDO → discard + re-invoke with plan
    │
    └─> StatsTracker.record() → GET /v1/stats + GET /metrics?format=prom
```

## Install

```bash
git clone https://github.com/Omegabrotein/semantic-router.git
cd semantic-router
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the tests (no upstream required)

```bash
python3 test_escalation_advisor.py
```

The test file uses only stdlib + the modules in this repo. It does
not require a live proxy or any upstream gateway. Green = the
semantic core is correctly wired.

Deployment-specific tests (cross-profile config drift, multi-proxy
sanity, live-judge smoke) live with the proxy wrappers, not in this
repo — they're tied to your specific systemd unit layout.

## Run the proxy (template)

The `*-router-proxy.py` files are not included (they are
deployment-specific). A minimal template:

```python
#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json, urllib.request

import smart_prompt_router as SPR
import router_escalation as RE
import router_tiers as RT

LISTEN = ("127.0.0.1", 8898)
UPSTREAM = "http://127.0.0.1:20128"   # your OpenAI-compatible gateway


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404); return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        req = json.loads(body)
        msgs = req.get("messages", [])
        cls = SPR.classify_conversation(msgs, msgs[-1].get("content", ""))
        domain = cls["domain"]
        demoted = cls.get("demoted_to_fast", False)
        tier = RE.resolve_tier(domain, demoted=demoted)
        chain = RT.FALLBACK_CHAINS.get(tier, RT.FALLBACK_CHAINS["tier-fallback"])
        for model in chain:
            try:
                upstream_req = urllib.request.Request(
                    f"{UPSTREAM}/v1/chat/completions",
                    data=json.dumps({**req, "model": model}).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": self.headers.get("Authorization", "")},
                )
                with urllib.request.urlopen(upstream_req, timeout=180) as r:
                    out = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(out))
                self.send_header("x-domain", domain)
                self.send_header("x-model-used", model)
                self.end_headers()
                self.wfile.write(out)
                return
            except Exception as e:
                continue
        self.send_error(503, "All models unavailable")

if __name__ == "__main__":
    SPR.preload()
    srv = ThreadingHTTPServer(LISTEN, Handler)
    print(f"listening on {LISTEN}")
    srv.serve_forever()
```

Then wire your client to `http://127.0.0.1:8898/v1` with
`model: "auto"` and the router takes over.

## Configuration

All knobs are environment variables. Defaults match a small-footprint
deployment; tune to taste.

| Env var | Default | Effect |
|---|---|---|
| `TRADING_FLOOR` | `0.18` | Below this score, a `trading_*` classification is demoted to fast tier regardless of domain |
| `LOW_CONF_DEMOTE` | `0.18` | Below this score (non-trading), request is routed to `tier-quota-burn` |
| `ABSOLUTE_FLOOR` | `0.20` | Below this score, request is treated as `general_easy` (default-detector) |
| `SHORT_PROMPT_CHARS` | `120` | Length gate for per-domain caps (research/creative/coding_routine) |
| `ESCALATION_BUDGET_PER_HOUR` | `12` | Global hourly cap on latches per session |
| `ESCALATION_CONFIRMATIONS` | `2` | Consecutive escalate verdicts required to latch |
| `ADVISOR_MODEL` | `cc/claude-fable-5-1` | Frontier reviewer for `trading_decision` / `coding_hard` terminal turns |
| `ADVISOR_MIN_TOOL_RESULTS` | `3` | Don't fire `no_tool_call` trigger until 3 tool results have accumulated |
| `ADVISOR_STALL_TURNS` | `30` | Fire `stall` trigger after N assistant turns without triggering |
| `ADVISOR_MAX_REVIEWS` | `3` | Per-session budget for advisor reviews |
| `RESPONSE_CACHE_TTL_S` | `900` | Time-to-live on cached completions (set to 0 to disable) |
| `RESPONSE_CACHE_MAX` | `5000` | LRU cap on response cache |
| `JUDGE_MODELS` | (chain) | Comma-separated fallback chain for the escalation judge |

## Verification (the five-step audit)

Status codes prove almost nothing. Reasoning models return 200 with
`content: ''` when `max_tokens` is tight. The classifier module can be
imported fine while the proxy silently serves keyword classification.
Verify the proxy is actually wired by running these in order:

1. **Wired, not merely alive** — `curl http://127.0.0.1:8898/health`
   (your proxy's health endpoint, if any) returns 200 AND asserts
   `method == "semantic"` for the classifier.
2. **Every tier actually routes.** Send one prompt per domain and read
   `x-model-used` / `x-domain` response headers. A table of
   domain → model chosen is the only real proof.
3. **Content is non-empty.** Assert on `choices[0].message.content`,
   not on HTTP status. Test at a deliberately tight `max_tokens` to
   catch the reasoning-model empty-content failure mode.
4. **Tool calls survive the proxy.** Send a request with a `tools`
   array and confirm `tool_calls` comes back populated. Test streaming
   (`stream:true`) separately from non-streaming.
5. **Zero failures / no open breakers** in `/metrics` after the sweep.

Run all five against every proxy instance you patch, not just the
default one — multi-profile copies drift silently.

## Pitfalls (the silent-degradation failure modes)

These are documented in detail in the test files' docstrings, the
module comments, and the upstream [Hermes skill for semantic
routing](https://hermes-agent.nousresearch.com/docs). The short list:

- **Classifier alive but silent-keyword fallback.** Three-way
  differential probe: live proxy `x-domain` vs offline `classify()`
  vs offline `classify_fallback()`. Disagreement = silent
  degradation.
- **Per-domain caps matter.** Without the per-domain cap on
  research/creative, `explain X in one paragraph` lands on
  Sonnet when it should land on Haiku.
- **Reasoning models return empty content** at low `max_tokens`.
  Floor the budget in the proxy forward path (`payload['max_tokens']
  = max(payload.get('max_tokens', 0), 1500)`).
- **SSE → JSON collapse.** Upstream OpenAI-compatible gateways
  sometimes respond with `text/event-stream` for `stream:false`
  requests. Branch on whether the body starts with `data:`.
- **Cache key on last message collides with framework boilerplate.**
  "You just executed tool calls but returned an empty response" is a
  Hermes nudge that appears at the end of every retry; keying the
  response cache on the last message replays the same cached answer
  to every different conversation that ends in that string. Key on
  the whole conversation + TTL.
- **Orphan processes mask a systemd crash loop.** `systemctl --user
  is-active` shows `activating (auto-restart)` while a manually
  started orphan squats on the port. Verify the `MainPID` from
  `systemctl --user show` matches the PID actually holding the port
  in `ss -ltnp`.

## License

MIT. See `LICENSE`.

## Credits

Built by [Omegabrotein](https://github.com/Omegabrotein) and
[Hermes Agent](https://hermes-agent.nousresearch.com/docs). The full
"why" and operational pitfalls live in the companion Hermes skill
`semantic-routing-auto-model`.
