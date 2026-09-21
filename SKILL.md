---
name: semantic-routing-auto-model
description: "Build, debug, or audit a semantic LLM router — pick model per prompt, fix routing that's silently serving keyword classification, verify every tier actually routes, harden against the proxy-still-alive-but-classifier-dead failure mode, port a foreign router (Switchyard, LiteLLM) into shadow mode, set up escalation/advisor gates, or add multi-tier caching and circuit breakers to an OpenAI-compatible proxy. Use when routing is wrong, the proxy is alive but misclassifying, model output leaks (think blocks, 'sorry hit send too early' preambles), or the user wants auto-routing across multiple Hermes profiles."
version: 1.0.0
author: Omegabrotein
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [semantic-routing, auto-model, caching, circuit-breaker, fallback-chains, classification, llm-proxy]
    related_skills: [hermes-profile-configuration]
    homepage: https://github.com/Omegabrotein/semantic-router
---

# Semantic Routing Auto-Model Systems

Build and deploy production-grade auto-routing LLM systems with semantic classification, multi-layer caching, circuit breakers, and fallback chains.

## Use When

- Building auto-routing that picks best model per query
- Deploying semantic classification for cost optimization
- Setting up intelligent fallback chains (never lose requests)
- Adding multi-layer caching and circuit breakers to LLM proxy
- Auditing routing efficiency from 65/100 → 95/100
- Deploying identical auto-routing across multiple Hermes profiles
- Fixing **leaked model output** (think blocks, self-correction preambles,
  internal chit-chat) bleeding into chat replies — the stripper patterns
  in this skill apply to any "model output hygiene" problem at the proxy
  layer
- **Adding a new response-cleanup step** to existing proxies — every
  new sibling proxy (`*-router-proxy.py`) needs both `_strip_think`
  and `_strip_preamble` wired into `_try_model`
- **Diagnosing a router that silently serves keyword classification** while
  claiming semantic — see
  `references/classifier-degradation-diagnosis.md`
- **Evaluating whether to adopt an upstream router library** (e.g. NVIDIA
  Switchyard, LiteLLM Router) and translating its patterns onto our
  7-tier/9-domain stack — see `references/switchyard-evaluation-2026-09-07.md`
  for the worked example
- **Adding observability, escalation, or an advisor-gate to the router**
  — see the Architecture section below and
  `references/router-escalation-advisor-2026-09-07.md` for the live module
- **Porting a foreign router's algorithm into the stack without
  replacing the existing pipeline** — shadow-mode deployment (compute
  + log, never route); see Components §0d and Switchyard steal #5
- **Honest `/v1/models` capability flags** so client SDKs (Codex,
  Claude Code, etc.) don't drop images or refuse tool calls — see
  "Capability flags" under Architecture

## Architecture

```
Request
  └─> Classification cache check (conversation-keyed)
       └─> miss → 9-domain embedding classifier (all-MiniLM-L6-v2)
       └─> tier demotion if score < ABSOLUTE_FLOOR or trading floor
  └─> Response cache check (conversation-keyed, stream bypassed)
  └─> Tier selection (DOMAIN_TO_TIER, containment-preserving)
  └─> Escalation latch check (per-session consecutive-streak; see below)
  └─> Primary call (first model in tier chain)
       └─> Circuit breaker per model (60s reset)
       └─> Fallback chain if primary fails — cheap lane vs frontier lane
  └─> Advisor gate (opt-in domains only; cc/claude-fable-5-1 by default)
       └─> APPROVE / REDO verdict → REDO discards draft and re-invokes executor
  └─> Escalation judge (cheap-lane unlatched turns; per-session streak)
       └─> Verdict moves streak; latch routes subsequent turns to Fable
  └─> Response cleanup (_strip_think, _strip_preamble)
  └─> StatsTracker.record()  →  GET /v1/stats + /metrics?format=prom
```

Query → Classification → Tier Selection → Fallback → Circuit Breaker → Upstream
→ Cache → Metrics, with the escalation + advisor + stats layer wired between
fallback and metrics.

## Components

### 0. Observability, escalation, and advisor layer (`router_escalation.py`)

The router ships with a shared module — `~/semantic-router/router_escalation.py`
— that adds three orthogonal capabilities on top of the existing tier system.
All three are wired into every sibling proxy by the idempotent patcher
`wire_escalation.py` so multi-profile copies cannot drift.

**StatsTracker — per-domain / per-model spend accounting**

A ring buffer of the last N requests (default 5000) recording `{ts, port,
domain, tier, model, tokens_in/out, latency_ms, demoted, lane, cache_type,
escalated, advised, fallback_reason}`. Surfaced at:

- `GET /v1/stats` — full JSON snapshot, counters and per-bucket aggregates.
- `GET /v1/stats` also exposes `routing_overhead.judge` and
  `routing_overhead.advisor` so judge/advisor cost is NEVER mixed into
  served-model tokens. Without this split, the cheap lane looks more
  expensive than it is and you mis-tune budgets.
- `GET /metrics?format=prom` — Prometheus text format with the same
  dimensions (`domain`, `model`, `tier`, plus overhead kind).

**EscalationTracker — judge-the-completed-turn pattern**

Per-session consecutive-escalate streak with a global hourly budget
(default 12/hr, env `ESCALATION_BUDGET_PER_HOUR`). Layered BEHIND the
classifier — the classifier still picks the starting tier; escalation
only moves a session up within the lane the classifier chose, never
across lanes. Gated to non-demoted cheap-lane domains only:

```python
ESCALATION_DOMAINS = {general, general_easy, personal, creative,
                      coding_routine, system_admin, trading_info}
```

This is what preserves the containment invariant: a `__demoted__` request
escalates within the cheap chain and never leaks into Claude.

**Fail-open semantics are explicit and tested.** A judge timeout, error,
or unparseable verdict must NOT manufacture an escalation. Concretely:
`None` verdict → HOLD streak unchanged (do not reset, do not latch). `decline`
→ reset streak to 0. `escalate` → increment streak; latch at
`ESCALATION_CONFIRMATIONS=2`. Latches are budget-capped; when the budget
is exhausted, the verdict still increments the streak but is denied the
latch and bumps the `escalation_budget_denied` counter so it is visible.

**AdvisorGate — terminal-turn review by a stronger model**

Default advisor is `cc/claude-fable-5-1` (verified live in 9router catalog).
Triggers a review only on terminal turns in opt-in domains (default:
`trading_decision`, `trading_emergency`, `coding_hard`):

- `no_tool_call` trigger fires only after `ADVISOR_MIN_TOOL_RESULTS=3`
  tool results have accumulated — skips early chatty turns.
- `stall` trigger fires after `ADVISOR_STALL_TURNS=30` assistant turns
  without ever triggering — catches executors that grind without declaring
  done.

On `APPROVE` the draft replays unchanged. On `REDO` the draft is
**discarded** (client never sees it) and the advisor's plan is appended
as user feedback before the executor is re-invoked. Per-session budget
`ADVISOR_MAX_REVIEWS=3` and `fail_open=true` — any advisor failure
releases the original turn unchanged and bumps `advisor_failed`.

The gate deliberately skips when the executor is already a frontier model
(`cc/...`). Reviewing Opus with Opus-1 is no lift and double-bills Claude.

### 0b. Capability flags (`/v1/models`)

Switchyard PR #567 measured this at the wire: Codex reads
`input_modalities` from the model card and, when it reads text-only,
replaces an attached image with the literal string
`"image content omitted because you do not support image input"` BEFORE
sending. Outbound 248 KB with the text-only declaration vs 739 KB with
`["text","image"]`. The image may never have left the client.

The proxies expose `GET /v1/models` returning an OpenAI-shaped body where
each tier's `input_modalities` is the **AND** across every model in that
tier's chain. A chain that can fall through to a text-only model
**must not** advertise vision, or we hand an image to a backend that
cannot read it. Fail closed. Tier-level capabilities table lives in
`router_escalation.MODEL_CAPABILITIES`; unverified models default to
`vision=False` rather than True.

If a client still sends an image and the tier is text-only, the proxy
forwards the request unchanged — the client SDK is responsible for
inspecting `/v1/models` and refusing to attach the image in the first
place. The proxy does not synthesize or drop image content itself.

### 0c. Wiring new proxies and rebuilding existing ones

`~/semantic-router/wire_escalation.py` is the idempotent patcher that
adds the stats/escalation/advisor block to a `*-router-proxy.py` file.
Re-running on a file that already has the block is a no-op (each
insertion is guarded by a marker string). Always run with `--check`
first to see what it would change, then without `--check` to apply.
Backups land in `.bak-<ts>-<fname>` next to the patched file.

The patcher anchors on specific string sequences in the proxy code, so
if you edit `do_POST` substantially the anchor may shift and the patcher
will report `ANCHOR MISS`. Read the proxy source before running the
patcher after a non-trivial edit, and either restore the anchor or patch
manually.

### 0d. Shadow-mode routing hooks (porting patterns without breaking live traffic)

When a foreign router, classifier library, or routing algorithm looks
promising but adopting it would mean **two routers fighting over the
same request** (the existing 7-tier/9-domain pipeline + a new one),
don't replace — port the algorithmic core into a shadow hook that
**computes + logs but never routes**. The full Switchyard steal list and
its update log live in `references/switchyard-evaluation-2026-09-07.md`
section 5; the protocol below is the reusable procedure.

**The protocol — six steps, in order:**

1. **Port + test in isolation.** Write the foreign algorithm as
   pure-Python modules in `~/semantic-router/` (NOT inside a proxy file).
   Tests must pass before the next step; standard library only unless the
   algorithm genuinely requires `numpy`/`torch`. Mirror the upstream test
   names so failures map 1:1 if you need to re-read the source later.
2. **Wire a single hook into ONE proxy.** Add a function that takes
   `(request_data, messages, headers)`, calls the ported module, and
   appends ONE JSONL line per request. Wrap the call in
   `try/except Exception: logger.debug(...); pass` — a port bug must
   never break live routing. Anchor the hook to the existing session-key
   block (`_sid_hdr = self.headers.get('x-session-id')` line) so it
   fires AFTER session identity is resolved but BEFORE the cache check
   (the cache hit path is short-circuited; you only see fresh requests).
3. **Live-smoke that one proxy.** Restart it, send one cheap prompt,
   confirm the JSONL line parses and the decision field matches
   expectation for a benign prompt (severity=0, source=ambiguous,
   confidence=0.0).
4. **Replicate to the other N proxies.** Same anchor string, same JSONL
   path. One shared log file works — current lines are tagged by session
   key (derived from per-port `X-Session-Id` headers); add a `port`
   field per record if your cross-port analysis needs it. Restart N
   systemd units. Verify all `active`.
5. **Cross-port smoke.** One prompt per port, confirm N distinct records
   with the expected session keys.
6. **Read the JSONL for a week before flipping enforce.** The three
   green lights: (a) override fires only when it should — zero
   false-positive critical escalations on clean prompts; (b) affinity pin
   stays within a stable model family per session; (c) the shadow
   decision's source field agrees with the existing tier pick ≥ 80% of
   the time. Disagreement is informative (it shows where the new
   algorithm would route differently); high agreement is what makes
   enforce safe.

**Pitfalls specific to shadow-mode:**

- **Don't read `x-model-used` from the response inside the hook.** The
  hook fires on the request path, before the existing pipeline has
  decided what to serve. Coupling the hook to the response adds
  complexity the shadow phase doesn't need — join the two at analysis
  time.
- **Don't fail-open on the shadow's own bug.** A failing JSONL write
  should swallow and log debug, but never break the proxy. The wrapper
  is the rule, not a suggestion.
- **Don't back up proxies with `cp foo.py foo.py.bak-<date>-<reason>`**
  for the ported modules in `~/semantic-router/` — those are version-
  controlled alongside the tier table. Back up only the proxy files you
  actually edit.
- **Don't run two shadow hooks at once without namespace separation.**
  If you shadow-score two algorithms in parallel, prefix the JSONL path
  (`switchyard_shadow.jsonl` vs `<algo>_shadow.jsonl`) so the analyzers
  don't accidentally interleave. Live on Sep 15 2026:
  `~/semantic-router/switchyard_shadow.jsonl` for the Switchyard port.

### 1. Semantic Classifier (Embeddings)

```python
from sentence_transformers import SentenceTransformer

class SemanticClassifier:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.domains = {
            'trading': 'margin call liquidated stop loss',
            'coding': 'debug memory leak race condition',
            'general': 'hello what is basic question',
        }
    
    def classify(self, query):
        query_emb = self.model.encode(query)
        scores = {d: self._cosine_sim(query_emb, self.model.encode(text)) 
                  for d, text in self.domains.items()}
        return max(scores, key=scores.get) if max(scores.values()) > 0.7 else 'general'
```

**Key:** Use embeddings (semantic), not keywords.

### 2. Tier Mapping

```python
TIER_SYSTEM = {
    'tier-reasoning': {
        'models': ['claude-opus-5', 'claude-fable-5'],
        'domains': ['trading', 'coding'],
    },
    'tier-fast': {
        'models': ['claude-haiku', 'glm-5.2'],
        'domains': ['general'],
    },
}
```

**Keep `DOMAIN_TO_TIER` keys in sync with what the classifier emits.**
A domain the classifier produces but the map lacks falls through the
`.get(domain, default)` to the default tier — silently, with no error and
no metric. A cheap query then quietly bills your premium plan.

Diff the two sets whenever you touch either file:

```python
import re
proxy = open('smart-router-proxy.py').read()
mapped = set(re.findall(r'"([a-z_]+)":', re.search(
    r'DOMAIN_TO_TIER = \{(.*?)\}', proxy, re.S).group(1)))
print(sorted(mapped))          # compare against classifier's domain labels
```

Real instance: the classifier emitted `coding_routine` while the map had
only `coding_simple` — a dead key. Every routine coding question skipped
the cheap tier and hit a premium model, with metrics showing 0 failures
the whole time.

### 3. Multi-Layer Cache

```python
class MultiLayerCache:
    def __init__(self, max_entries=5000):
        self.classification_cache = OrderedDict()  # domain lookups
        self.response_cache = OrderedDict()        # full responses
        self.model_stats = {}
```

Warm-up: Day 1: 0–10% | Week 1: 60–75% | Month 1: 85–90%

### 4. Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, model, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.open = False
        self.last_failure = 0
    
    def is_available(self):
        if self.open and time.time() - self.last_failure > self.reset_timeout:
            self.open, self.failures = False, 0
        return not self.open
```

### 5. Fallback Chains

```python
FALLBACK_CHAINS = {
    'trading': ['claude-opus-5', 'claude-fable-5', 'minimax-m3'],
    'coding': ['claude-fable-5', 'claude-opus-5', 'minimax-m3'],
    'general': ['claude-haiku', 'glm-5.2', 'minimax-m3'],
}
```

## HTTP Proxy

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, urllib.request

class AutoModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/v1/chat/completions':
            return
        
        body = self.rfile.read(int(self.headers['Content-Length']))
        request_data = json.loads(body)
        query = request_data['messages'][-1]['content']
        
        # 1. Response cache
        cached = cache.get_response(query)
        if cached:
            self._send_response(200, cached, 'response')
            return
        
        # 2. Classify
        domain = classifier.classify(query)
        
        # 3. Fallback chain
        chain = FALLBACK_CHAINS.get(domain, FALLBACK_CHAINS['general'])
        
        # 4. Try models
        for model in chain:
            if not circuit_breaker[model].is_available():
                continue
            
            try:
                response_body = self._forward(request_data, model)
                cache.set_response(query, response_body)
                self._send_response(200, response_body, 'model', model, domain)
                return
            except Exception as e:
                circuit_breaker[model].record_failure()
                continue
        
        self._send_error(503, "All unavailable")
    
    def do_GET(self):
        if self.path == '/metrics':
            metrics = {
                'cache_stats': cache.get_stats(),
                'circuit_breaker_status': {m: cb.get_status() 
                                         for m, cb in circuit_breaker.items()},
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode())
    
    def _forward(self, request_data, model):
        req = urllib.request.Request(
            'http://127.0.0.1:20128/v1/chat/completions',
            data=json.dumps({**request_data, 'model': model}).encode(),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()  # Read FULL body
    
    def _send_response(self, status_code, body, cache_type, model=None, domain=None):
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            if model:
                self.send_header('x-model-used', model)
            if domain:
                self.send_header('x-domain', domain)
            self.send_header('x-cache-type', cache_type)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
```

## Deployment

### Step 1: Create Profile Copies

```bash
cp proxy.py jimmy-router-proxy.py
sed -i "s/8898/8901/" jimmy-router-proxy.py

cp proxy.py mila-router-proxy.py
sed -i "s/8898/8902/" mila-router-proxy.py
```

See `references/multi-profile-local-proxy-deployment.md`.

### Step 2: Deploy

**Default (systemd):**
```ini
[Unit]
Description=Auto Model Router

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/proxy.py
Restart=always

[Install]
WantedBy=default.target
```

**Others (background):**
```bash
python3 jimmy-router-proxy.py &
python3 mila-router-proxy.py &
```

### Step 3: Config

```yaml
model:
  base_url: http://127.0.0.1:8898/v1
  default: auto
```

### Step 4: Test

```bash
curl -X POST http://127.0.0.1:8898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"test"}]}'

curl http://127.0.0.1:8898/metrics | python3 -m json.tool
```

## Verifying a Router (do this before declaring it working)

> **Fail-open layers need their own verification method.** A judge / advisor /
> escalation gate is built to swallow its own errors, so "no errors in the log"
> proves nothing — a crashed layer and a layer that approves everything look
> identical. Verify by asserting a counter MOVED
> (`escalation_judged 0 -> 1`), and treat any fail-open warning that repeats on
> a fixed interval as a dead layer until proven otherwise. Full method,
> including the helper-arity-drift bug class and the gate-clause table that
> silently suppresses test traffic:
- `references/cross-user-shared-modules.md` — when a module is imported by a
  service owned by ANOTHER Linux user: renames break them invisibly (running
  services hold old code), your invariants are not their invariants, and
  skipped tests are not guards. Includes the minimal setfacl grant.
- `references/verifying-fail-open-layers.md`

A 200 status proves almost nothing. Verify these five things, in order —
each catches a failure the previous one misses.

1. **Wired, not merely alive** — see the first pitfall below.
2. **Every tier actually routes.** Send one prompt per domain and read
   the `x-model-used` / `x-domain` response headers. A table of
   domain → model chosen is the only real proof:

   ```bash
   curl -s -D /tmp/h -o /tmp/b http://127.0.0.1:8898/v1/chat/completions \
     -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
     -d '{"model":"auto","stream":false,"messages":[{"role":"user","content":"margin call my stop loss triggered"}],"max_tokens":400}'
   grep -iE '^x-model-used|^x-domain' /tmp/h
   ```

3. **Content is non-empty.** Assert on `choices[0].message.content`, not
   on HTTP status — reasoning models return 200 with `content: ''`
   (see pitfall below). Test at a deliberately tight `max_tokens`.
4. **Tool calls survive the proxy.** Agent frameworks break instantly if
   `tool_calls` are dropped in the forward path. Send a request with a
   `tools` array and confirm `tool_calls` comes back populated. Test
   streaming (`stream:true`) separately from non-streaming.
5. **Zero failures / no open breakers** in `/metrics` after the sweep.

Run this sweep against **every** proxy instance you patched, not just the
default one — multi-profile copies drift silently.

`scripts/verify-router-tiers.sh` runs the whole sweep. The script is
self-contained, takes `--port` / `--key` flags, and uses `KEY` /
`NINEROUTER_API_KEY` env vars. It exits non-zero if any step fails.

`scripts/restart-proxies.sh` performs the canonical clean restart of
all three proxy units (PID-from-ss + `__pycache__` wipe + systemd
restart + MainPID-matches-listener verification). Use this AFTER
editing any of `smart-prompt-router.py`, `smart-router-proxy.py`,
`jimmy-router-proxy.py`, `mila-router-proxy.py`, or `router_tiers.py`.
Never `pkill -f 'router-proxy.py'` from the agent shell — that pattern
kills the agent's own bash if the shell command line contains the
matching string (verified Aug 30, 2026). Override `PORTS` / `UNITS` /
`PROXY_DIR` env vars if your layout differs from the Sep 1 default.

`scripts/preamble_unit_test.py` runs the 12-case preamble stripper
unit suite against any proxy file without needing the proxy to be
running. Use it whenever you edit `_strip_preamble*` functions, before
restarting the proxy:

```bash
$ ~/.venvs/<name>/bin/python3 scripts/preamble_unit_test.py /path/to/<sibling>-router-proxy.py
[PASS] T1 real-leak stripped
  ...
=== 12 passed, 0 failed (of 12) ===
```

`scripts/bench-cheap-models.py` runs a head-to-head benchmark of cheap-tier
models (default: Flash, MiniMax-M3, GLM 5.2, Haiku) on representative
domain prompts and reports per-model pass rate, latency, and per-domain
breakdown. **Use this BEFORE adding a new model to a tier chain** to find
where it actually wins — pass-rate + speed is the only honest signal.
Distinct from `verify-router-tiers.sh` (which only checks routing is wired).

```bash
# Compare candidate models for a tier-slot decision
MODELS='glm-cn/glm-5.3-flash,glm-cn/glm-5.2,minimax/MiniMax-M3' \
  python3 scripts/bench-cheap-models.py
```

## Efficiency Audit: 65 → 95/100

**Phase 1 (Audit):**
1. Baseline metrics
2. Identify gaps
3. Prioritize

**Phase 2 (Optimize):**
4. Add caching (70%+)
5. Add circuit breaker (0 fails)
6. Add rate limiting (150 req/s)
7. Add fallback chains
8. Deploy metrics

**Score:**
```
(routing + quality + reliability + caching + 
 rate_limiting + fault_tolerance + monitoring) / 7
```

**Trajectory:**
- Initial: 65/100
- Cache+CB: 85/100
- Full: 92/100
- Fine-tune: 93–95/100 (diminishing)

See `references/efficiency-audit-and-optimization.md`.

### Orphan Processes Mask a systemd Crash Loop

`systemctl --user is-active` can show `activating (auto-restart)` or
`failed` with a climbing restart counter while the proxy still answers
requests correctly — because a **manually-started orphan process** from an
earlier debugging session is squatting on the port (`OSError: Address
already in use` in the journal). The unit fights the orphan forever and
never actually serves; the orphan has no supervisor, no restart-on-crash,
and no fresh env if you patched `.env`/config since it started.

Detect it: `ps -o pid,lstart,cmd -p <pid from ss -ltnp>` — if the process
start time predates your latest fix and doesn't match the current
`systemctl status` `Main PID`, it's an orphan. Kill it and let systemd bind
the port; only then does `Restart=on-failure` protection actually apply.

### Calling Claude Models Directly Through 9Router Returns SSE Even for `stream:false`

9Router's OpenAI-compatible endpoint (`:20128/v1/chat/completions`) can
respond with `content-type: text/event-stream` and chunked `data: {...}`
lines for `cc/claude-*` models regardless of the `stream` field in the
request body — plain `json.loads(raw)` raises
`JSONDecodeError: Expecting value: line 1 column 1`. Always branch on
whether the body starts with `data:` and parse SSE chunks (concatenating
`choices[].delta.content`) as a fallback path in any one-off script that
calls 9Router directly, not just in the production proxy's forward path.
Responses may also carry a `<think>...</think>` block before the real
answer — strip it before treating the text as final.

## Pitfalls

### Router Is Alive But Silently Serves Keyword Classification (the silent-degradation failure mode)

This is the highest-value audit technique in the skill. The router can be
**alive, wired, returning 200, with healthy `/metrics`** — and routing
every request through the keyword fallback while the classifier module
works perfectly when imported by hand.

The differential probe (three-way comparison on the SAME string):
1. **Live proxy** → read `x-domain` / `x-model-used` response headers
2. **Offline `classify()`** → import the classifier in a fresh process
3. **Offline `classify_fallback()`** → the keyword path, called directly

If (1) matches (3) on every probe and disagrees with (2), the running proxy
is on the keyword path. Full diagnosis, root-cause layers, Fable-review
findings on scoring math and threshold sit in
`references/classifier-degradation-diagnosis.md`. Read that reference
before claiming "semantic routing works" from any test that does not
include this three-way comparison.

Three concise points the rest of this pitfall depends on:

1. **Keyword lists collide with other domains.** `classify_fallback`
   checks **emergency first** and uses substring match. `"exit"` in
   "exit code 1", `"crash"` in "application crashing in production",
   `"forced"`, `"urgent"`, `"should i"` — all fire on ordinary non-trading
   text and burn the most expensive tier. If you keep a fallback at all,
   use **word-boundary regexes** and require **>=2 signals** for a trading
   tier.
2. **The 0.30 absolute threshold sits inside the confusion band.** Correct
   classifications land 0.26-0.78 while genuine general tops out at 0.26;
   creative/coding_hard/research at 0.26-0.27 fall BELOW 0.30 and get
   dumped into general. Lower the floor to 0.20, let `general_easy` win by
   argmax rather than using the threshold as a general-detector.
3. **Abstain direction is the expensive bug.** Below-threshold, error, and
   too-long paths route to the **cheap** tier — backwards under asymmetric
   risk. Abstain should route to **standard (Sonnet)**, with a hard rule
   `if best_domain.startswith("trading_") and best_score >= 0.18: never
   downgrade`. The 50K-char rejection path is the worst offender: a trader
   pasting a portfolio dump gets the cheapest model with confidence 0.0.

### Phase 4 pitfall: a library-tier demotion does nothing unless every proxy honors it

This is the natural follow-up to the threshold pitfall above. Once you
raise the absolute floor to 0.20, borderline creative/coding/research
classifications are no longer "abstained" — they leak through at
0.24-0.27 with full-priced tier mappings (creative→Sonnet,
research→frontier). The library flags them honestly with
`demoted_to_fast: True`, but **the proxy ignores the flag** because it
re-derives the tier from the `domain` field via its own
`DOMAIN_TO_TIER` table. The proxy never reads the library's `tier` key.

Patching the library alone is a silent no-op — tests on `classify()`
look fixed but every real chat still bills Sonnet for chili recipes.

**The convention that closes the gap** (3-step, must be applied to
every sibling proxy, not just `smart-router-proxy.py`):

1. Library emits a new field on low-confidence results:
   `demoted_to_fast: bool` (e.g. when `top_similarity < 0.18` and
   domain doesn't start with `trading_`).
2. Proxy checks that field, and if true, prefixes the cached domain
   with `__demoted__` (an out-of-band sentinel that no real classifier
   domain starts with). Strip the prefix before logging / setting
   `x-domain` so telemetry still shows the original domain.
3. The proxy's tier-resolution step handles `__demoted__*` by routing
   to `tier_quota_burn` (cheap GLM / MiniMax-M3) regardless of what
   `DOMAIN_TO_TIER` would have said.

Two reasons this convention beats alternative approaches:
- **Preserves semantic intent for telemetry.** The original domain
  (`creative` for chili) survives in logs; only the cost assignment
  changes.
- **Cache-key compatible.** Classification cache stores `__demoted__creative`
  rather than `creative`, so repeat queries hit the demoted path
  immediately and don't re-consult the library.

Floor tuning: `LOW_CONF_DEMOTE = 0.18` is empirically right with
`all-MiniLM-L6-v2` — below 0.18 the classifier is essentially guessing,
above it the classification is usually load-bearing. Tune per-model if
you swap embeddings. Trading domains are deliberately exempt: their
asymmetric cost (`false-negative` > `overspend Opus`) means the trading
floor at 0.18 wins over any general demote.

**Apply the patch to EVERY proxy.** Multi-profile copies drift. The
session that landed Phase 4 had to patch `smart-router-proxy.py`,
`jimmy-router-proxy.py`, AND `mila-router-proxy.py` — the Fable-style
end-to-end test caught jimmy/mila serving Sonnet for chili on the
first run because only smart had been patched. Sweep the test against
every proxy, not the canonical one, before declaring done.

### Proxy returns 503 / "All models unavailable" because the caller forgot the Authorization header

`smart-router-proxy.py` and its siblings forward
`'Authorization': self.headers.get('Authorization', '')`. When the caller
omits the header (common from one-off curl/HTTP scripts), the proxy
forwards an empty string, 9Router returns 401, every tier in the
fallback chain returns 401, and the proxy surfaces
`{"error":{"message":"All models unavailable"}}` with HTTP 503.

The 503 is misleading — every upstream is fine, but none will accept
the empty auth. Three diagnostic shortcuts that don't require reading
proxy code:

1. Hit 9Router directly with no auth — if you also get 401 with
   `"Missing API key"`, the problem is upstream auth, not proxy logic.
2. `curl` the proxy WITHOUT `-H 'Authorization: ...'` and check the
   response body — `"All models unavailable"` ⇒ auth issue; `"Bad
   Gateway"` or 5xx ⇒ upstream health; `<stream of upstream error>`
   ⇒ circuit breaker.
3. The 9Router live key is in `~/.hermes/.env` as `NINEROUTER_API_KEY`,
   NOT in `~/.9router/.env`. Export it and pass
   `-H "Authorization: Bearer $NINEROUTER_API_KEY"` to any direct
   `curl` test. (Proxies running under Hermes wire it automatically
   because the gateway injects it; the proxy code itself never reads
   it from disk.)

### Response cache keyed on the LAST MESSAGE replays stale answers verbatim (the "agent is stuck repeating itself" bug)

This is the most user-visible cache failure and it is NOT the empty-content
poisoning described below — the cached content is a perfectly good answer,
just to a *different question*.

`query_text = messages[-1].get('content', '')` plus
`_hash(text[:1000])` means the cache key is **only the final user message**.
Agent frameworks re-send identical trailing strings across completely
unrelated conversations:

- `"You just executed tool calls but returned an empty response..."` (Hermes nudge)
- `"[IMPORTANT: You are running as a scheduled cron job...]"` (cron preamble)
- any tool-result block whose first 1000 chars are identical

Every one of those collides. The first answer produced under that key is
then replayed at 0.0ms for every future turn that ends the same way — so
the agent appears to be looping, answering a question the user asked ten
turns ago, ignoring new input entirely. Real instance (Aug 27, 2026): 27
cache hits on the empty-response nudge and 23 on a tool-output block; the
user saw the same "exit_code=143 / 21/21 passed" paragraph returned to
three different questions in a row.

**Detection:**
```bash
grep -n "RESPONSE CACHE HIT" /tmp/smart-router.log \
  | awk -F': ' '{print $NF}' | sort | uniq -c | sort -rn | head
```
Double-digit hit counts on a *framework boilerplate* string (not a real
user question) is the signature. Confirm in one line:
```python
h = lambda t: hashlib.md5(t[:1000].encode()).hexdigest()
h(convA[-1]["content"]) == h(convB[-1]["content"])  # True => colliding
```

**Fix — three changes, applied to EVERY sibling proxy:**

1. **Key on the whole conversation, not the last message.**
```python
def _conversation_cache_key(messages):
    parts = []
    for m in messages or []:
        c = m.get("content")
        if not isinstance(c, str):
            c = json.dumps(c, sort_keys=True, default=str)
        parts.append(f"{m.get('role','')}:{c}")
    return "\n".join(parts)
```
Then `_conv_key = _conversation_cache_key(messages)` and pass `_conv_key`
to BOTH `cache.get_response(...)` and `cache.set_response(...)`.
Note `_hash` still truncates at `[:1000]` — that is fine once the key is
conversation-wide, because the divergence between two conversations is
almost always in the early messages. If you have very long shared system
prompts, hash the full string instead of the first 1000 chars.

2. **Add a TTL** so a bad entry heals itself. Store
`(body, time.monotonic())` and reject on read when
`time.monotonic() - ts > RESPONSE_CACHE_TTL_S` (900s is a good default).

3. **Never cache contentless completions** — guard `set_response`:
```python
def _is_cacheable_response(raw):
    try: d = json.loads(raw)
    except Exception: return False
    for ch in d.get("choices") or []:
        msg = ch.get("message") or {}
        c = msg.get("content")
        if isinstance(c, str) and c.strip(): return True
        if msg.get("tool_calls"): return True
    return False
```

**Verification that actually proves the fix** (status codes prove nothing):
send two DIFFERENT first-messages that share the SAME trailing nudge, and
assert the two `content` strings differ. Then re-send one identical
conversation twice and assert `x-cache-type: response` on the second —
that proves you fixed the collision without disabling caching. Run both
against all three ports.

#### Four follow-on bugs the obvious fix introduces (all found in adversarial review — do these in the SAME change)

Switching the key from last-message to whole-conversation is necessary but
creates worse bugs if you stop there. An adversarial reviewer (Claude Fable)
caught all four; each was then reproduced empirically.

**(a) `_hash(text[:1000])` becomes CATASTROPHIC once the key is
conversation-scoped.** The conversation key starts with
`system:<system prompt>`, and every agent profile front-loads a system prompt
well over 1000 chars. So every conversation in a profile shares the first 1000
chars → **one cache key per profile** → all conversations collide. That is
strictly worse than the bug you set out to fix, and the TTL is the only thing
masking it. Reproduced: a 1542-char shared system prompt made "What is 2+2?"
and "Flatten all my positions now" hash identical.
Fix: `hashlib.sha256(text.encode())` with **NO truncation**. Hashing 100KB is
microseconds. Never truncate a composite key.

**(b) The CLASSIFICATION cache has the same collision AND no TTL.** It is a
separate `OrderedDict` keyed on the same last-message string, so a generic
nudge caches one domain that every later conversation inherits — including a
trading emergency, which then gets forced to the cheap tier and lives there
until LRU eviction at 5000 entries (effectively forever). Key it on the
conversation too and give it its own TTL. Also **stop storing the Phase-4
`'__demoted__'+domain` string sentinel**: store
`{'domain': ..., 'demoted': bool}` and re-derive the tier per request. The
sentinel both forces the cheapest tier permanently and leaks into `x-domain`.

**(c) The real root cause is that `classify()` only ever sees
`messages[-1]`.** Fixing both cache keys is still not enough — measured live:
the nudge string alone classifies `system_admin @ 0.25`, so a trading
emergency two turns up was routed to MiniMax instead of Opus even with correct
keys. The classifier must see the conversation:

```python
_CLASSIFY_LOOKBACK = 3
_BOILERPLATE_PATTERNS = [re.compile(p, re.I) for p in (
    r"you just executed tool calls", r"returned an empty response",
    r"^\s*\[?important:\s*you are running as a scheduled cron",
    r"please (respond|continue)\.?\s*$",
    r"^\s*(continue|resume|go on|proceed)\.?\s*$")]

def _classify_conversation(messages, fallback_text):
    cands = [m["content"] for m in messages
             if m.get("role") == "user" and isinstance(m.get("content"), str)
             and m["content"].strip()][-_CLASSIFY_LOOKBACK:]
    if not cands:
        return classify(fallback_text)
    cands = [c for c in cands if not any(p.search(c) for p in _BOILERPLATE_PATTERNS)] or cands
    return max((classify(c) for c in cands), key=lambda r: r.get("confidence", 0))
```

Two design points that were empirically necessary, not stylistic:
- **Take max-confidence over candidates; do NOT concatenate them.** Joining
  the turns into one blob scored "fun fact about otters" + nudge as
  `trading_decision @ 0.196` — a false positive that bills Opus for chit-chat.
- **Filter boilerplate, with a fallback to the unfiltered list** (`or cands`),
  so a conversation consisting *only* of boilerplate still classifies.

Validated on 7 cases: otters+nudge→general_easy, panic+nudge→trading_emergency
(0.959), code+cron→coding_hard, real sysadmin→system_admin, plain chat, nudge
alone, creative.

**(d) The key must include `stream`, `model`, `tools`, `temperature`.**
Otherwise a `stream:true` request matching a cached non-streaming entry gets a
JSON blob where the client expects SSE; the client's retry-on-empty machinery
then re-fires the nudge — **the original symptom returns from a different
direction**. Simplest safe option: bypass the response cache entirely when
`stream` is truthy, on both get and set.

Also worth doing while you're in there: include `tool_calls` in the message
serialization (an assistant tool-call turn has `content: None`, so tool-heavy
conversations serialize to `"assistant:null"` and collide); only `popitem()`
when inserting a genuinely new key (overwriting a hot key otherwise evicts an
unrelated victim); and drop the response TTL to ~60s, since chat completions
are not idempotent and a 15-minute TTL replays identical answers to legitimate
retries.

#### Restart the proxies through systemd, not as background orphans

While fixing the above I twice restarted the proxies as plain background
processes and then measured **stale code** — the units
`{smart,jimmy,mila}-semantic-router-proxy` already existed and were `inactive`
because orphans held the ports (their `ExecStartPre` port-guard makes systemd
refuse to start rather than fight). Always:

```bash
pkill -f "router-proxy.py"; sleep 5
systemctl --user start smart-semantic-router-proxy jimmy-semantic-router-proxy mila-semantic-router-proxy
for u in smart jimmy mila; do systemctl --user show -p MainPID -p NRestarts --value $u-semantic-router-proxy; done
```
Then confirm the `MainPID`s are the processes actually holding 8898/8901/8902
in `ss -ltnp`. If they differ, you are testing an orphan.

#### Do not read `model_failures` without checking WHY

After the sweep, `/metrics` showed `failures=8` on 8898 and `5` on 8902 — all
of them were my own deliberately-unauthenticated control probes returning 401,
not real upstream problems. Confirm before reporting a regression:
```bash
journalctl --user -u smart-semantic-router-proxy --since "10 min ago" \
  | grep -i "failed after 3" | grep -v 401     # empty => all failures were auth probes
```

### The "empty response / model loop" can be the response cache, not the model

When the same prompt string is sent through the proxy and the model
emits nothing, the proxy stores the empty response in its
in-memory `OrderedDict` (LRU, no TTL) and serves it on every
subsequent identical request in ~0ms. Result: a model that appears
to be in a loop or returning empty responses is actually serving a
**stale cached empty string from a prior turn**.

Distinct from "reasoning model returned empty content at low
`max_tokens`" — that one produces a fresh 200 every time with
`content: ''`, this one produces identical bytes every time at near-zero
latency.

**Detection:** tail the proxy log for `RESPONSE CACHE HIT` at
`< 1ms` latency on prompts that look like new turns. If you see
dozens of cache hits at 0.0ms on a prompt that just changed,
the cache is poisoned.

**Fix (all three):**
1. Restart the proxy process — clears the in-memory cache.
2. Add a TTL on `response_cache.set` keyed on `time.monotonic()`;
   default ~1h is plenty for dedupe and breaks poisoning.
3. Don't cache responses with `content == ''` or `finish_reason == length`
   when `max_tokens < 1000` — reasoning-only outputs are reusable
   per-prompt-input but their empty-content variants poison the cache
   for every retry.

The 3 layers that let this degrade silently are also covered in the
reference: import swallow (`HAS_ML=False` never surfaces), `classify_semantic`
returning `None` on any exception (outer try/except is dead code), and
`_MODEL` staying `None` after a failed load (every request pays a network
timeout + gets the wrong answer). The fix is to load the model **at startup
before binding the port**, set `HF_HUB_OFFLINE=1` with a pinned local
snapshot (or ship precomputed `.npz` embeddings), crash on failure, and
make `/health` assert `method == "semantic"` for every probe.

### `bl config agent` (Alibaba Bailian CLI) silently overwrites the `model:` block

Running `bl config agent --agent hermes --model <slug> --key <key>` rewrites
`~/.hermes/config.yaml`'s entire `model:` block — base_url flips from the
semantic proxy (`:8898`) to Alibaba's Token Plan endpoint, `default: auto`
becomes the qwen slug, and the RAW API key lands inline in the file
(replacing `${NINEROUTER_API_KEY}`). Verified Sep 7, 2026: everything else
in the file survives; only `model:` is clobbered. `bl` also writes
`~/.npmrc` and `~/.bailian/` (telemetry on by default).
Recovery: restore the `model:` block from the newest `config.yaml.bak.*`
(the bl run itself creates one), confirm no `sk-` key remains inline
(`grep -c 'sk-' config.yaml`), then restart the gateway. If the user runs
`bl` regularly, expect this regression after every `bl config agent` call.

### `auto` vanishes from the /model picker after switching models mid-session

A raw-`base_url` `model:` block (provider `custom`) only shows in the
picker as "Custom endpoint" WHILE it is the session's current model. The
moment the user switches the session to any named provider (e.g.
`custom:9router`), the raw-custom row disappears and `auto` is
unreachable from the picker — the user reports "i dont see the auto".
Fix: add the proxy as a NAMED entry in `custom_providers:` so it is
always listed:

```yaml
custom_providers:
  - name: semantic-router
    api_mode: chat_completions
    base_url: http://127.0.0.1:8898/v1
    key_env: NINEROUTER_API_KEY
    context_length: 1000000
    models: {auto: {}, tier_primary_critical: {}, tier_primary_complex: {},
             tier_primary_standard: {}, tier_primary_fast: {}, tier_fallback: {},
             tier_general_minimax: {}, tier_quota_burn: {}}
```

Then the user can return via `/model semantic-router:auto`. Gateway must
restart once to refresh the picker snapshot. Verify with
`hermes_cli.model_switch_providers.list_picker_providers(...)` simulating a
non-custom current provider and asserting `has_auto=True` on the
`custom:semantic-router` row.

### Proxy Runs But Nothing Uses It (check this FIRST)

A healthy `systemd` unit proves the proxy is **alive**, not that it is
**wired**. A router can serve zero traffic for weeks while every chat
bypasses it.

Audit both halves before debugging routing quality:

```bash
systemctl --user is-active smart-semantic-router-proxy   # alive?
grep -A3 '^model:' ~/.hermes/config.yaml                 # wired?
```

`model.base_url` MUST point at the proxy port (`http://127.0.0.1:8898/v1`)
and `model.default` MUST be `auto`. If `base_url` points straight at a
provider or at 9Router (`:20128`), the router is bypassed entirely.

Also add `auto` to the `custom_providers[].models` list or it won't appear
in the model selector.

**The gateway caches config at startup** — after editing `config.yaml`,
`systemctl --user restart hermes-gateway`, or the running chat keeps the
old connection and your fix looks like it did nothing.

### Wrong Plan Prefix (billing goes to the wrong account)

On 9Router the slug **prefix selects which connected plan pays**. Same
model, different prefix, different account:

| Slug | Plan billed |
|---|---|
| `nvidia/minimaxai/minimax-m3` | NVIDIA NIM account |
| `minimax/MiniMax-M3` | your MiniMax plan |
| `nvidia/z-ai/glm-5.2` | NVIDIA NIM account |
| `glm-cn/glm-5.3` | your GLM plan |

List the plans actually connected before choosing prefixes:

```bash
sqlite3 -header -column ~/.9router/db/data.sqlite \
  "SELECT provider,authType,name,isActive FROM providerConnections ORDER BY provider"
```

Match every tier slug to a `provider` in that table. A slug that resolves
and returns 200 can still be the wrong plan — HTTP status does not prove
correct billing.

### Don't Claim a Model "Doesn't Exist" Without Probing the Live Catalog

Catalog existence is per-9router-version. Models get added in npm releases
and the running daemon only knows about what shipped with its binary —
`/v1/models` is the source of truth, **not** the npm version you have
installed, **not** external knowledge, and **not** prior-session memory.

Real instance (Aug 30, 2026): asked for "GLM 5.3 Flash" in semantic
routing. Agent asserted the model "doesn't exist — Z.AI doesn't publish a
Flash variant" — wrong. The model shipped in 9router v0.5.59. The pre-update
catalog simply didn't include it yet; the user's intent was correct.

The three-step verification before any "model X doesn't exist" claim:

```bash
# 1. Pull the catalog from the LIVE daemon (not from a stale cache,
#    not from web search, not from memory)
KEY=$(sqlite3 ~/.9router/db/data.sqlite "SELECT key FROM apiKeys WHERE isActive=1")
curl -s "http://127.0.0.1:20128/v1/models" \
  -H "Authorization: Bearer $KEY" \
  | python3 -c "import json,sys; print('\n'.join(sorted(m['id'] for m in json.load(sys.stdin)['data'])))" \
  | grep -i '<pattern>'
# 2. If absent, probe the model anyway — a non-listed slug may still resolve
#    if it's a known alias upstream
# 3. THEN, and only then, conclude it doesn't exist (and even then frame
#    as "not in your catalog as of v<N>" — not as a factual claim about
#    Z.AI's lineup)
```

If you have to say "doesn't exist," say where you looked and what the
live catalog said — never speak from prior knowledge when a one-second
`/v1/models` call is available.

### Reasoning Models Return EMPTY Content

GLM 5.3, MiniMax M3, and Claude Opus/Fable spend **300–500 tokens
thinking before emitting any visible text**. A low `max_tokens` is
consumed entirely by reasoning, and the caller gets a valid HTTP 200 with
`content: ''`:

```
finish_reason: length
content        : ''
usage.completion_tokens_details.reasoning_tokens: 398 / completion: 400
```

This is not an outage and no circuit breaker trips — it silently looks
like the model "said nothing".

**Fix — floor the budget in the proxy forward path**, applied to every
tier (Claude included, not just GLM/MiniMax):

```python
payload = {**request_data, 'model': model}
try:
    if int(payload.get('max_tokens') or 0) < 1500:
        payload['max_tokens'] = 1500
except (TypeError, ValueError):
    payload['max_tokens'] = 1500
```

Verify by testing at the **worst case** (`max_tokens: 200`), not a
comfortable default — a floor bug only shows up under a tight budget.

### Regex Pattern Design Pitfalls for Response-Content Strippers

Three non-obvious pitfalls that bit real implementations and would
re-bite any future agent writing a similar stripper:

**1. `+` vs `*` in separator classes is the difference between
matching nothing and matching everything.** A regex like
`[,.\\s]+(?:i\\s+)?(?:hit|...)` greedily consumes `. I ` between
"sorry" and "hit", leaving nothing for `hit` to match. Use `*`
(zero-or-more) so a single period can sit alone: `[,.\\s]*`. Easy
mistake to make, easy to miss because the test case that fires it
is the one with the period-only separator.

**2. Forgetting to extend the separator class to ALL observed
punctuation.** A regex that handles `.` after "early" but not `,`
silently fails on `"too early, here's the actual answer: ..."`. Test
variants explicitly: period, comma, em-dash, en-dash, no-space,
double-space. The stripper that "works" against the first variant
that fired in production will fail on the second.

**3. The "head-only fast path" silent-truncation bug.** A common
optimization is `head = c[:250]; match against head; if no match,
set content = head` — this silently chops the message to 250 chars
when the preamble isn't in the head. The correct pattern: match
against the head (fast path), but **on no-match, leave content
untouched**. Only overwrite content when `m.end() > 0` for an
actual match. The bug presents as "messages are mysteriously shorter
sometimes" and is hard to spot without a long-input unit test.

### Self-Correction Preamble Leaks to User

Some models (notably MiniMax M3 on long, complex tasks — and observed on
GLM 5.x as well) emit a draft-then-redo preamble at the START of an
assistant message and then continue with the real answer in the SAME
response. Most common form observed (Aug 27, 2026, Telegram gateway
leak):

> "Sorry, hit send too early. Here's the actual answer: …real answer…"

Hermes sees `finish_reason=stop` and forwards the whole string to the
chat surface verbatim. The preamble is the model's internal
self-correction bleeding into the chat.

**Fix — strip in the proxy forward path**, applied after `_strip_think`
so reasoning leaks are removed first. JSON + SSE (streaming) bodies are
both handled. Only fires when the preamble phrase is in the FIRST ~250
chars of the message, so we never mutilate a real answer that
legitimately mentions the words later in the body.

```python
import re as _re_preamble

_PREAMBLE_PATTERNS = [
    _re_preamble.compile(
        r"^\s*(?:sorry|my apologies|i apologize)[,.\s—–\-]*"
        r"(?:i\s+)?(?:hit|pressed|sent)\s+(?:send\s+)?too\s+early"
        r"[,.\s]*here(?:'s| is)\s+the\s+actual\s+answer"
        r"\s*[:\.\-—–]?\s*",
        _re_preamble.IGNORECASE | _re_preamble.DOTALL,
    ),
]
_PREAMBLE_MAX_HEAD_CHARS = 250  # only strip if phrase is near the top


def _strip_preamble_json(raw: bytes) -> bytes:
    try:
        d = json.loads(raw)
    except Exception:
        return raw
    changed = False
    for ch in d.get("choices", []) or []:
        msg = ch.get("message")
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        if not isinstance(c, str) or not c:
            continue
        head = c[:_PREAMBLE_MAX_HEAD_CHARS]
        for pat in _PREAMBLE_PATTERNS:
            m = pat.match(head)
            if m:
                # Consume off the ORIGINAL content (not truncated head)
                # so a preamble that straddles the 250-char boundary
                # is fully stripped.
                tail = c[m.end():].lstrip()
                msg["content"] = tail
                changed = True
                break
    return json.dumps(d).encode() if changed else raw
```

Wire it into the forward path right next to `_strip_think`:

```python
with urllib.request.urlopen(forward_request, timeout=180) as response:
    raw = response.read()
    raw = _strip_preamble(_strip_think(raw))
    return True, raw
```

**Critical regex design notes (from the Aug-27 implementation):**

1. **Separator class is `*` (zero-or-more), NOT `+`** — `+` greedily eats
   `. I ` between "sorry" and "hit", leaving nothing for the verb. The
   `*` form lets a single `.` (period) sit alone between them.
2. **Comma MUST be in `[,.]\s—–\-]*`** after "early" — the variant
   `early, here's` (comma instead of period) was a real observed case.
3. **Head-only fast path**: match against `c[:250]`, but consume from the
   ORIGINAL `c[m.end():]` so a preamble straddling the boundary is fully
   stripped. Don't overwrite content with `head` when no match — that
   silently truncates the message.
4. **Empty-content guard**: skip the strip if `c` is empty or not a string.
5. **Reasoning + preamble chain**: `_strip_preamble(_strip_think(raw))`
   — `_strip_think` first because the preamble can also live inside a
   think block on some models.

**Known limitation: the response-cache path returns `cached_response`
without calling `_strip_preamble`.** If a leaked preamble gets cached,
it stays cached. Fix by also stripping on the cache-hit path before
`self.wfile.write(cached_response)`. Add this to any new router that
copies the smart-router pattern.

**Concrete cache-hit patch** (add to `do_POST` just before the early
return for cache hits):

```python
# Cache hits previously dropped routing metadata, making audits
# show None and look like a routing failure. Re-emit it.
_cd, _ch = cache.get_classification(query_text)
if _ch:
    self.send_header('x-domain', str(_cd))
try:
    _bm = json.loads(cached_response).get('model')
    if _bm:
        self.send_header('x-model-used', str(_bm))
except Exception:
    pass
# ADD: strip preamble on cache-hit path too — a leaked preamble
# once cached stays cached otherwise.
_cached_stripped = _strip_preamble(cached_response)
self.send_header('Content-Length', len(_cached_stripped))
self.end_headers()
self.wfile.write(_cached_stripped)
return
```

Then change the cache-write side to also strip BEFORE caching so future
hits don't re-need the strip:

```python
cache.set_response(query_text, _strip_preamble(resp_body))
```

**Verification (12 cases minimum):**
- Real leaked message from session → preamble stripped, real answer kept
- Clean message → untouched
- Preamble phrase mid-message → preserved (false-positive guard)
- SSE streaming body with preamble → stripped
- All observed variants: `,`/`.`/`—` between "sorry" and verb, both
  `hit`/`pressed`/`sent too early`, both `here's`/`here is`
- Long prefix (300 spaces before preamble) → untouched (head fast-path
  exits)
- Preamble that straddles 250-char boundary → fully stripped
- Tool calls survive intact
- Empty content stays empty
- Reasoning tokens in `usage` are untouched
- False-positive `"Sorry, here's the actual answer: I forgot to add X."`
  (no "hit/pressed/sent too early") → NOT stripped

### Trading-floor gate: a `trading_*` classification below `TRADING_FLOOR` is NOT exempt from demotion

The Phase 4 pitfall above is correct that trading domains should be protected
from demotion **when the classifier is confident**. But the protection must
NOT be unconditional — `if not domain.startswith('trading_')` exempts the
domain from demotion regardless of score, and that creates a real leak
(see `references/phase4c-leak-fix-2026-08-27.md` for the full transcript
with concrete prompts and live test outputs).

- `"how tall is mount everest"` → `trading_decision @ 0.094` → `tier-reasoning` (Opus)
  - The classifier produced `trading_decision` because "everest" / "how tall"
    semantically rhymes with trading lingo (rising, high, etc.).
  - Score 0.094 is far below `TRADING_FLOOR=0.18`, so the trading floor rule
    should refuse to route this to Opus.
  - With an unconditional `if not trading_` exemption, the proxy ignores the
    library's demote signal and spends Opus on a chit-chat question.

The fix is to **gate by score, not by domain prefix**:

```python
cls_score = cls.get('top_similarity', 1.0)
trading_below_floor = (
    domain.startswith('trading_')
    and cls_score < TRADING_FLOOR
)
if cls.get('demoted_to_fast') or trading_below_floor:
    _demoted = True
# ...
if _demoted:
    primary_tier = "tier_quota_burn"
```

Two things to get right:
1. **Mirror the library rule at the proxy.** The library itself decides
   whether a trading prompt is "clear enough to honor" by comparing
   `best_score >= TRADING_FLOOR`. The proxy must replicate this comparison
   because the proxy's `DOMAIN_TO_TIER.get(domain, ...)` would otherwise pick
   `tier_primary_critical` (Opus) for *any* `trading_decision` regardless of
   confidence.
2. **The library must export `TRADING_FLOOR` as a module-level constant.**
   Inside `classify_semantic`, it's read from `os.environ.get(...)` per call
   for live-tuning, but proxies need it at import time for routing
   decisions. Add module-level:
   ```python
   TRADING_FLOOR = float(os.environ.get("TRADING_FLOOR", "0.18"))
   LOW_CONF_DEMOTE = float(os.environ.get("LOW_CONF_DEMOTE", "0.18"))
   ABSOLUTE_FLOOR = float(os.environ.get("ABSOLUTE_FLOOR", "0.20"))
   SHORT_PROMPT_CHARS = int(os.environ.get("SHORT_PROMPT_CHARS", "120"))
   ```
   Then `from smart_prompt_router import classify, preload, TRADING_FLOOR`
   in each proxy. Without this the proxy crashes on import with
   `ImportError: cannot import name 'TRADING_FLOOR'`.

### Per-domain confidence caps catch the "high-conf but wrong-domain" leak

The absolute floor (`LOW_CONF_DEMOTE = 0.18`) handles **low-confidence**
chit-chat like `what is a good recipe for chili → creative @ 0.165`. But two
classes of leak slip past it:

1. **Length-sensitive explainers** that land on a broad-footprint domain at
   medium confidence. Examples observed:
   - `explain what a hash map is in one paragraph` → `research @ 0.261`
   - `give me a quick dinner idea` → `creative @ 0.284`
   - `what does the president do` → `research @ 0.199`
   All three legitimately semantically belong to the domain (the
   classifier is right about the topic) but the *length and depth signal*
   says the user wants a one-liner, not Sonnet-grade output.
2. **Score above the absolute floor but below the genuine-depth band** for
   the domain. Real research queries average 0.40+; chit-chat explainers
   sit at 0.25-0.30.

The fix is **per-domain confidence caps, gated by prompt length**:

```python
PER_DOMAIN_CAP = {
    "research":       0.30,  # real research scores 0.40+; sweep explainers
    "analysis":       0.30,
    "creative":       0.30,  # real creative writing scores 0.40+
    "coding_routine": 0.30,  # one-liners score 0.20-0.30
    "system_admin":   0.30,  # short factual sysadmin questions leak in
}
SHORT_PROMPT_CHARS = 120    # only apply the cap to short prompts

# In classify_semantic, after the trading-floor branch:
elif (
    best_domain in PER_DOMAIN_CAP
    and best_score < PER_DOMAIN_CAP[best_domain]
    and len(prompt) <= SHORT_PROMPT_CHARS
):
    tier_info = DEFAULT_TIER
    demoted_to_fast = True
    demote_reason = f"{best_domain}_cap<{cap}_short<={SHORT_PROMPT_CHARS}"
```

Two design points that were empirically necessary:
- **Length-gate the cap.** A 200-char prompt that lands on research@0.30 IS
  likely genuine research even at the cap. The cap is for chit-chat, and
  chit-chat is short. Without `len(prompt) <= SHORT_PROMPT_CHARS` the cap
  over-fires on prompts the user actually wants depth on.
- **`trade-offs are explicit and per-domain.** Different domains have
  different genuine-depth floors. Don't use a single global cap — research
  explainers and creative one-shots overlap with real research/creative
  requests in different ways than coding_routine leaks do.

Calibrate by scoring a labeled set end-to-end and finding the floor that
eliminates chit-chat explainers without dropping genuine-depth prompts.
The defaults (0.30 for all five domains) work for `all-MiniLM-L6-v2`;
recalibrate per embedding model.

### The `__pycache__` masks whether the running proxy has the new code

Two ways the running proxy silently uses stale code:

1. **`.pyc` cache predates the `.py` source edit.** Python's import system
   uses the source `.py` mtime to decide when to rebuild the `.pyc`. If the
   source was edited but Python still has the old `.pyc` loaded (e.g. in a
   long-running process), the proxy serves the old code until restart.
   Check both: `stat -c '%y' smart-prompt-router.py` vs
   `stat -c '%y' __pycache__/smart_prompt_router.cpython-311.pyc`. The `.pyc`
   should be NEWER than the `.py`.
2. **`systemctl --user restart` doesn't always pick up the new code if the
   ports are held by orphan processes.** The `ExecStartPre` port-guard makes
   systemd refuse to start (`activating (auto-restart)` forever) rather than
   fight an orphan. Symptom in `journalctl`:
   `Port 8898 already in use; code=exited, status=0/SUCCESS` followed by
   `code=exited, status=1/FAILURE`. The unit fights the orphan forever and
   never actually serves; the orphan has no supervisor, no restart-on-crash,
   and no fresh env if you patched `.env`/config since it started.

   Verify the running PID matches `systemctl --user show -p MainPID
   smart-semantic-router-proxy`. If they differ, you are testing an orphan
   and your "fixed it" change never made it to the serving process. Kill
   the orphan (`kill -9 <pid from ss -ltnp>`) and let systemd bind the port.

The clean restart sequence after editing any of the four files
(`smart-prompt-router.py`, `smart-router-proxy.py`, `jimmy-router-proxy.py`,
`mila-router-proxy.py`):

```bash
# NEVER use `pkill -f 'router-proxy.py'` — the regex matches the parent
# shell too if the shell command line contains the string. Real instance
# (Aug 30, 2026): `pkill -9 -f 'router-proxy.py'` killed the agent's own
# bash, returning exit code -9 with no useful output. Always kill by PIDs
# resolved from `ss`:
PIDS=$(ss -ltnp 2>/dev/null | grep -E ':(8898|8901|8902)' \
       | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
for p in $PIDS; do kill -9 $p 2>/dev/null; done
sleep 3
systemctl --user restart smart-semantic-router-proxy jimmy-semantic-router-proxy mila-semantic-router-proxy
sleep 20
ss -ltnp | grep -E ':(8898|8901|8902)'
```

Then verify the PIDs in `ss` output match `systemctl --user show -p MainPID`
for each unit, AND that a quick classification probe returns the new
expected domain (not a cached stale one).

Identical `reasoning_tokens`/`finish_reason` across two runs with
*different* `max_tokens` means you are reading a **cached** response, not
a fresh upstream call. Change the prompt text (not just the parameters)
to force a cache miss, or restart the proxy to wipe it.

Restarting to clear cache resets hit-rate to 0% — expected, not a
regression.

### Quota-cascade silently disables the cheap lane and every dependent feature

A plan's quota hitting its cap is not a normal 4xx — it returns a 503 with
a Chinese-language error body (`code 1310, ...限额将在 YYYY-MM-DD...`)
that is opaque to anyone reading the English log line. The
circuit breaker treats it like any other failure, the breaker trips,
the next request falls through to the next tier in the chain, and the
next tier may itself be quota-capped. Symptom: the cheap lane appears
to be down for days even though no model is actually unhealthy.

Worse, any feature that **calls a hard-coded model out of band** —
escalation judge, advisor, classifier verdict re-check — fails open
silently on the same quota error. The user sees the cheap model answer,
the escalation counter stays at 0, and nothing in the logs says why.

**Detection (before the bill arrives, before you tune the wrong knob):**

```bash
# Probe each plan directly with a tiny request, not via the proxy
for m in glm-cn/glm-5.3-flash glm-cn/glm-5.3 glm-cn/glm-5.2 \
         minimax/MiniMax-M3 cc/claude-fable-5-1; do
  code=$(curl -s -o /tmp/o -w '%{http_code}' -m 60 \
    http://127.0.0.1:20128/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
    -d "{\"model\":\"$m\",\"stream\":false,\"max_tokens\":50,\"messages\":[{\"role\":\"user\",\"content\":\"reply ok\"}]}")
  printf "%-26s HTTP=%s\n" "$m" "$code"
done
# HTTP=200 -> model answers. HTTP=503 with body containing 'code\\\":\\\"1310'
# or 'limit\\\":' or Chinese characters => plan is quota-capped.
```

**Fix — chains, not single models, for any out-of-band call.**

The escalation judge in `router_escalation.py` is a chain
`JUDGE_MODELS = [glm-cn/glm-5.3-flash, glm-cn/glm-5.2, minimax/MiniMax-M3]`,
walked in order until one answers. When you add a new out-of-band call
(advisor, classifier verdict, anything not the user's primary request),
build it as a chain too. Hard-coding a single model means a single
quota event silently breaks that feature, with no log line that says
"the judge is permanently disabled because of plan rate-limit 1310".

Three more things to do at the same time:

1. **Surface the chain in `/v1/stats`** — the `routing_overhead.judge`
   and `routing_overhead.advisor` buckets per-model count calls so a
   quota shift changes which model appears there. If the bucket only
   ever shows `glm-cn/glm-5.3-flash` you know nothing is failing
   through to M3.
2. **When a judge fails open, count it.** `escalation_judged` is
   incremented regardless of whether the verdict was parseable. A
   sustained delta between `escalation_judged` and `(escalation_latched
   + decline_verdicts + escalate_verdicts_that_hit_budget)` indicates
   the judge is failing open silently.
3. **Probe at deploy time, not on first user request.** After any plan
   renewal, run the loop above before declaring the proxy healthy.

### Per-port stats ring buffer resets on every restart

`StatsTracker` keeps its 5000-entry ring and all aggregates in process
memory. Every proxy restart (orphans or systemd) wipes them — counters
go back to zero, `uptime_s` restarts, the recent ring is empty.

Implications:

- A long-running drift in `escalation_latched - escalation_budget_denied`
  ratios vanishes on the next restart. If you need historical data,
  scrape Prometheus (`/metrics?format=prom`) into a TSDB before restarting.
- After editing `router_escalation.py` you will see counters at zero for
  ~30 seconds while traffic warms. That is normal, not a regression.
- Do not rely on `/v1/stats` for SLA numbers — it is operational
  visibility, not historical record. Wire Prometheus scraping before you
  need the data for an incident.

### Classifier Latency (0.5–1s)

**Fix:** Classification cache. At 70%+ hit, 70% skip classification.

### Chunked Responses

**Issue:** 9router uses `Transfer-Encoding: chunked` even for non-streaming.

**Fix:** Read full body, set `Content-Length`, write.

```python
with urllib.request.urlopen(req) as resp:
    body = resp.read()

self.send_response(200)
self.send_header('Content-Length', len(body))
self.end_headers()
self.wfile.write(body)
```

### BrokenPipeError Crashes

**Fix:** Wrap writes in try-except.

```python
try:
    self.wfile.write(body)
except (BrokenPipeError, ConnectionResetError):
    pass
```

### Every proxy must use the SAME Python interpreter

The killer silent failure: a per-profile unit that runs
`ExecStart=/usr/bin/python3` instead of the venv python. System python has no
`sentence_transformers`, so the classifier silently falls back to KEYWORD
matching. The service is `active`, metrics are healthy, requests return 200 —
but semantic routing never happens and hard questions land on the cheap tier.

Symptom: one profile scores worse on an identical routing audit than another
with byte-identical tier config.

```bash
grep -H ExecStart ~/.config/systemd/user/*semantic-router-proxy.service
/usr/bin/python3 -c "import sentence_transformers"   # must NOT fail
```

Fix: point every unit at the same venv interpreter, then
`systemctl --user daemon-reload && systemctl --user restart <units>`.

### Tier config drifts between profile copies

Duplicated proxy files diverge over time. Real drift found in practice: a dead
`coding_simple` key (classifier emitted `coding_routine`) and `general_easy`
mapped to the expensive fast tier instead of quota burn — so secondary
profiles burned the premium plan on trivial chat.

Sync `DOMAIN_TO_TIER` from the canonical proxy and verify with a parity check
comparing hardening flags + tier blocks across all copies
(`~/semantic-router/check_parity.py`).

### sglang/vLLM/ollama binds to Tailscale-only — local judge silently fails open

sglang and vLLM commonly bind to a single interface (Tailscale IPv4/IPv6, or the LAN IP) instead of `0.0.0.0`, so a `ROUTER_JUDGE_LOCAL_URL=http://127.0.0.1:11434/v1` config will 000-refuse-connection from the proxy even when the local model is healthy. Symptom: `escalation_judged` increments but `judge_local_used` stays 0 and `judge_unreachable` doesn't move either — because the cloud chain quietly answers through 9router. The local judge path looks "alive" because `JUDGE_USE_LOCAL` is True, but every local call is hitting `Connection refused`, and the only way to find out is to read the proxy log for `local_judge_unreachable` or grep the journal.

Two fixes, both needed:

1. **`call_judge_local` auto-discovers** the local URL by trying a small list of candidates (configured URL first, then Tailscale IPv4 from `socket.gethostbyname(socket.gethostname())`, then `[::1]`, then `127.0.0.1`). Surface which URL actually answered in the failure case so operator logs are actionable. Verified Sep 8 2026: discovered a Tailscale-bound upstream port (e.g. `<lan-ip>:11434`) by walking the fallback list when the configured `127.0.0.1` refused.
2. **Add `judge_unreachable` and `judge_local_used` / `judge_cloud_used` counters** to `StatsTracker` and bump them right after the chain walk in each proxy. Without these, a local-judge outage + concurrent cloud 429s is invisible — `escalation_judged` increments normally because M3 (the durable tail) catches the call, and the operator sees no error.

Always verify with a three-way probe, not a single curl: `curl http://127.0.0.1:11434/v1/models` AND `curl http://<lan-ip>:11434/v1/models` AND the actual proxy's `/v1/stats` counter. If `ss -ltn` shows the port bound to a non-`0.0.0.0` IP, fix the URL — don't trust the env-var alone.

### Quota-aware tier design (know the plan limits BEFORE assigning tiers)

Ask the user (or read the provider dashboards) for each plan's actual limit,
then let scarcity dictate order — capability alone is the wrong sort key.
Real example that inverted an entire design:

| Plan | Limit | Role |
|---|---|---|
| MiniMax | 13B tokens/month | **workhorse** — primary for general/info/routine |
| GLM 5.3 | 80 prompts/5h, **400/week** | **scarce** — last-resort fallback ONLY |
| Claude Max | subscription, session-capped | quality tier for hard/important work |

The original design had GLM as PRIMARY for easy chat — the highest-volume\ncategory — which would exhaust 400 prompts/week in days. Demoting GLM to\nfallback-last and promoting MiniMax to workhorse cost nothing in quality for\ncasual queries (all models scored 6/6 on domain evals; see\n`model-capability-benchmarking`).\n\n**User can invert this deliberately (Aug 30-31, 2026): \"burn GLM first.\"**\nWhen the user WANTS the small quota consumed before it expires, GLM-first\nis correct: every cheap tier leads `glm-5.3 → glm-5.3-flash → glm-5.2 →\nMiniMax-M3 → M2.7`, with the circuit breaker handling 429 quota exhaustion\nso traffic falls through to M3 (13B tok/mo) automatically. Claude tiers\nstay reserved for `trading_emergency/decision` (Opus) and\n`coding_hard/research/analysis` (Fable). `system_admin` + `creative` were\nmoved off Sonnet onto the GLM chain too — benchmarks (Aug 31) showed\nGLM 5.3/5.2/M3 all pass sysadmin/creative/routine probes at 1.6-3s.\nQuota policy is a user preference, not a fixed rule — ask which direction\nthey want before assuming scarcity-last ordering.

Heuristic: **rank plans by requests-per-day, not by model quality.** The
most-limited prompt-count plan goes last in every chain; the biggest
token-budget plan leads the highest-volume domains.

### 9router-native `auto` is NOT semantic routing

A profile with `provider: 9router` + `default: auto` looks wired but sends
**every request to one fixed model** (observed: everything → Haiku, including
emergencies). 9router's `auto` has no classifier, no tiers, no quota logic.
When auditing whether a profile uses YOUR router, the test is `base_url`:
`:20128` (9router direct = bypassed) vs `:8898` (semantic proxy = wired).
Route one emergency-phrased query and read `x-model-used` — if it's the same
model as a trivial chat query, it's the native auto, not your router.

### Panic phrases need keyword rules AND embedding examples

Calm trading queries ("should I exit NVDA at the 20MA") classify fine, but
ejection phrasings ("flatten everything now", "sell everything right now",
"get me out", "kill all my trades") scored 0/4 on the embedding path — they
have no domain vocabulary for the classifier to lock onto. Belt and braces:

1. Add them to the emergency keyword list (instant, exact).
2. Add 3-4 as `trading_emergency` DOMAIN_EXAMPLES (semantic coverage for
   paraphrases).

Verify with fresh phrasings end-to-end — the response cache will happily
serve you stale pre-fix results for the exact strings you tested before.

### Cross-user deployment (multiple Linux accounts, one machine)

Every Linux user with a `~/.hermes` is a separate deployment surface —
enumerate them (`for d in /home/*/; do ls $d.hermes/profiles; done`), not
just your own profiles dir. Wiring their configs needs:

- sudo to edit their `config.yaml` (same parse→mutate→dump edit, backup first)
- the router key appended to THEIR `~/.hermes/.env`, then
  `chown user:user` + `chmod 600` — a root-owned .env is unreadable to them
- localhost binding is fine: proxies on `127.0.0.1` serve all local users
- their gateways cache config too — they (or sudo) must restart; you cannot
  restart another user's gateway from your session

### Port Conflicts

**Fix:** Unique per profile (8898, 8901, 8902).

### Editing profile configs: NEVER use sed/regex

`custom_providers[].models` is a LIST in some profiles and a DICT in others.
A text insert (`sed`, string replace) that assumes a list will corrupt a
dict-shaped config into invalid YAML. Always parse -> mutate -> dump:

```python
import yaml, shutil
shutil.copy2(cfg, cfg + ".bak")
d = yaml.safe_load(open(cfg))
d["model"]["base_url"] = f"http://127.0.0.1:{port}/v1"
d["model"]["default"] = "auto"
for cp in d.get("custom_providers", []):
    if "20128" in str(cp.get("base_url", "")):
        m = cp.get("models")
        if isinstance(m, dict):  cp["models"] = {"auto": {}, **m}
        elif isinstance(m, list): cp["models"] = ["auto"] + m
yaml.safe_dump(d, open(cfg, "w"), sort_keys=False)
```

Then DIFF the parsed keys against the backup to prove nothing was dropped.
Helper: `~/model-research/point_profile.py <config.yaml> <port>`.

**Two distinct config shapes coexist across `~/.hermes/profiles/*/config.yaml`
on the same machine** — `custom_providers:` (list-of-dicts) AND the flat
`providers.<name>:` / `<name>:` key under `model:`. Real instance (Aug 30,
2026): `jimmy-docked` used the flat shape — `d.get("custom_providers") or []`
silently skipped it, leaving Haiku in the picker after a "successful"
patch. Two patterns to handle both:

```python
# Walk BOTH locations
def all_model_blocks(d):
    blocks = []
    for cp in d.get("custom_providers") or []:
        if isinstance(cp, dict) and isinstance(cp.get("models"), (dict, list)):
            blocks.append(("custom_providers.models", cp["models"]))
    provs = d.get("providers") or {}
    for name, prov in provs.items():
        if isinstance(prov, dict) and isinstance(prov.get("models"), (dict, list)):
            blocks.append((f"providers.{name}.models", prov["models"]))
    # also the flat '<name>:' key (rare)
    for k, v in d.items():
        if k in ("model", "custom_providers", "providers"): continue
        if isinstance(v, dict) and isinstance(v.get("models"), (dict, list)):
            blocks.append((f"{k}.models", v["models"]))
    return blocks
```

Always re-verify after patching by `grep -lE 'claude-haiku' ~/.hermes/profiles/*/config.yaml`
against the actual file path — not just `custom_providers:`.

### Upstream timeout too short

Reasoning models (GLM 5.3) take 35-55s. A 30s `urlopen` timeout causes retry
storms and falls through to the expensive tier, defeating quota burn.
**Use timeout=180.**

### Cache hits drop routing headers

If the response-cache path only sets `x-cache-type`, every warm request
reports `x-model-used: None` and audits look broken. Re-emit `x-domain` and
`x-model-used` (read `model` from the cached body) on cache hits.

### Generic (multi-profile) domain coverage

**This router is general-purpose infrastructure, not a trading tool.** Trading
is one domain among nine (`trading_emergency`, `trading_decision`,
`trading_info`), and the other six — `general`, `general_easy`, `personal`,
`coding_hard`, `coding_routine`, `system_admin`, `research`, `analysis`,
`creative` — carry the majority of real traffic across the youtube /
jimmy-docked / mila-docked / default profiles. Never evaluate a routing
change, a new upstream, or a candidate replacement engine on trading
behavior alone: a change that improves trading routing while degrading
`personal` or `coding_routine` is a net loss, because those domains see
more requests.

Practical consequences when assessing ANY routing change:

- **Weight the evaluation by traffic, not by stakes.** Trading has the
  highest per-request stakes and the lowest request volume. A benchmark
  that only measures agentic-coding accuracy (e.g. Terminal-Bench) tells
  you nothing about how the change treats "what should I make for dinner"
  or "summarize this transcript".
- **Any engine with only two tiers (efficient ↔ capable) is a poor fit.**
  Generic traffic needs the full 7-tier fan-out plus the tier-aware
  fallback lanes (a demoted/cheap request must never escalate into a
  Claude tier). Two-tier routers force you to run N parallel instances
  and re-implement the containment yourself.
- **Sweep all nine domains on all three ports** before declaring any
  routing change done — `scripts/verify-router-tiers.sh` exists for this.
  Trading-only verification is how the `personal`-domain leak survived
  until Sep 1.

The original guidance still applies: audit with a mixed workload (chat,
routine code, hard code, sysadmin, research, creative, media, household
how-to) and check cheap-vs-strong tier placement. Common gap:
architecture/schema/query-tuning questions fall through to `general` ->
cheap model. Add explicit examples to `coding_hard` and `research` in the
classifier's DOMAIN_EXAMPLES.

### Cache Doesn't Warm

**Normal:** 0% → 70%+ over week 1. Monitor `/metrics`.

### Circuit Breaker Stuck

**Fix:** Reset timeout must fire.

```python
if self.open and time.time() - self.last_failure > RESET_TIMEOUT:
    self.open = False
```

## Checklist

- [ ] Classifier (embeddings, >0.7)
- [ ] Tier mapping
- [ ] Multi-layer cache
- [ ] Circuit breaker (60s reset)
- [ ] Fallback chains
- [ ] Rate limiting (150 req/s)
- [ ] /metrics endpoint
- [ ] Response handling (full body, Content-Length)
- [ ] Error handling (BrokenPipe)
- [ ] Deployed (systemd+background)
- [ ] Configs (base_url+custom)
- [ ] Routing tested (3/3)
- [ ] Stress tested (70%+)
- [ ] Cache monitored
- [ ] Scored (93–95/100)

## Production Ready

Declare **production-ready** when:
- Routing 80%+
- Reliability 100% (stress)
- Availability 100% (models)
- Cache 66%+ (growing)
- Failures 0
- Efficiency 93–95/100
- Monitoring live

Stop at 93–95/100. ROI drops. Focus monitoring, not tuning.

## Semantic-Rhyme Absorption Leak (added Sep 1, 2026)

A class of routing bug where a prompt semantically rhymes with an
adjacent domain at a confidence score that CLEARS that domain's
floor/cap, even though the prompt genuinely belongs elsewhere.

**Concrete instance — "Personal Domain":** `trading_decision`'s example
bank is full of `"how should i..."`, `"what's the best..."`, `"should
i..."` phrasings. So `"how do i tie a tie properly"` (a genuine
household how-to) embeds closest to `trading_decision @ 0.233` and
clears `TRADING_FLOOR=0.18` → Opus. Verified live Sep 1, 2026 across
all 3 proxies (8898/8901/8902) before and after the fix.

**The general symptom is hard to spot because:** the routing looks
"correct" by the classifier's own metric. The classifier produced a
confident match. The tier mapping honored the match. Every layer
above the data agreed. But the *data itself* (the example bank)
absorbed a prompt that didn't belong there in the first place.

**The three fixes, in order of preference:**

1. **Add a dedicated domain** (best when the absorbed prompts form a
   coherent cluster you can name — cooking, parenting, trips,
   household how-to, scheduling). Add 15-25 example sentences and
   map it to the cheap tier. This is what we did for `personal`.

2. **Raise the absorber's floor / lower the absorber's cap** (when the
   absorbed prompts are scattered and not a coherent cluster). Cost:
   you will also demote some *legitimate* absorber-domain prompts
   at the same score band. Profile-incompatible with floor-of-trading
   domains where the user has explicitly accepted asymmetric risk.

3. **Add a keyword counter-route** (last resort). When "X" appears in
   the prompt, force-route to a different domain regardless of
   classifier score. Brittle and a sign the semantic bank needs more
   work, but it works.

**Don't:** lower the global `TRADING_FLOOR` or
`PER_DOMAIN_CAP` to fix this. You'll demote real trading prompts or
real research/creative queries that happen to land at the same
score. Solve the leak at its source — the example bank.

**How to detect (before the bill arrives):** sweep a non-specialty
sample through every proxy and inspect `x-domain` for clusters that
shouldn't be there. Any prompt where the `x-model-used` is more
expensive than `x-domain` content suggests IS the leak. Cross-port
comparison (same prompt on 8898/8901/8902) catches per-proxy drift
in the example bank.

After applying this fix:

- how-to/planning prompts → `personal` → tier_quota_burn → GLM (or M3
  fallback when GLM is 429ing)
- genuine trading prompts → `trading_decision/emergency` → Opus
  (unaffected)
- cooking "what should i make for dinner" → was `general_easy`, now
  `personal` (similar cost; slightly more targeted intent)

When the user says "set up my other profile for general-purpose
questions" or "mila/jimmy is for non-trading use", this is the
specific config that makes that work — without it, non-trading
profiles will silently bill Claude Max on how-to questions.

## Cross-Profile Traffic Shape (verified Sep 1, 2026)

Hermes profiles are NOT 1:1 with proxy service names. Live mapping
(verified by reading `~/.hermes/profiles/*/config.yaml`):

| Profile | Proxy service | Port |
|---|---|---|
| `youtube` | `smart-router-proxy.py` | 8898 |
| `jimmy-docked` | `jimmy-router-proxy.py` | 8901 |
| `mila-docked` | `mila-router-proxy.py` | 8902 |
| `default` (trading) | `mila-router-proxy.py` | **8902** |

## Alibaba Token Plan integration (Sep 7, 2026 — grill-session decisions)

User bought Alibaba Cloud AI Token Plan Standard ($18/mo, ~40,000 credits/mo).
9router provider `alitp-intl` (apikey, connection name "Token plan") exposes 6
slugs, all live-verified:

| Slug | Verified | Notes |
|---|---|---|
| `alitp-intl/qwen3.6-flash` | 100% pass bench, 1.7-4.1s, tool calls ✓ | **STABLE WORKHORSE** — leads the Alibaba segment |
| `alitp-intl/qwen3.8-max-preview` | 100% bench BUT **hangs (HTTP 000 >90s) on some prose prompts** — 4/4 fails at 03:15 PDT after passing at 02:45 | 'preview' endpoint = unstable; kept as 2nd hop only. Promo: 2X usage/credit |
| `alitp-intl/qwen3.7-plus` | **vision ✓ ("Red" on red PNG)** | pinned as the local-cluster vision route |
| `alitp-intl/qwen3.7-max` | 200, heavy reasoning | not in chains |
| `alitp-intl/glm-5.2` | 200, 8s | GLM via Alibaba credits — SEPARATE quota from capped glm-cn |
| `alitp-intl/deepseek-v4-pro` | in catalog | 2X-credit promo model |
| qwen3.8-max-preview context | **723,291-token prompt accepted, both markers recalled** | passes the 1M-only tier policy |

**Cheap-lane chain order (user decision B — GLM keeps burn-first):**
`glm-cn/glm-5.3 → glm-5.3-flash → glm-5.2 → alitp/qwen3.6-flash →
alitp/qwen3.8-max-preview → alitp/glm-5.2 → minimax/M3`
in all 3 cheap tiers of `router_tiers.py` + 8 nine-router combos
(Hermes/Jimmy/Mila/LocalOverflow/Judge/tier-fast/tier-standard/tier-reasoning).
While glm-cn is 429-capped the circuit breaker skips straight to the Alibaba
segment — the cheap lane has a working non-MiniMax option for the first time.

**Judge (user decision A — local primary):** local `qwen38-27b-abliterated`
(sglang :11434, free) stays primary; `alitp-intl/qwen3.6-flash` is the
fallback ahead of GLM/M3. Env on all 3 units:
`JUDGE_MODELS=qwen38-27b-abliterated,alitp-intl/qwen3.6-flash,glm-cn/glm-5.3-flash,alitp-intl/deepseek-v4-pro,minimax/MiniMax-M3`
plus `ROUTER_JUDGE_LOCAL_URL`/`ROUTER_JUDGE_LOCAL_MODEL`.

**Sep 9 2026: `glm-cn/glm-5.2` removed from the judge chain** — it is a
200k-context model and the standing policy is 1M-only for anything
auto-routed. Replaced by `alitp-intl/deepseek-v4-pro`. NOTE this value lives
in BOTH the code default (`router_escalation.py`) and
`Environment=JUDGE_MODELS=` in all three systemd units; the unit WINS. Editing
only the module is a silent no-op — see
`references/verifying-fail-open-layers.md` § "Config precedence".

**Session-sticky domains (shipped same night):** the classification cache was
keyed on the growing conversation so every turn re-classified → domain drift
within one session (personal→research→personal). Fix: cache the domain under
`'sess:' + session_key(messages)` (stable prefix hash); first classification
wins for the session. **Safety override:** a confident `trading_*`
classification (score ≥ TRADING_FLOOR AND passes the vocabulary guard) still
wins per-turn and routes at the trading tier WITHOUT overwriting the session
domain — so one absorbed false-positive self-heals next turn, while a real
margin-call conversation re-triggers the override every turn.

**Trading-absorption guard (`trading_signal_ok` in router_escalation.py):**
embedding score alone is NOT sufficient for the trading override. Live leak:
"That is too brief and missing documents. I need every form field and the fees
covered thoroughly." → trading_emergency ABOVE floor → one Opus call burned.
Guard rules: hard keyword (margin call, liquidat*, flatten everything...) OR
≥1 STRONG signal (portfolio, broker, trades, trading, stop-loss, margin,
ticker names...) OR ≥2 distinct WEAK signals (position, order, exit, long,
sell, hold...) OR 1 weak + ALL-CAPS ticker. 12/13 unit cases; the accepted
fail is fail-SAFE direction (over-flags, costs one classifier lookup).

**Verification artifacts:** `verify_alibaba_integration.py` (sibling ports +
drift test + trading override test), `final_sweep.py` (9 domains × 3 ports),
`test_escalation_advisor.py` (75+ unit tests). All green Sep 7 03:30 PDT.

**PITFALL — 'preview' cloud endpoints can hang mid-session.**
qwen3.8-max-preview passed every probe at 02:45 then hung 4/4 at 03:15
(HTTP 000, no response, no error) while sibling slugs on the SAME provider
answered in 8-27s. A hanging model is worse than a 429: the proxy pays
retries × 180s timeout before the breaker trips. Never put a preview/unstable
slug first in a chain; always have a proven sibling immediately after it.


Default and mila-docked **share port 8902**. Any audit that infers
"profile X uses proxy Y, profile Y uses proxy Z" by reading
systemd unit names will be wrong. Two practical consequences:

1. **Cache poisoning crosses profiles.** A cached response served to
   a mila chat can later be served to a default trading chat if the
   conversation key collides. The 1542-char shared system-prompt
   prefix collision (see cache-key pitfall above) is WORSE on a
   shared port because both profiles' users see the symptom.
2. **Cost attribution is ambiguous.** When port 8902 lands on Opus
   via the trading-decision path, it could be the trading profile
   (legit) or the mila profile (leak — what the personal-domain
   fix targets). Read the journal per-port AND check the
   conversation context before declaring "Opus was burned for
   legitimate work."

**All profiles share the same classifier and tier table** —
`router_tiers.py` for tiers, `smart-prompt-router.py` for the
example bank. The `personal` domain we added for mila is therefore
also in effect on the trading profile (port 8902). Verified: trading
prompts still route to Opus there. The shared example bank is fine
for routing decisions, just not for cost attribution.
