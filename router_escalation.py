#!/usr/bin/env python3
"""Shared observability + escalation + advisor layer for ALL semantic-router
proxies (smart/8898, jimmy/8901, mila/8902).

Created Sep 7, 2026 after evaluating NVIDIA-NeMo/Switchyard. We did NOT adopt
Switchyard (its algorithms are all N=2 efficient<->capable, it has no intent
classifier, and no quota-shape awareness -- see
skills/.../semantic-routing-auto-model/references/switchyard-evaluation-2026-09-07.md).
We ported four of its ideas onto our existing 7-tier / 9-domain router:

  1. StatsTracker      -> per-domain/per-model spend accounting at /v1/stats
  2. EscalationTracker -> judge the COMPLETED turn, latch a session upward
                          after N consecutive bad verdicts
  3. AdvisorGate       -> a stronger model reviews terminal turns and can
                          force a REDO before the client ever sees them
  4. capability flags  -> honest input_modalities/tool_calling at /v1/models

THIS ROUTER IS GENERAL-PURPOSE. Trading is 1 of 9 domains. Every knob below is
tuned for the whole traffic mix (personal, general_easy, creative,
coding_routine, system_admin, research, analysis + the trading domains), not
for trading alone.

Design rule inherited from the proxies: FAIL OPEN. A judge/advisor that times
out, errors, or returns garbage must never break the user's request and must
never manufacture an escalation. Every failure path here degrades to "serve
what the cheap model already produced".
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# The advisor / escalation strong model. User directive Sep 7, 2026:
# "the advisor will be Claude fable 5.1". Verified present in the live
# 9router catalog on Sep 7 2026 (`cc/claude-fable-5-1`); cc/claude-fable-5
# is the OLDER 5.0 build and is what tier_primary_complex still uses.
ADVISOR_MODEL = os.environ.get("ADVISOR_MODEL", "cc/claude-fable-5-1")

# The judge for escalation verdicts. Deliberately CHEAP -- it runs on every
# unlatched turn, so a frontier judge would cost more than the escalation
# saves.
#
# This is a CHAIN, not a single model (Sep 7 2026): the GLM plan hit its
# weekly/monthly cap (429 code 1310, resets 2026-09-11) during commissioning,
# which would have made a hard-coded GLM judge fail-open on every single turn
# -- escalation would silently never fire and nothing would say why. The judge
# walks this list and uses the first model that answers. MiniMax-M3 (13B
# tok/mo) is the durable tail.
# A local model in front of the cloud chain is the cheapest, most reliable
# judge. Set ROUTER_JUDGE_LOCAL_URL=http://127.0.0.1:11434/v1 and
# ROUTER_JUDGE_LOCAL_MODEL=qwen38-27b-abliterated to enable.
# The local model is tried FIRST; only if it returns nothing parseable do
# we walk the cloud chain. This eliminates the "GLM quota-capped -> judge
# fails open every turn" failure mode for free.
_LOCAL_JUDGE_URL = os.environ.get("ROUTER_JUDGE_LOCAL_URL", "").strip()
_LOCAL_JUDGE_MODEL = os.environ.get("ROUTER_JUDGE_LOCAL_MODEL", "").strip()
# Sep 7 user decision (grill session): local qwen38-27b stays PRIMARY judge
# (free, unlimited), alitp-intl/qwen3.6-flash is the fast fallback when the
# local box is busy/down (~2s verdicts, verified tool-calling + JSON format).
# GLM entries remain for after the Sep 11 quota reset; MiniMax-M3 is the
# durable tail.
JUDGE_MODELS = (
    ([_LOCAL_JUDGE_MODEL] if _LOCAL_JUDGE_MODEL and _LOCAL_JUDGE_URL else [])
    + [m.strip() for m in os.environ.get(
        "JUDGE_MODELS",
        # 1M-CONTEXT ONLY. glm-cn/glm-5.2 was removed Sep 9 2026: it is a
        # 200k-context model and the standing rule is that no auto-routed
        # model may be under 1M. deepseek-v4-pro replaces it as the third
        # stop (1M ctx, text-only, cheap, on the Alibaba plan).
        "alitp-intl/qwen3.6-flash,glm-cn/glm-5.3-flash,alitp-intl/deepseek-v4-pro,minimax/MiniMax-M3"
    ).split(",") if m.strip()]
)
JUDGE_MODEL = JUDGE_MODELS[0] if JUDGE_MODELS else ""
JUDGE_USE_LOCAL = bool(_LOCAL_JUDGE_URL and _LOCAL_JUDGE_MODEL)

# ── Escalation ────────────────────────────────────────────────────────────
# Switchyard's benchmarked defaults (confirmations=2, recent_turn_window=28,
# window_message_chars=500) ported as-is; they were tuned on multi-turn agent
# traffic which is what our profiles actually send.
ESCALATION_ENABLED = os.environ.get("ESCALATION_ENABLED", "1") == "1"
ESCALATION_CONFIRMATIONS = int(os.environ.get("ESCALATION_CONFIRMATIONS", "2"))
ESCALATION_TURN_WINDOW = int(os.environ.get("ESCALATION_TURN_WINDOW", "28"))
ESCALATION_MSG_CHARS = int(os.environ.get("ESCALATION_MSG_CHARS", "500"))

# Domains eligible for escalation. Frontier domains are excluded: they ALREADY
# start on Claude, so there is nothing to escalate to. Demoted requests are
# excluded separately and unconditionally (see should_escalate).
ESCALATION_DOMAINS = {
    "general", "general_easy", "personal", "creative",
    "coding_routine", "system_admin", "trading_info",
}

# Cross-lane budget. Escalation is the ONE sanctioned way a cheap-lane session
# may reach a Claude model, so it is hard-capped. The containment invariant
# (a demoted/cheap request never *accidentally* escalates via a breaker
# cascade) is untouched: this path is deliberate, judged, per-session, and
# budgeted, and it is logged as ESCALATION LATCH rather than FALLBACK.
ESCALATION_BUDGET_PER_HOUR = int(os.environ.get("ESCALATION_BUDGET_PER_HOUR", "12"))
ESCALATION_SESSION_TTL_S = int(os.environ.get("ESCALATION_SESSION_TTL_S", "3600"))
ESCALATION_MAX_SESSIONS = 2000

# ── Advisor gate ──────────────────────────────────────────────────────────
# Narrow by design. On `general`/`personal` an advisor is pure added latency
# and cost (Switchyard measured no lift on strong executors), so the gate is
# opt-in per domain and defaults to the two places a cheap-model hallucination
# is genuinely expensive.
ADVISOR_ENABLED = os.environ.get("ADVISOR_ENABLED", "1") == "1"
ADVISOR_DOMAINS = set(
    d for d in os.environ.get(
        "ADVISOR_DOMAINS", "trading_decision,trading_emergency,coding_hard"
    ).split(",") if d.strip()
)
ADVISOR_MAX_REVIEWS = int(os.environ.get("ADVISOR_MAX_REVIEWS", "3"))
ADVISOR_MIN_TOOL_RESULTS = int(os.environ.get("ADVISOR_MIN_TOOL_RESULTS", "3"))
ADVISOR_STALL_TURNS = int(os.environ.get("ADVISOR_STALL_TURNS", "30"))
ADVISOR_MAX_TOKENS = int(os.environ.get("ADVISOR_MAX_TOKENS", "2048"))
ADVISOR_TRANSCRIPT_MAX_CHARS = int(os.environ.get("ADVISOR_TRANSCRIPT_MAX_CHARS", "200000"))
ADVISOR_FAIL_OPEN = os.environ.get("ADVISOR_FAIL_OPEN", "1") == "1"

# Only advise a turn the cheap lane produced. If the executor was already
# Fable/Opus, a Fable advisor adds nothing (Switchyard: no lift on strong
# executors) and would double the bill on the session-capped Claude plan.
ADVISOR_SKIP_IF_EXECUTOR_PREFIX = ("cc/",)

STATS_RING_SIZE = int(os.environ.get("STATS_RING_SIZE", "5000"))


# ═══════════════════════════════════════════════════════════════════════════
# CAPABILITY FLAGS  (Switchyard steal #3)
# ═══════════════════════════════════════════════════════════════════════════
# Switchyard PR #567 measured this at the wire: Codex reads input_modalities
# from the model card and, when it reads text-only, replaces an attached image
# with the literal string "image content omitted because you do not support
# image input" BEFORE sending. 248KB outbound vs 739KB with ["text","image"].
# That is almost certainly the Sep 6 symptom where MiniMax M3 called a red PNG
# "brown" -- the bytes may never have left the client.
#
# FAIL CLOSED: declare vision only when EVERY model a route can select accepts
# images. A tier whose chain can fall through to a text-only model must not
# advertise vision, or we hand an image to a backend that cannot read it.
MODEL_CAPABILITIES = {
    "cc/claude-opus-5":              {"vision": True,  "tool_calling": True, "reasoning": True},
    "cc/claude-fable-5":             {"vision": True,  "tool_calling": True, "reasoning": True},
    "cc/claude-fable-5-1":           {"vision": True,  "tool_calling": True, "reasoning": True},
    "cc/claude-sonnet-5":            {"vision": True,  "tool_calling": True, "reasoning": True},
    "cc/claude-haiku-4-5-20251001":  {"vision": True,  "tool_calling": True, "reasoning": False},
    # GLM + MiniMax: text-only through our 9router path until proven
    # otherwise with a real image round-trip. Unverified == False.
    "glm-cn/glm-5.3":                {"vision": False, "tool_calling": True, "reasoning": True},
    "glm-cn/glm-5.3-flash":          {"vision": False, "tool_calling": True, "reasoning": True},
    "glm-cn/glm-5.2":                {"vision": False, "tool_calling": True, "reasoning": False},
    "minimax/MiniMax-M3":            {"vision": False, "tool_calling": True, "reasoning": True},
    # Alibaba Token Plan (alitp-intl), live-verified Sep 7 2026:
    #  - qwen3.8-max-preview: 723,291-token prompt accepted + markers recalled
    #    (real 1M context), tool_calls verified, no vision tested -> False.
    #  - qwen3.7-plus: VISION VERIFIED (correctly answered "Red" on a red PNG
    #    through 9router). Pinned as the local-cluster vision route.
    #  - qwen3.6-flash: tool_calls verified, fast (~1.7-2.6s).
    #  - qwen3.7-max: heavier reasoning, no tool test yet -> conservative.
    "alitp-intl/qwen3.8-max-preview": {"vision": False, "tool_calling": True, "reasoning": True},
    "alitp-intl/qwen3.7-plus":        {"vision": True,  "tool_calling": False, "reasoning": True},
    "alitp-intl/qwen3.7-max":         {"vision": False, "tool_calling": False, "reasoning": True},
    "alitp-intl/qwen3.6-flash":       {"vision": False, "tool_calling": True, "reasoning": True},
    "alitp-intl/glm-5.2":             {"vision": False, "tool_calling": True, "reasoning": False},
    "alitp-intl/deepseek-v4-pro":     {"vision": False, "tool_calling": False, "reasoning": True},
}
_DEFAULT_CAP = {"vision": False, "tool_calling": True, "reasoning": False}


def tier_capabilities(models):
    """Fold per-model capabilities into one honest claim for a tier.

    AND across the chain: the route can select ANY model in `models`, so a
    capability is only advertised when every candidate has it.
    """
    caps = {"vision": True, "tool_calling": True, "reasoning": True}
    if not models:
        return {"vision": False, "tool_calling": False, "reasoning": False}
    for m in models:
        mc = MODEL_CAPABILITIES.get(m, _DEFAULT_CAP)
        for k in caps:
            caps[k] = caps[k] and bool(mc.get(k, False))
    return caps


def build_models_payload(tier_system, domain_to_tier, extra_ids=("auto",)):
    """OpenAI-shaped GET /v1/models body with honest capability flags."""
    now = int(time.time())
    data = []
    for tier_name, cfg in (tier_system or {}).items():
        caps = tier_capabilities(cfg.get("models") or [])
        data.append({
            "id": tier_name,
            "object": "model",
            "created": now,
            "owned_by": "semantic-router",
            "input_modalities": ["text", "image"] if caps["vision"] else ["text"],
            "vision": caps["vision"],
            "tool_calling": caps["tool_calling"],
            "reasoning": caps["reasoning"],
            "context_window": 1000000,
            "_models": cfg.get("models") or [],
            "_plan": cfg.get("plan"),
        })
    # The `auto` pseudo-model is what clients actually send. It can resolve to
    # ANY tier, so it must fold capabilities across every tier -- fail closed.
    all_models = [m for cfg in (tier_system or {}).values() for m in (cfg.get("models") or [])]
    auto_caps = tier_capabilities(all_models)
    for eid in extra_ids:
        data.insert(0, {
            "id": eid,
            "object": "model",
            "created": now,
            "owned_by": "semantic-router",
            "input_modalities": ["text", "image"] if auto_caps["vision"] else ["text"],
            "vision": auto_caps["vision"],
            "tool_calling": auto_caps["tool_calling"],
            "reasoning": auto_caps["reasoning"],
            "context_window": 1000000,
        })
    return {"object": "list", "data": data}


# ═══════════════════════════════════════════════════════════════════════════
# SESSION IDENTITY
# ═══════════════════════════════════════════════════════════════════════════
# Switchyard requires clients to send x-switchyard-session-id. Our agent
# clients will never send that, and a header nobody sends means escalation
# silently never latches (their own docs call this out as a failure mode).
#
# Instead derive a stable identity from the conversation PREFIX: the system
# prompt plus the FIRST user message. That is invariant as the conversation
# grows, unlike the full-conversation cache key which changes every turn.

_BOILERPLATE_RE = [re.compile(p, re.I) for p in (
    r"you just executed tool calls",
    r"returned an empty response",
    r"^\s*\[?important:\s*you are running as a scheduled cron",
    r"please (respond|continue)\.?\s*$",
    r"^\s*(continue|resume|go on|proceed)\.?\s*$",
)]


def _text_of(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, sort_keys=True, default=str)
    except Exception:
        return str(content)


def session_key(messages, request_header_sid=None):
    """Stable per-conversation identity, invariant across turns.

    Honors an explicit client session header when one IS present (some
    harnesses do send it), else derives from the conversation prefix.
    """
    if request_header_sid:
        return "sid:" + hashlib.sha256(str(request_header_sid).encode()).hexdigest()[:32]
    sys_txt = ""
    first_user = ""
    for m in messages or []:
        role = m.get("role")
        if role == "system" and not sys_txt:
            sys_txt = _text_of(m.get("content"))
        elif role == "user":
            t = _text_of(m.get("content"))
            if t.strip() and not any(p.search(t) for p in _BOILERPLATE_RE):
                first_user = t
                break
    return "cnv:" + hashlib.sha256((sys_txt + "\x00" + first_user).encode()).hexdigest()[:32]


def count_assistant_turns(messages):
    return sum(1 for m in (messages or []) if m.get("role") == "assistant")


def count_tool_results(messages):
    n = 0
    for m in messages or []:
        if m.get("role") == "tool":
            n += 1
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            n += len(m.get("tool_calls") or [])
    return n


def render_transcript(messages, max_chars=ADVISOR_TRANSCRIPT_MAX_CHARS):
    """Middle-out truncation: keep the task statement and the recent work."""
    parts = []
    for m in messages or []:
        role = m.get("role", "?")
        txt = _text_of(m.get("content"))
        if m.get("tool_calls"):
            try:
                names = ", ".join(
                    (tc.get("function") or {}).get("name", "?")
                    for tc in m["tool_calls"])
                txt = (txt + f"\n[tool_calls: {names}]").strip()
            except Exception:
                pass
        parts.append(f"<{role}>\n{txt}")
    full = "\n\n".join(parts)
    if len(full) <= max_chars:
        return full
    head = max_chars // 2
    tail = max_chars - head
    return (full[:head] + "\n\n...<middle of the conversation truncated>...\n\n"
            + full[-tail:])


# ═══════════════════════════════════════════════════════════════════════════
# STATS  (Switchyard steal #1)
# ═══════════════════════════════════════════════════════════════════════════

class StatsTracker:
    """Per-domain / per-model spend accounting.

    We already had /metrics (cache + breaker state) but nothing that could
    answer "is GLM-first actually burning GLM" or "which domain costs most".
    Judge and advisor calls land in their OWN buckets so routing overhead
    stays visible instead of hiding inside the served model's numbers.
    """

    def __init__(self, port=None, ring_size=STATS_RING_SIZE):
        self.port = port
        self.lock = threading.Lock()
        self.ring = deque(maxlen=ring_size)
        self.by_domain = {}
        self.by_model = {}
        self.by_tier = {}
        self.overhead = {"judge": {}, "advisor": {}}
        self.counters = {
            "requests": 0,
            "escalation_judged": 0,
            "escalation_latched": 0,
            "escalation_budget_denied": 0,
            "judge_unreachable": 0,    # local+cloud chain all failed -> fail-open
            "judge_local_used": 0,     # last judge was the local model
            "judge_cloud_used": 0,     # last judge was a cloud model
            "advisor_reviews": 0,
            "advisor_redo": 0,
            "advisor_approve": 0,
            "advisor_failed": 0,
            "invariant_violations": 0,
        }
        self.started = time.time()

    @staticmethod
    def _usage(raw):
        try:
            d = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
            u = (d or {}).get("usage") or {}
            return (int(u.get("prompt_tokens") or 0),
                    int(u.get("completion_tokens") or 0))
        except Exception:
            return (0, 0)

    def _bump(self, bucket, key, tin, tout, latency_ms, ok):
        e = bucket.setdefault(key, {
            "calls": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0,
            "latency_ms_total": 0.0})
        e["calls"] += 1
        if not ok:
            e["errors"] += 1
        e["tokens_in"] += tin
        e["tokens_out"] += tout
        e["latency_ms_total"] += float(latency_ms or 0)

    def record(self, *, domain, tier, model, latency_ms, ok=True,
               raw=None, demoted=False, lane=None, cache_type="miss",
               escalated=False, advised=None, fallback_reason=None):
        tin, tout = self._usage(raw) if raw else (0, 0)
        with self.lock:
            self.counters["requests"] += 1
            self._bump(self.by_domain, domain or "unknown", tin, tout, latency_ms, ok)
            self._bump(self.by_model, model or "none", tin, tout, latency_ms, ok)
            self._bump(self.by_tier, tier or "unknown", tin, tout, latency_ms, ok)
            self.ring.append({
                "ts": round(time.time(), 3), "domain": domain, "tier": tier,
                "model": model, "demoted": demoted, "lane": lane,
                "latency_ms": round(float(latency_ms or 0), 1),
                "tokens_in": tin, "tokens_out": tout, "ok": ok,
                "cache_type": cache_type, "escalated": escalated,
                "advised": advised, "fallback_reason": fallback_reason,
            })

    def record_overhead(self, kind, model, latency_ms, ok=True, raw=None):
        tin, tout = self._usage(raw) if raw else (0, 0)
        with self.lock:
            self._bump(self.overhead.setdefault(kind, {}), model or "none",
                       tin, tout, latency_ms, ok)

    def bump(self, counter, n=1):
        with self.lock:
            self.counters[counter] = self.counters.get(counter, 0) + n

    @staticmethod
    def _finalize(bucket):
        out = {}
        for k, v in bucket.items():
            c = max(v["calls"], 1)
            out[k] = {
                "calls": v["calls"], "errors": v["errors"],
                "tokens_in": v["tokens_in"], "tokens_out": v["tokens_out"],
                "tokens_total": v["tokens_in"] + v["tokens_out"],
                "avg_latency_ms": round(v["latency_ms_total"] / c, 1),
            }
        return out

    def snapshot(self, recent=50):
        with self.lock:
            return {
                "port": self.port,
                "uptime_s": round(time.time() - self.started, 1),
                "counters": dict(self.counters),
                "by_domain": self._finalize(self.by_domain),
                "by_model": self._finalize(self.by_model),
                "by_tier": self._finalize(self.by_tier),
                "routing_overhead": {k: self._finalize(v)
                                     for k, v in self.overhead.items()},
                "recent": list(self.ring)[-recent:],
            }

    def prometheus_text(self):
        s = self.snapshot(recent=0)
        L = []
        p = self.port or 0
        L.append("# HELP router_requests_total Requests served")
        L.append("# TYPE router_requests_total counter")
        L.append(f'router_requests_total{{port="{p}"}} {s["counters"]["requests"]}')
        for name, val in s["counters"].items():
            if name == "requests":
                continue
            L.append(f'router_counter{{port="{p}",name="{name}"}} {val}')
        for dim, bucket in (("domain", s["by_domain"]), ("model", s["by_model"]),
                            ("tier", s["by_tier"])):
            for k, v in bucket.items():
                lbl = f'port="{p}",{dim}="{k}"'
                L.append(f'router_calls_total{{{lbl}}} {v["calls"]}')
                L.append(f'router_errors_total{{{lbl}}} {v["errors"]}')
                L.append(f'router_tokens_total{{{lbl}}} {v["tokens_total"]}')
                L.append(f'router_avg_latency_ms{{{lbl}}} {v["avg_latency_ms"]}')
        for kind, bucket in s["routing_overhead"].items():
            for k, v in bucket.items():
                lbl = f'port="{p}",kind="{kind}",model="{k}"'
                L.append(f'router_overhead_calls_total{{{lbl}}} {v["calls"]}')
                L.append(f'router_overhead_tokens_total{{{lbl}}} {v["tokens_total"]}')
        return "\n".join(L) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# ESCALATION  (Switchyard steal #2)
# ═══════════════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM = (
    "You are a routing judge. You will see a conversation between a user and "
    "an assistant, ending with the assistant's most recent completed turn. "
    "Decide whether the assistant is STUCK: repeating itself, looping on the "
    "same failed approach, contradicting itself, ignoring the user's actual "
    "question, producing empty or obviously broken output, or plainly out of "
    "its depth.\n"
    "Judge the work that was ACTUALLY done, not how hard the task looks. "
    "A correct, useful answer is never stuck, however short.\n"
    'Reply with ONLY compact JSON: {"verdict":"escalate"|"decline",'
    '"reason":"<10 words>"}'
)

_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(escalate|decline)"', re.I)


def parse_verdict(raw):
    """Extract a verdict. Returns 'escalate' | 'decline' | None (unparseable).

    None is NOT a decline -- the caller must fail open and HOLD the streak,
    never reset it and never latch on it.
    """
    try:
        d = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
        content = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
    except Exception:
        content = raw if isinstance(raw, str) else None
    if not content:
        return None
    if isinstance(content, bytes):
        content = content.decode("utf-8", "replace")
    m = _VERDICT_RE.search(content)
    if m:
        return m.group(1).lower()
    low = content.strip().lower()
    if low.startswith("escalate"):
        return "escalate"
    if low.startswith("decline"):
        return "decline"
    return None


class EscalationTracker:
    """Per-session consecutive-escalate streak with a global hourly budget."""

    def __init__(self, stats=None):
        self.lock = threading.Lock()
        self.sessions = OrderedDict()   # skey -> {streak, latched, ts}
        self.spend = deque()            # timestamps of granted latches
        self.stats = stats

    def _gc(self, now):
        while self.sessions:
            k, v = next(iter(self.sessions.items()))
            if now - v["ts"] > ESCALATION_SESSION_TTL_S:
                self.sessions.popitem(last=False)
            else:
                break
        while len(self.sessions) > ESCALATION_MAX_SESSIONS:
            self.sessions.popitem(last=False)
        while self.spend and now - self.spend[0] > 3600:
            self.spend.popleft()

    def is_latched(self, skey):
        with self.lock:
            v = self.sessions.get(skey)
            if not v:
                return False
            if time.time() - v["ts"] > ESCALATION_SESSION_TTL_S:
                self.sessions.pop(skey, None)
                return False
            return bool(v["latched"])

    def budget_available(self):
        with self.lock:
            self._gc(time.time())
            return len(self.spend) < ESCALATION_BUDGET_PER_HOUR

    def record_verdict(self, skey, verdict):
        """Apply a verdict. Returns (streak, latched_now: bool).

        verdict None (judge failed/unparseable) HOLDS the streak -- it never
        resets it and never latches. A judge failure must not create a
        strong-tier latch (Switchyard's rule, and it protects the Claude plan).
        """
        now = time.time()
        with self.lock:
            self._gc(now)
            v = self.sessions.get(skey) or {"streak": 0, "latched": False, "ts": now}
            v["ts"] = now
            latched_now = False
            if verdict == "escalate":
                v["streak"] += 1
                if (v["streak"] >= ESCALATION_CONFIRMATIONS and not v["latched"]):
                    if len(self.spend) < ESCALATION_BUDGET_PER_HOUR:
                        v["latched"] = True
                        latched_now = True
                        self.spend.append(now)
                    else:
                        if self.stats:
                            self.stats.bump("escalation_budget_denied")
                        logger.warning(
                            "ESCALATION BUDGET EXHAUSTED (%d/hr) -- holding session "
                            "on cheap lane", ESCALATION_BUDGET_PER_HOUR)
            elif verdict == "decline":
                v["streak"] = 0
            # verdict is None -> hold streak unchanged (fail open)
            self.sessions[skey] = v
            self.sessions.move_to_end(skey)
            return v["streak"], latched_now

    def build_judge_request(self, messages, assistant_reply_text, model=None):
        """Trailing-window transcript + the completed turn, for the judge."""
        window = (messages or [])[-ESCALATION_TURN_WINDOW:]
        lines = []
        for m in window:
            t = _text_of(m.get("content"))
            if len(t) > ESCALATION_MSG_CHARS:
                t = t[:ESCALATION_MSG_CHARS] + "…"
            lines.append(f"<{m.get('role','?')}> {t}")
        reply = assistant_reply_text or ""
        if len(reply) > ESCALATION_MSG_CHARS:
            reply = reply[:ESCALATION_MSG_CHARS] + "…"
        lines.append(f"<assistant:JUST_COMPLETED> {reply}")
        return {
            "model": model or JUDGE_MODELS[0],
            "max_tokens": 1500,   # MIN_MAX_TOKENS: thinking models emit '' below this
            "temperature": 0,
            "stream": False,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
        }


def should_escalate(domain, demoted, tier, lane):
    """Gate BEFORE any judge call is made (the judge itself costs money)."""
    if not ESCALATION_ENABLED:
        return False
    # Demoted == the classifier was guessing (chit-chat that landed on
    # creative/research at low confidence). Never worth a Claude rescue.
    # This preserves the containment invariant exactly as written.
    if demoted:
        return False
    if lane != "cheap":
        return False
    return domain in ESCALATION_DOMAINS


# ═══════════════════════════════════════════════════════════════════════════
# ADVISOR GATE  (Switchyard steal #4)  -- advisor = cc/claude-fable-5-1
# ═══════════════════════════════════════════════════════════════════════════

_ADVISOR_SYSTEM = (
    "You are a senior reviewer gating another model's work. You will see the "
    "task, the actions taken, their results, and the turn the executor is "
    "about to send to the user.\n"
    "Reply with APPROVE or REDO as the FIRST WORD of your reply.\n"
    "APPROVE when the turn is correct, complete, and safe to send.\n"
    "REDO when the executor claims completion it did not achieve, skipped "
    "verification it claimed to do, is stalling or looping, made an unsafe or "
    "irreversible recommendation on thin evidence, or answered a different "
    "question than the one asked.\n"
    "After REDO, give a short concrete plan telling the executor exactly what "
    "to do next. Be strict about unverified completion claims; be lenient "
    "about style."
)

_REDO_PREFIX = (
    "[Reviewer feedback — your previous draft was not sent to the user. "
    "Address this and produce the corrected response:]\n"
)


class AdvisorGate:
    """A stronger model reviews terminal turns before the client sees them.

    Per-session review budget. Fail-open by default: any advisor failure
    releases the executor's turn unchanged rather than erroring the request.
    """

    def __init__(self, stats=None):
        self.lock = threading.Lock()
        self.budget = OrderedDict()   # skey -> {used, failed, ts}
        self.stats = stats

    def _entry(self, skey):
        now = time.time()
        with self.lock:
            while self.budget:
                k, v = next(iter(self.budget.items()))
                if now - v["ts"] > ESCALATION_SESSION_TTL_S:
                    self.budget.popitem(last=False)
                else:
                    break
            while len(self.budget) > ESCALATION_MAX_SESSIONS:
                self.budget.popitem(last=False)
            e = self.budget.get(skey) or {"used": 0, "failed": 0, "ts": now}
            e["ts"] = now
            self.budget[skey] = e
            self.budget.move_to_end(skey)
            return dict(e)

    def _commit(self, skey, used_delta=0, failed_delta=0):
        with self.lock:
            e = self.budget.get(skey) or {"used": 0, "failed": 0, "ts": time.time()}
            e["used"] += used_delta
            e["failed"] += failed_delta
            e["ts"] = time.time()
            self.budget[skey] = e

    def should_review(self, *, domain, executor_model, messages, reply_has_tool_calls):
        """Decide whether this completed turn is worth an advisor consult."""
        if not ADVISOR_ENABLED or not ADVISOR_DOMAINS:
            return False, "disabled"
        if domain not in ADVISOR_DOMAINS:
            return False, "domain_not_gated"
        # No lift reviewing a frontier executor, and it double-bills Claude.
        if executor_model and executor_model.startswith(ADVISOR_SKIP_IF_EXECUTOR_PREFIX):
            return False, "executor_already_frontier"
        e = self._entry(session_key(messages))
        if e["used"] >= ADVISOR_MAX_REVIEWS:
            return False, "budget_spent"
        if e["failed"] >= 3:
            return False, "consult_failures_capped"

        # Trigger 1 (no_tool_call): the natural "I'm done / here's my plan"
        # moment on a function-calling harness.
        if not reply_has_tool_calls:
            if count_tool_results(messages) >= ADVISOR_MIN_TOOL_RESULTS:
                return True, "no_tool_call"
            # Trigger 2 (stall): executor grinding without ever declaring done.
            if (ADVISOR_STALL_TURNS
                    and count_assistant_turns(messages) >= ADVISOR_STALL_TURNS):
                return True, "stall"
            return False, "early_chatty_turn"
        if (ADVISOR_STALL_TURNS
                and count_assistant_turns(messages) >= ADVISOR_STALL_TURNS):
            return True, "stall"
        return False, "has_tool_calls"

    def build_review_request(self, messages, draft_reply_text):
        transcript = render_transcript(messages)
        return {
            "model": ADVISOR_MODEL,
            "max_tokens": ADVISOR_MAX_TOKENS,
            "stream": False,
            "messages": [
                {"role": "system", "content": _ADVISOR_SYSTEM},
                {"role": "user", "content":
                    f"=== TRANSCRIPT ===\n{transcript}\n\n"
                    f"=== TURN AWAITING REVIEW ===\n{draft_reply_text}\n\n"
                    f"APPROVE or REDO?"},
            ],
        }

    @staticmethod
    def parse_review(raw):
        """Returns ('approve'|'redo', plan_text) or (None, None) if unparseable."""
        try:
            d = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
            content = ((d.get("choices") or [{}])[0].get("message") or {}).get("content")
        except Exception:
            content = raw if isinstance(raw, str) else None
        if not content:
            return None, None
        if isinstance(content, bytes):
            content = content.decode("utf-8", "replace")
        stripped = content.strip()
        head = stripped[:32].upper()
        if head.startswith("APPROVE"):
            return "approve", None
        if head.startswith("REDO"):
            return "redo", stripped[4:].lstrip(" :.-\n") or "Revise and verify."
        return None, None

    def note(self, skey, *, used=0, failed=0):
        self._commit(skey, used_delta=used, failed_delta=failed)


def extract_reply_text(raw):
    """Pull assistant text out of a non-streaming completion body."""
    try:
        d = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
        msg = ((d.get("choices") or [{}])[0].get("message") or {})
        c = msg.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(
                b.get("text", "") for b in c if isinstance(b, dict))
        return ""
    except Exception:
        return ""


def reply_has_tool_calls(raw):
    try:
        d = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
        msg = ((d.get("choices") or [{}])[0].get("message") or {})
        return bool(msg.get("tool_calls"))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL-FIRST JUDGE CALL
# ═══════════════════════════════════════════════════════════════════════════

def call_judge_local(payload, *, timeout=120):
    """Talk to the local judge directly. Returns (ok, raw_body).

    Ok=False means transport failure (unreachable / timeout / non-2xx).
    raw_body is bytes on success, the bytes of the error response on failure.

    Tries the configured ROUTER_JUDGE_LOCAL_URL first, then falls through
    a small list of local-bind candidates (127.0.0.1, ::1, host Tailscale
    IPv4, host primary IPv4) before giving up. This is the safety net for
    when sglang/vllm/ollama bind to a specific interface (Tailscale, LAN)
    instead of 0.0.0.0 and the operator forgot to update the URL.
    """
    if not (_LOCAL_JUDGE_URL and _LOCAL_JUDGE_MODEL):
        return False, b""
    import json as _json, socket, urllib.request as _ur
    base = _LOCAL_JUDGE_URL.rstrip("/")
    # Build candidate URLs: configured first, then host IP fallbacks.
    candidates = [base]
    try:
        host = socket.gethostname()
        # Tailscale-style 100.x IPv4 (CGNAT range)
        tailscale_ip = socket.gethostbyname(host)
        candidates += [
            f"http://{tailscale_ip}:11434/v1",
            f"http://[{socket.getaddrinfo(host, None, socket.AF_INET6)[0][4][0]}]:11434/v1",
        ]
    except Exception:
        pass
    candidates += [
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
    ]
    # De-dupe, preserve order, drop anything that doesn't look like a chat URL.
    seen, urls = set(), []
    for u in candidates:
        if u in seen or not u.endswith("/v1"):
            continue
        seen.add(u); urls.append(u)
    body = _json.dumps({**payload, "model": _LOCAL_JUDGE_MODEL}).encode()
    last_err = b""
    for url in urls:
        req = _ur.Request(
            url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with _ur.urlopen(req, timeout=timeout) as resp:
                # Surface which URL actually answered in the failure case so
                # operator logs are actionable.
                if url != base:
                    print(f"[judge] local judge answered on fallback {url} (config={base})",
                          flush=True)
                return True, resp.read()
        except Exception as e:
            last_err = f"local_judge_unreachable {url}: {e}".encode()
            continue
    return False, last_err


def call_judge_direct(payload, model, *, timeout=120):
    """Skip 9router entirely. Goes to the upstream base_url from
    MODEL_LOCAL_BASE_URLS (per model) when set, else falls through to the
    9router path inside _try_model. Used by the proxies in the judge block.
    """
    import json as _json, urllib.request as _ur
    base = MODEL_LOCAL_BASE_URLS.get(model)
    if not base:
        return None   # signal: caller should use 9router path
    body = _json.dumps({**payload, "model": model}).encode()
    req = _ur.Request(base.rstrip("/") + "/chat/completions",
                      data=body,
                      headers={"Content-Type": "application/json"},
                      method="POST")
    try:
        with _ur.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        return f"direct_unreachable: {e}".encode()


# Per-model direct base URLs. Populated lazily by the proxies from env.
MODEL_LOCAL_BASE_URLS = {}


# ── Trading-absorption guard (Sep 7 2026) ─────────────────────────────
# Live-observed leak: "That is too brief and missing documents. I need
# every form field and the fees covered thoroughly." classified as
# trading_emergency ABOVE TRADING_FLOOR (dissatisfaction phrasing rhymes
# with panic phrasing in the embedding bank) and burned one Opus call.
# Embedding score alone is not enough for the per-turn trading override:
# require actual trading vocabulary. Word-boundary regexes, >=2 signals
# or >=1 hard keyword -- the same rule the skill prescribes for the
# keyword fallback path.
_TRADING_HARD_RE = re.compile(
    r"\b(margin\s+call|liquidat\w*|blow\s?n?\s*up|blowing\s+up|"
    r"forced\s+(sell|close|liquidat\w*)|flatten\s+(everything|all|my)|"
    r"sell\s+everything|get\s+me\s+(out|flat)|kill\s+all\s+my\s+"
    r"(trades|positions)|circuit\s+breaker|stop\s?loss\s+hit|"
    r"margin\s+requirement|rug\s+pull|bank\s+run)\b", re.I)
# STRONG soft signals: unambiguous trading vocabulary. One is enough.
_TRADING_STRONG_RE = re.compile(
    r"\b(portfolio|broker(age)?|shares?|equit\w+|stocks?|options?|"
    r"futures|ticker|trades|trading|leverage|drawdown|stop[\s-]loss|"
    r"take[\s-]profit|ibkr|alpaca|margin|unrealized|"
    r"realized\s+(gain|loss)|pnl|p/?l\b|etf|spy|qqq|"
    r"short\s+(the\s+)?(stock|market|position|squeeze)|"
    r"overnight\s+(hold|risk)|pyramid(ing)?\s+into|order\s+fill)\b", re.I)
# WEAK soft signals: collide with ordinary text ("exit code", "position:"
# in CSS, "in order to", "a long time"). Need two, or one + a ticker.
_TRADING_WEAK_RE = re.compile(
    r"\b(positions?|orders?|entry|entries|exit|exits|long|short|"
    r"buy|sell|hold|close|open|fills?|setup|setups|scalp|swing)\b", re.I)
# ALL-CAPS ticker-like token (NVDA, AAPL, TSLA), 2-5 letters.
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")


def trading_signal_ok(text):
    """True when `text` carries real trading vocabulary.

    Pass rules (strictest first):
      1. hard keyword anywhere (margin call, liquidation, flatten...) --
         unambiguous emergency phrasing.
      2. >=1 STRONG signal (portfolio, broker, trades, stop-loss...).
      3. >=2 distinct WEAK signals, or 1 weak + an ALL-CAPS ticker.
    The weak tier exists because "position", "order", "exit", "long" all
    collide with ordinary text (the exact collisions the classifier skill
    documents). The live leak this guards against -- "missing documents...
    fees covered thoroughly" classified trading_emergency above floor --
    has no strong signal, no hard keyword, and only weak collisions at
    most one, so it correctly fails here.
    """
    if not text:
        return False
    if _TRADING_HARD_RE.search(text):
        return True
    if _TRADING_STRONG_RE.search(text):
        return True
    weaks = {m.group(0).lower() for m in _TRADING_WEAK_RE.finditer(text)}
    if len(weaks) >= 2:
        return True
    if len(weaks) == 1 and _TICKER_RE.search(text):
        return True
    return False
