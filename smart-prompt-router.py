#!/usr/bin/env python3
"""
Smart Prompt Router — Production Grade (1% Trader Standard)
===========================================================
- Semantic classification with embedding similarities
- Graceful fallback when ML unavailable
- Input validation, error handling, metrics
- Suitable for high-velocity trading ops

Tier routing:
  trading_emergency → Opus    (margin calls, liquidations)
  trading_decision  → Sonnet  (all other trading decisions)
  trading_info      → GLM     (simple market info)
  coding_hard       → Sonnet  (debugging, architecture)
  coding_routine    → GLM     (simple code tasks)
  system_admin      → Sonnet  (deployment, config)
  research          → GPT-Sol (analysis, synthesis)
  creative          → GPT-Sol (writing, brainstorming)
  general           → Haiku   (trivial/fallback)

Usage:
  python smart-prompt-router.py "should i close my position"
"""

import json
import sys
import os
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_ML = True
    ML_IMPORT_ERROR = None
except ImportError as e:
    HAS_ML = False
    ML_IMPORT_ERROR = str(e)

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger('smart-router')

# ─── EXTENDED TRAINING EXAMPLES (for robust classification) ───
EXAMPLES = {
    "trading_emergency": [
        "my account got a margin call what should i do immediately",
        "i'm getting liquidated how do i stop this",
        "my positions are being forced closed help",
        "the market is crashing and i'm losing everything",
        "i'm about to blow up my account what do i do",
        "my broker just liquidated everything",
        "flatten everything now",
        "sell everything right now and get me flat",
        "get me out of all my positions",
        "kill all my trades",
        "i need to cut all positions right now urgently",
        "circuit breaker hit what happens now",
        "gap down at open i'm down huge",
        "my stop got hit but i'm still losing money why",
    ],
    "trading_decision": [
        "should i close this position right now",
        "analyze my portfolio risk and tell me what to do",
        "is this a good entry point for this stock",
        "should i pyramid into this winning trade",
        "compare these two exit strategies for risk management",
        "what's the best stop loss placement here",
        "should i take partial profits or let it run",
        "is this breakout real or a fakeout",
        "how should i size this position given my risk",
        "does this setup meet my trading criteria",
        "review my trades from today and tell me what went wrong",
        "scan the market and find me the best opportunities right now",
        "the momentum is fading should i exit",
        "this position just went red am i underwater",
        "should i hold overnight or close before market close",
        "what's my win rate on this type of setup",
        "did the shield re-place my stop after the fill",
        "is my stop loss still active on this position",
        "did my auto trader actually place the order",
        "are my protective stops in place right now",
        "check if my orders went through at the broker",
        "do i add to my swing position",
        "should i add more shares to this trade",
        "what is my cost basis on this position",
    ],
    "trading_info": [
        "what time does the market open",
        "is the market open today",
        "what's the current price of this stock",
        "when does trading end today",
        "what's the margin requirement for options",
        "what does rsi mean",
        "what's the difference between a market and limit order",
        "how much buying power do i have",
        "what's the bid ask spread on this",
        "what's the average volume for this stock",
    ],
    "coding_hard": [
        "debug this error and find the root cause",
        "design a distributed system architecture for this",
        "there's a race condition in my multi-threaded code",
        "help me optimize this bottleneck in production",
        "refactor this entire module to be cleaner",
        "this code has a memory leak help me find it",
        "design a fault-tolerant system that handles failures",
        "review this code for security vulnerabilities",
        "why is my application crashing in production",
        "implement a complex algorithm with proper error handling",
        "this thing keeps crashing intermittently under load",
        "the websocket handler is dropping connections",
        "our api is timing out in production",
        "we're leaking connections in the database pool",
        "design a database schema for this application",
        "design a data model for multi tenant billing",
        "why is my sql query slow and doing a full table scan",
        "how do i add an index to speed up this query",
        "what is the best architecture for this service",
        "how should i structure this codebase",
    ],
    "coding_routine": [
        "write a function to sort a list",
        "convert this code from python to javascript",
        "write a unit test for this function",
        "add a regex to validate this input",
        "write a sql query to get this data",
        "parse this json and extract the name field",
        "write a simple hello world program",
        "add type hints to this function",
        "fix this typo in my code",
        "how do i convert this string to int",
    ],
    "system_admin": [
        "set up docker compose for this application",
        "install and configure nginx reverse proxy",
        "deploy this to my production server",
        "configure my systemd service file",
        "set up ssl certificates for my domain",
        "help me configure this cron job",
        "troubleshoot why my server won't start",
        "set up monitoring and alerting for my system",
        "configure my firewall rules",
        "help me set up a new user account with ssh access",
        "install and wire up this tool into my stack",
        "i need to make this accessible from the internet securely",
        "how do i backup this database",
        "my server is out of disk space what do i do",
    ],
    "research": [
        "research the latest developments in this field and summarize",
        "analyze this document and extract key insights",
        "compare and contrast these different approaches",
        "summarize this research paper for me",
        "do a deep dive into how this technology works",
        "evaluate the pros and cons of these options",
        "write a detailed analysis of this market trend",
        "find me the best resources to learn about this topic",
        "what are the key risks i'm missing here",
        "compare the tradeoffs between these two technologies",
        "which approach should i choose and why",
        "explain the differences between these frameworks",
        "what are the pros and cons of x versus y",
        # Q5 FIX (Fable review): every example above is ABSTRACT ("these two
        # technologies", "x versus y") with no concrete technical nouns, so a
        # real named-stack comparison had nothing to match on:
        # "compare vLLM and llama.cpp for serving on a GB10" scored only
        # 0.264 -- barely above chit-chat ("explain what a hash map is" =
        # 0.261) -- and got demoted to the cheap tier by the 0.30 research
        # cap. Separating those two by threshold is impossible (0.003 of
        # headroom, below embedding noise), so the fix is to teach the
        # classifier what a concrete infra/tooling comparison looks like.
        "compare vLLM and TGI for serving llama models on a single GPU",
        "benchmark llama.cpp versus onnxruntime for local inference",
        "should i use postgres or clickhouse for time series at this scale",
        "evaluate kafka against redis streams for this ingest pipeline",
        "compare quantization formats for running this model on consumer hardware",
        "which inference runtime gives the best throughput on this hardware",
    ],
    "creative": [
        "write a blog post about this topic",
        "draft an email to my team about this",
        "write a story about this character",
        "brainstorm some creative ideas for this project",
        "write a product description for my website",
        "help me write a presentation script",
        "create engaging social media content",
        "write a press release for this announcement",
        "draft a professional email to investors",
    ],
    "general_easy": [
        "what time is it",
        "hello how are you",
        "what's the capital of france",
        "thanks",
        "good morning",
        "what's your name",
        "how are you doing today",
        "what does this word mean",
        "give me a quick summary",
        "hey what's up",
    ],
    # personal: everyday-life how-to/planning/chat from non-trading profiles
    # (mila = schedule/trip/family planning, jimmy = general questions).
    # Added Sep 1 2026: "how do i tie a tie properly" scored trading_decision
    # @ 0.233 (>= TRADING_FLOOR 0.18) -> Opus, because trading_decision's
    # example bank is full of "how should i..." phrasings. This domain gives
    # those prompts a correct home on the cheap tier.
    "personal": [
        "how do i tie a tie properly",
        "what should i make for dinner tonight",
        "help me plan my schedule for this week",
        "what's a good bedtime for a 3 year old",
        "how do i pack a suitcase for a long trip",
        "help me plan our trip to china in march",
        "draft a message to my sister about the trip",
        "what should i pack for the kids for the flight",
        "remind me to call my mom this weekend",
        "how do i get my toddler to eat vegetables",
        "plan a weekly meal menu for the family",
        "what time should we leave for the airport",
        "how do i fix a leaky faucet at home",
        "best way to organize my closet",
        "how do i remove a coffee stain from a shirt",
        "what should i get my friend for their birthday",
        "how do i book a flight for three people",
        "schedule a dentist appointment for next week",
        "what's a fun weekend activity for the family",
        "how do i set up a chore chart for my kids",
    ],
}

# ─── TIER MAPPING ───
TIER_MAP = {
    "trading_emergency": {"tier": "tier-reasoning",  "reasoning": "high",   "label": "🔥 EMERGENCY → tier-reasoning"},
    "trading_decision":  {"tier": "tier-reasoning",  "reasoning": "high",   "label": "📈 TRADING → tier-reasoning"},
    "trading_info":      {"tier": "tier-fast",       "reasoning": "low",    "label": "📊 INFO → tier-fast"},
    "coding_hard":       {"tier": "tier-reasoning",  "reasoning": "high",   "label": "💻 CODING → tier-reasoning"},
    "coding_routine":    {"tier": "tier-standard",   "reasoning": "medium", "label": "⚙️ ROUTINE → tier-standard"},
    "research":          {"tier": "tier-frontier",   "reasoning": "high",   "label": "🔬 RESEARCH → tier-frontier"},
    "system_admin":      {"tier": "tier-standard",   "reasoning": "medium", "label": "🔧 SYSADMIN → tier-standard"},
    "creative":          {"tier": "tier-standard",   "reasoning": "medium", "label": "✨ CREATIVE → tier-standard"},
    "general_easy":      {"tier": "tier-fast",       "reasoning": "none",   "label": "💬 SIMPLE → tier-fast"},
}

DEFAULT_TIER = {"tier": "tier-fast", "reasoning": "none", "label": "💬 DEFAULT → Haiku (free)"}

# ─── Global state (cached model) ───
_MODEL = None
_EMBEDDINGS = None
_EXAMPLE_KEYS = None
_METRICS = {"total_calls": 0, "semantic_success": 0, "fallback_used": 0, "errors": 0}

# H1 FIX (Fable review): _METRICS was mutated from request threads with no
# guard. Under the plain (serial) HTTPServer that was harmless, but the
# proxies now run ThreadingHTTPServer, so `d[k] += 1` -- a non-atomic
# read-modify-write -- races and silently loses counts. Guard every mutation
# and take a consistent snapshot on read.
_METRICS_LOCK = threading.Lock()


def _metric_inc(key: str, n: int = 1) -> None:
    """Thread-safe counter bump."""
    with _METRICS_LOCK:
        _METRICS[key] = _METRICS.get(key, 0) + n


def _validate_input(prompt: str) -> Optional[str]:
    """Validate and sanitize input. Return error string if invalid, None if OK."""
    if prompt is None:
        return "Prompt is None"
    if isinstance(prompt, bytes):
        try:
            prompt = prompt.decode('utf-8')
        except UnicodeDecodeError:
            return "Prompt is not valid UTF-8"
    if not isinstance(prompt, str):
        return f"Prompt must be str, got {type(prompt).__name__}"
    
    prompt = prompt.strip()
    if len(prompt) == 0:
        return "Prompt is empty"
    if len(prompt) > 50000:
        return "Prompt exceeds 50K chars (too long)"
    
    return None


# Phase 1 (Sep 19 2026 — user directive): STRICT_SEMANTIC is now a HARD,
# non-overridable invariant. The keyword fallback path has been REMOVED from
# this module entirely. The router MUST classify every prompt with the
# embedding model; if the model fails to load, classify() raises and the
# proxy exits non-zero. There is no `classify_fallback()` left to call.
# Setting SEMANTIC_STRICT=0 is a no-op — kept as a constant for backwards
# compatibility with imports, but classify() never reads it.
STRICT_SEMANTIC = True
os.environ.pop("SEMANTIC_STRICT", None)  # make the env override a hard no-op

# Phase 4 / 4c: module-level constants that mirror the env-var defaults.
# The classify() function re-reads these for each call (so a process can
# tune them via env), but proxies need access at import time to make
# routing decisions -- so we expose them as module constants.
TRADING_FLOOR = float(os.environ.get("TRADING_FLOOR", "0.18"))
ABSOLUTE_FLOOR = float(os.environ.get("ABSOLUTE_FLOOR", "0.20"))
LOW_CONF_DEMOTE = float(os.environ.get("LOW_CONF_DEMOTE", "0.18"))
SHORT_PROMPT_CHARS = int(os.environ.get("SHORT_PROMPT_CHARS", "120"))

# Phase 4b: per-domain confidence caps for the "explain-this-basic-concept"
# trap. These domains have wide semantic footprints and absorb chit-chat at
# seemingly-strong scores ("explain what a hash map is" -> research @ 0.261).
# Below the cap AND short -> demote to the cheap tier.
#   - research/analysis: genuine research scores 0.40+; cap 0.30 sweeps chat.
#   - creative: genuine creative writing scores 0.40+; cap 0.30 sweeps recipes.
#   - coding_routine: one-liners score 0.20-0.30; real coding scores 0.40+.
#   - system_admin: short factual sysadmin questions leak in at 0.25-0.30.
#   - trading_*: deliberately NOT capped; the TRADING_FLOOR rule governs them.
# M1 FIX: single module-level definition. classify_semantic must NOT shadow
# these with local env re-reads (that was the shadowing bug Fable caught).
PER_DOMAIN_CAP = {
    "research":       float(os.environ.get("RESEARCH_CONF_CAP", "0.30")),
    "analysis":       float(os.environ.get("RESEARCH_CONF_CAP", "0.30")),
    "creative":       float(os.environ.get("CREATIVE_CONF_CAP", "0.30")),
    "coding_routine": float(os.environ.get("CODING_ROUTINE_CONF_CAP", "0.30")),
    "system_admin":   float(os.environ.get("SYSADMIN_CONF_CAP", "0.30")),
}

# H3 FIX (Fable review): the Phase 4c trading exemption was written as
#     score < LOW_CONF_DEMOTE and not (trading and score >= TRADING_FLOOR)
# which at the default config (both == 0.18) requires score < 0.18 AND
# score >= 0.18 -- provably unsatisfiable, i.e. dead code. It only "worked"
# because the two constants happened to be equal. Guard against a future
# tune silently resurrecting the mount-everest leak (or, worse, demoting a
# genuine trading emergency) by asserting the invariant the design needs:
# a trading prompt must never be demotable while it is at/above its floor.
if TRADING_FLOOR > LOW_CONF_DEMOTE:
    logger.warning(
        "TRADING_FLOOR (%.3f) > LOW_CONF_DEMOTE (%.3f): trading prompts "
        "scoring between them are exempt from the low-confidence demote. "
        "This is the intended knob, but verify it is deliberate.",
        TRADING_FLOOR, LOW_CONF_DEMOTE,
    )

# Phase 1: guard against the check-then-act race where two concurrent first
# requests both enter _load_model() and double-load the model.
_MODEL_LOCK = threading.Lock()


def _load_model():
    """Load embedding model (one-time). Cached globally. Thread-safe."""
    global _MODEL, _EMBEDDINGS, _EXAMPLE_KEYS
    if _MODEL is not None:
        return

    with _MODEL_LOCK:
        # Re-check inside the lock: another thread may have loaded it already.
        if _MODEL is not None:
            return

        logger.info("Loading SentenceTransformer model...")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")

            texts, domains = [], []
            for domain, examples in EXAMPLES.items():
                for ex in examples:
                    texts.append(ex)
                    domains.append(domain)

            logger.info(f"Encoding {len(texts)} examples...")
            embeddings = model.encode(texts, convert_to_tensor=True)

            # Publish to globals only after every step succeeded, so a partial
            # failure can never leave a half-initialised model visible.
            _EMBEDDINGS = embeddings
            _EXAMPLE_KEYS = domains
            _MODEL = model
            logger.info("Model ready")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise


def preload(strict: bool = True) -> bool:
    """Eagerly load the model at STARTUP, before the server binds its port.

    Phase 1 fix: the proxy calls this before serving. On failure the caller
    exits non-zero so systemd restarts it, instead of quietly serving keyword
    routing (which mis-routes 'exit code 1' -> Opus and real trading questions
    -> the cheap tier).
    """
    if not HAS_ML:
        msg = f"sentence_transformers unavailable: {ML_IMPORT_ERROR}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False

    _load_model()

    # Prove the semantic path actually works end-to-end, not just that the
    # model object exists.
    probe = classify_semantic("flatten everything now get me out of all positions")
    if not probe or probe.get("method") != "semantic":
        raise RuntimeError("preload probe did not return a semantic result")
    logger.info(
        f"Semantic preload verified (probe -> {probe['domain']} "
        f"@ {probe.get('top_similarity')})"
    )
    return True


def classify_semantic(prompt: str) -> Dict[str, Any]:
    """Classify using embedding similarity."""
    try:
        _load_model()
        
        # Encode prompt
        prompt_emb = _MODEL.encode([prompt], convert_to_tensor=True)
        
        # Cosine similarity: normalize then dot product
        prompt_norm = prompt_emb / torch.norm(prompt_emb, dim=1, keepdim=True)
        emb_norm = _EMBEDDINGS / torch.norm(_EMBEDDINGS, dim=1, keepdim=True)
        sims = torch.mm(prompt_norm, emb_norm.T)[0]
        sims_np = sims.cpu().numpy()
        
        # Average top-3 scores per domain
        domain_scores = {}
        for i, domain in enumerate(_EXAMPLE_KEYS):
            sim = float(sims_np[i])
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(sim)
        
        domain_avg = {}
        for domain, scores in domain_scores.items():
            top_k = sorted(scores, reverse=True)[:3]
            domain_avg[domain] = sum(top_k) / len(top_k)
        
        # Find best domain
        best_domain = max(domain_avg, key=domain_avg.get)
        best_score = domain_avg[best_domain]
        
        # Confidence: gap between 1st and 2nd best
        all_scores_sorted = sorted(domain_avg.values(), reverse=True)
        second_best = all_scores_sorted[1] if len(all_scores_sorted) > 1 else 0
        margin = best_score - second_best
        
        # Confidence = product of (absolute score) and (relative gap)
        confidence = min(best_score * max(0.5, 1.0 + (margin * 1.5)), 1.0)
        
        # ── Phase 3 threshold policy ──
        # Old: best_score < 0.30 -> blanket dump to "general" tier-fast.
        # That sat inside the confusion band on the wrong side: correct
        # classifications at 0.264 (creative, coding_hard, research) were
        # demoted while genuine general queries topped out at 0.26.
        #
        # New policy (per Fable's review):
        # 1. Floor at 0.20 - between the 0.26 general ceiling and 0.20 stops
        #    we leak some general->domain. Phase 4 re-routes that leak back to
        #    the cheap tier (quota-burn/Haiku) so we don't pay Sonnet for what
        #    is almost certainly a chit-chat prompt.
        # 2. Trading-floor rule: trading_* domains have a 0.18 floor because
        #    a missed trading_decision is more expensive than over-spending
        #    Opus by your own criteria (asymmetric risk).
        # 3. Always report the ARGMAX winner. Below floor we still pick the
        #    best-matching domain (could be general_easy) at low confidence;
        #    we never overwrite the answer with a fixed 'general' string.
        # M1 FIX (Fable review): these thresholds used to be re-read from env
        # as LOCALS here, shadowing the module-level constants that the proxies
        # import. That is the exact bug class Phase 4c claimed to fix -- if a
        # systemd unit set TRADING_FLOOR per-process, the library and the proxy
        # would silently enforce different floors. There is now ONE definition,
        # at module scope (see ~line 248). Do not re-read env here.
        #
        # Phase 4b: per-domain confidence caps for the "explain-this-basic-
        # concept" trap. These domains have a wide semantic footprint and
        # absorb chit-chat prompts at seemingly-strong scores (e.g. "explain
        # what a hash map is" → research @ 0.261; "give me a quick dinner
        # idea" → creative @ 0.284). The cap is the score at or above which
        # we trust the user actually wants a depth-grade answer for this
        # domain. Below the cap, demote to tier-fast.
        #   - trading_*: NOT capped -- the trading-floor rule governs those.
        #
        # NOTE: caps alone over-fire. Phase 4c adds a length/tone override:
        # a LONG, structured prompt that lands on a research/creative domain
        # IS likely genuine, even at low score. Caps only apply to SHORT
        # prompts (< SHORT_PROMPT_CHARS chars). Above that length we trust
        # the topic domain at any score.

        # Track whether the domain was demoted (Phase 4 signal for telemetry
        # and for verifying 'chit-chat didn't get Sonnet' end-to-end).
        demoted_to_fast = False
        demote_reason = None

        # Override the best domain if trading safety floor applies:
        # even if another domain scored slightly higher but a trading_*
        # domain scored at or above its own floor, prefer the trading one.
        if best_score >= ABSOLUTE_FLOOR:
            tier_info = TIER_MAP.get(best_domain, DEFAULT_TIER)
        else:
            trading_candidates = {
                d: s for d, s in domain_avg.items()
                if d.startswith("trading_") and s >= TRADING_FLOOR
            }
            if trading_candidates and (
                best_domain not in trading_candidates
                or best_domain.startswith("trading_") is False
            ):
                # If a trading domain clears its own floor, use it regardless
                # of which domain is the global argmax. (User pref: trading
                # false-negative is more expensive than a slight Opus overspend.)
                best_domain = max(trading_candidates, key=trading_candidates.get)
                best_score = trading_candidates[best_domain]
                tier_info = TIER_MAP.get(best_domain, DEFAULT_TIER)
            else:
                # Below all floors: still return the argmax winner (typically
                # 'general' or 'general_easy'). We do NOT override to a fixed
                # 'general' string - that destroyed the correct
                # creative/coding/research classifications under the old code.
                tier_info = TIER_MAP.get(best_domain, DEFAULT_TIER)

        # ── Phase 4: tier demote for low-confidence non-trading domains ──
        # Below LOW_CONF_DEMOTE the classifier is essentially guessing. The
        # asymmetric cost says: a wrong-but-cheap answer to a borderline
        # creative/research/coding prompt costs less than burning Sonnet on
        # what is overwhelmingly likely to be a chit-chat prompt.
        #
        # Trading domains are treated specially:
        #   - If a trading_* domain cleared its own floor (>= TRADING_FLOOR),
        #     we trust it: the user explicitly opted in to expensive models
        #     for these.
        #   - If a trading_* domain is the argmax winner BUT scored BELOW
        #     TRADING_FLOOR, we demote to fast. The classifier wasn't
        #     confident enough to merit Opus, and a false positive on
        #     "mount everest -> trading_decision @ 0.094" should not burn
        #     tier-reasoning budget.
        if (
            best_score < LOW_CONF_DEMOTE
            and not (
                best_domain.startswith("trading_")
                and best_score >= TRADING_FLOOR
            )
        ):
            tier_info = DEFAULT_TIER  # tier-fast (Haiku / GLM quota-burn)
            demoted_to_fast = True
            demote_reason = f"abs_score<{LOW_CONF_DEMOTE}"

        # ── Phase 4b: per-domain confidence cap ──
        # Catches the 'high-conf but wrong domain' leak: chit-chat prompts
        # that semantically belong to research/creative/coding_routine but
        # are too lightweight to merit Sonnet/Fable. Each domain has its own
        # cap (see PER_DOMAIN_CAP). Below the cap → demote to tier-fast.
        #
        # Phase 4c: only apply the cap to SHORT prompts. A 200-char prompt
        # that lands on research@0.30 IS likely genuine research even if
        # the score sits right at the cap. The cap is for chit-chat, and
        # chit-chat is short.
        elif (
            best_domain in PER_DOMAIN_CAP
            and best_score < PER_DOMAIN_CAP[best_domain]
            and len(prompt) <= SHORT_PROMPT_CHARS
        ):
            tier_info = DEFAULT_TIER  # tier-fast (Haiku / GLM quota-burn)
            demoted_to_fast = True
            demote_reason = (
                f"{best_domain}_cap<{PER_DOMAIN_CAP[best_domain]}_"
                f"short<={SHORT_PROMPT_CHARS}"
            )
        
        sorted_scores = sorted(domain_avg.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "domain": best_domain,
            "tier": tier_info["tier"],
            "reasoning_effort": tier_info["reasoning"],
            "label": tier_info["label"],
            "confidence": round(float(confidence), 3),
            "top_similarity": round(float(best_score), 3),
            "method": "semantic",
            "demoted_to_fast": demoted_to_fast,  # Phase 4 telemetry flag
            "demote_reason": demote_reason,      # Phase 4b telemetry
            "all_scores": {k: round(float(v), 3) for k, v in sorted_scores[:3]},
            # Sep 21 2026 — exposed for classifier_hybrid.maybe_hybrid_override().
            # Internal key (underscore-prefixed) so it doesn't leak into the
            # public response shape or the /v1/models card.
            "_all_dense_scores": {k: round(float(v), 4) for k, v in domain_avg.items()},
            "_dense_argmax": best_domain,
        }
    except Exception as e:
        # Sep 19 2026 — keyword fallback removed per user directive. Re-raise
        # so the caller surfaces a 500 and the proxy stays strictly semantic.
        logger.error(f"Semantic classification failed ({e}); no keyword fallback available")
        raise


def classify_fallback(prompt: str) -> Dict[str, Any]:
    """REMOVED Sep 19 2026 per user directive.

    The router is strictly semantic. This function exists ONLY so that
    any stale import does not crash at module load; calling it raises.
    """
    raise RuntimeError(
        "classify_fallback() has been removed. The semantic router classifies "
        "every prompt with embeddings; there is no keyword-based path."
    )


def classify(prompt: str) -> Dict[str, Any]:
    """Main entry point with validation and error handling."""
    _metric_inc("total_calls")
    
    # Phase 3 fix: validation errors and oversized prompts used to land on
    # tier-fast (the cheapest model). For trading work that's exactly
    # backwards - a 50K-char portfolio dump goes to Sonnet now.
    validation_error = _validate_input(prompt)
    if validation_error:
        _metric_inc("errors")
        logger.error(f"Input validation failed: {validation_error}")
        return {
            "domain": "general",
            "tier": "tier-standard",
            "reasoning_effort": "medium",
            "label": DEFAULT_TIER["label"],
            "confidence": 0.0,
            "method": "error",
            "error": validation_error,
        }
    
    # Sep 19 2026 — keyword fallback removed. classify_semantic() now
    # re-raises on failure, so this branch is the only return path.
    if not HAS_ML:
        _metric_inc("errors")
        raise RuntimeError(
            f"sentence_transformers unavailable: {ML_IMPORT_ERROR}"
        )

    semantic = classify_semantic(prompt)

    # Sep 21 2026 — Hybrid BM25+dense second opinion (steal #1 from
    # aurelio-labs/semantic-router). Consulted in the dense confusion band.
    # When BM25 has a clearly better answer, override the dense domain and
    # re-resolve the tier from TIER_MAP. Phase 4 (TRADING_FLOOR,
    # PER_DOMAIN_CAP, SHORT_PROMPT_CHARS) was already applied inside
    # classify_semantic() — we re-run the demote check on the new domain
    # so the trading-floor invariant survives a swap.
    try:
        from classifier_hybrid import maybe_hybrid_override
        dense_scores = semantic.get("_all_dense_scores") or {}
        if dense_scores:
            override = maybe_hybrid_override(
                prompt,
                dense_scores=dense_scores,
                dense_argmax=semantic.get("_dense_argmax") or semantic["domain"],
                dense_score=semantic.get("top_similarity", 0.0),
            )
            if override and override.get("method") == "hybrid":
                new_domain = override["domain"]
                # Re-resolve tier for the new domain. Phase 4 demote was
                # already applied to the original domain; for a non-trading
                # domain swap the tier is just TIER_MAP.get(new_domain).
                # For a trading_* swap (rare — BM25 rarely overrides into
                # trading), honor TRADING_FLOOR by checking dense_score.
                if new_domain.startswith("trading_"):
                    # Only honor trading-floor override if dense was
                    # actually confident in the original trading domain.
                    # Otherwise we'd let BM25 manufacture a trading_decision.
                    if semantic.get("top_similarity", 0.0) < TRADING_FLOOR:
                        # Don't let BM25 promote to trading without dense
                        # backing — keep dense's domain.
                        pass
                    else:
                        semantic["domain"] = new_domain
                        semantic["tier"] = TIER_MAP.get(new_domain, DEFAULT_TIER)["tier"]
                else:
                    semantic["domain"] = new_domain
                    semantic["tier"] = TIER_MAP.get(new_domain, DEFAULT_TIER)["tier"]
                semantic["method"] = "hybrid"
                semantic["hybrid_blended"] = override.get("hybrid_blended")
                semantic["hybrid_bm25_top"] = override.get("hybrid_bm25_top")
                # Re-flag demotion state for the new domain.
                if semantic["tier"] in ("tier-fast", "tier_quota_burn", "tier_local", "tier-general"):
                    semantic["demoted_to_fast"] = True
                    semantic["demote_reason"] = (
                        semantic.get("demote_reason") or "hybrid_override_cheap_tier"
                    )
    except ImportError:
        # rank-bm25 not installed — silent fallback to dense-only. The Sep 19
        # "strictly semantic" invariant is preserved.
        pass
    except Exception as _exc:
        # Any other hybrid failure (BM25 build error, score normalize crash)
        # must NOT break routing. Log and serve dense result.
        logger.debug(f"hybrid override skipped: {_exc}")

    _metric_inc("semantic_success")
    return semantic


def get_metrics() -> Dict[str, Any]:
    """Return classification metrics.

    H1 FIX: takes a consistent snapshot under the lock. The old version read
    _METRICS four separate times while other threads mutated it, so the
    derived rates could be computed from a torn view (e.g. semantic_success
    from after an increment, total_calls from before it -- yielding a rate
    above 100%). It also returned the LIVE dict on the total==0 path, letting
    callers mutate router state.
    """
    with _METRICS_LOCK:
        snap = dict(_METRICS)

    total = snap["total_calls"]
    if total == 0:
        return snap

    return {
        **snap,
        "semantic_rate": round(snap["semantic_success"] / total * 100, 1),
        "fallback_rate": round(snap["fallback_used"] / total * 100, 1),
        "error_rate": round(snap["errors"] / total * 100, 1),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smart-prompt-router.py '<prompt>'")
        print("       python smart-prompt-router.py --metrics")
        sys.exit(1)
    
    if sys.argv[1] == "--metrics":
        print(json.dumps(get_metrics(), indent=2))
        sys.exit(0)
    
    prompt = " ".join(sys.argv[1:])
    result = classify(prompt)
    print(json.dumps(result, indent=2))
