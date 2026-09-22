#!/usr/bin/env python3
"""Shared tier configuration for ALL semantic-router proxies (default/jimmy/mila).

SINGLE SOURCE OF TRUTH — consolidated Sep 1, 2026 after the Fable review found
3 drifting private copies of TIER_SYSTEM inside each proxy. Every proxy now does:

    from router_tiers import TIER_SYSTEM, DOMAIN_TO_TIER

Change tiers HERE only, then restart all three:
    systemctl --user restart smart-semantic-router-proxy jimmy-semantic-router-proxy mila-semantic-router-proxy

Only models the user actually has plans for:
  cc/ (Claude OAuth) | glm-cn/ (API key) | minimax/ (API key)
NEVER use nvidia/ slugs - that is a different account.

Context-window policy (Sep 1, live-tested — see reviews/empirical-context-proof-2026-09-01.md):
  1M-context models ONLY in every tier. glm-5.3 / glm-5.3-flash / glm-5.2 / MiniMax-M3
  all proven >262k live (catalog: 1M). MiniMax-M2.7 real ceiling = 262,144 (bisection
  Sep 1) — EXCLUDED from all tiers. Do NOT trust 9router /v1/models context_length
  values; they are stale defaults.
  Sep 7: alitp-intl/qwen3.8-max-preview PROVEN 723,291-token prompt live (both
  markers recalled) — passes the 1M-only policy. Alibaba Token Plan (40k credits/mo,
  user's Standard plan) inserted AFTER the GLM chain and BEFORE MiniMax per user
  decision Sep 7: GLM keeps its burn-first role (Aug 30 directive), Alibaba is the
  quota-exhaustion fallback. alitp-intl/glm-5.2 (GLM via Alibaba credits, separate
  quota from the capped glm-cn plan) sits between them.
  Sep 7 late: qwen3.8-max-preview observed HANGING (>120s, HTTP 000) on
  some prompts while qwen3.6-flash answers the same prompt in ~27s —
  flash inserted immediately after preview so a preview hang falls to a
  live sibling instead of burning 3x180s retries before the breaker trips.
  Sep 7 FINAL ORDER: qwen3.6-flash FIRST, qwen3.8-max-preview SECOND —
  preview hung (HTTP 000, >90s) on 4/4 prose prompts at 03:15 PDT while
  flash answered the same prompt in 27s and alitp glm-5.2 in 8s. A
  'preview' endpoint can be pulled or throttled at any time; flash is the
  stable workhorse. Preview stays in-chain as the smart fallback.
"""

# Minimum output budget. Thinking models (GLM 5.3, MiniMax M3, Claude with
# extended thinking) spend 300-1300 tokens reasoning BEFORE emitting content.
# A low max_tokens returns finish_reason='length' with content='' (empty reply).
MIN_MAX_TOKENS = 1500

# Upstream read timeout. GLM 5.3 measured 35-55s on reasoning tasks; a 30s
# timeout caused retry storms and fell through to Claude, defeating quota burn.
UPSTREAM_TIMEOUT = 180

TIER_SYSTEM = {
    "tier_quota_burn": {
        # Sep 19 2026 REBALANCE: M3 owns this lane (13B-tok plan), not Kimi.
        # Order: M3 → glm-flash → alitp-qwen3.6-flash → alitp-deepseek-v4-pro.
        # Kimi dropped (was at head, kept hitting 403 weekly); Sonnet dropped
        # (reserved for tier_complex); cx/* dropped (demoted to tier_fallback).
        "name": "M3 → glm-flash → alitp-qwen3.6-flash → alitp-deepseek-v4-pro (Quota Burn)",
        "models": [
            "minimax/MiniMax-M3",
            "glm-cn/glm-5.3-flash",
            "alitp-intl/qwen3.6-flash",
            "alitp-intl/deepseek-v4-pro",
        ],
        "plan": "alibaba_token_plan",  # secondary draw after M3
        "domains": ["system_admin", "creative"],   # general_easy/coding_routine/personal moved to tier_local (Sep 19)
    },
    "tier_general": {
        # Sep 19 2026 REBALANCE: same M3-first chain as tier_quota_burn.
        # Catch-all for prompts that don't classify to a domain — should
        # stay cheap-first so we never accidentally burn Claude quota.
        "name": "M3 → glm-flash → alitp-qwen3.6-flash → alitp-deepseek-v4-pro (General)",
        "models": [
            "minimax/MiniMax-M3",
            "glm-cn/glm-5.3-flash",
            "alitp-intl/qwen3.6-flash",
            "alitp-intl/deepseek-v4-pro",
        ],
        "plan": "alibaba_token_plan",
        "domains": ["general"],
    },
    "tier_vision": {
        # Sep 9 2026 NEW: vision-grounded prompts (image/screenshot input).
        # Leads with multimodal qwen3.7-plus (Vision2Web 77.8, OfficeQA Pro 62.4)
        # which beats GLM-Flash on screenshot→code tasks. Falls to GLM-Flash
        # (also multimodal, $0.15/$0.50 vs Plus $0.40/$1.60), then Sonnet.
        # NOTE: alitp-intl/qwen3.7-max is text-only per docs — do NOT add here.
        # Sep 19 2026 REBALANCE: M3 promoted above Sonnet (M3 has vision, and
        # burning the 13B plan is cheaper than burning Claude Max quota on a
        # vision task that M3 can handle).
        "name": "alitp-qwen3.7-plus → M3 → glm-flash → sonnet (Vision)",
        "models": [
            "alitp-intl/qwen3.7-plus",
            "minimax/MiniMax-M3",
            "glm-cn/glm-5.3-flash",
            "cc/claude-sonnet-5",
        ],
        "plan": "alibaba_token_plan",
        "domains": ["vision"],  # router maps multimodal-input detection to this domain
    },
    "tier_complex": {
        # Sep 19 2026 REBALANCE: Claude Sonnet owns this lane (coding_hard /
        # research / analysis). Claude Max plan confirmed (Q4-D, all Claude
        # models available), so Sonnet can be primary here. Falls to alibaba-
        # DeepSeek (good reasoning), then GLM-Flash, then escalates through
        # the rest of Claude (Opus → Fable) before M3 safety net. Kimi
        # dropped from this tier (kept only in tier_fallback as last-resort).
        "name": "sonnet → alitp-deepseek-v4-pro → glm-flash → opus → fable-5-1 → M3 (Complex)",
        "models": [
            "cc/claude-sonnet-5",
            "alitp-intl/deepseek-v4-pro",
            "glm-cn/glm-5.3-flash",
            "cc/claude-opus-5",
            "cc/claude-fable-5-1",
            "minimax/MiniMax-M3",
        ],
        "plan": "claude_max",
        "domains": ["coding_hard", "research", "analysis"],
    },
    "tier_trading_critical": {
        # Sep 9 2026: trading stays Claude-first per user directive — Kimi's
        # 164s TTFT is unacceptable for margin calls. Opus is the primary;
        # Fable 5.1 sits as the LAST Claude option (per user: Fable consumes
        # the most Max-plan quota, so we hold it in reserve). M3 is the safety
        # net. Subject to TRADING_FLOOR demote (separate config).
        "name": "Opus → Fable 5.1 → M3 (Trading Critical)",
        "models": [
            "cc/claude-opus-5",
            "cc/claude-fable-5-1",
            "minimax/MiniMax-M3",
        ],
        "plan": "claude_max",
        "domains": ["trading_emergency", "trading_decision"],
    },
    "tier_fallback": {
        # Sep 21 2026 REBALANCE: M3 first, then Kimi (last-resort per Q3),
        # then ChatGPT Plus (cx/*) as overflow (Q5 demote), then local Qwen
        # as the final free safety net.
        "name": "M3 → kimi → cx-luna → cx-terra → cx-luna-review → local Qwen (Last Resort Fallback)",
        "models": [
            "minimax/MiniMax-M3",
            "kimi/k3",                  # last-resort only (Q3-A)
            "cx/gpt-5.6-luna",          # demoted to fallback only (Q5-C)
            "cx/gpt-5.6-terra",
            "cx/gpt-5.6-luna-review",
            "local/hermes-agent",       # final free safety net — MOA gateway
        ],
        "plan": "minimax_m3",
        "domains": None,
    },
    "tier_local": {
        # Sep 21 2026 19:16 PDT REVERT (user confirmed): a sibling agent
        # swapped this to MOA / hermes-agent at end of day; user did not
        # request it. Pointed BACK at SGLang qwen3-30b-a3b-instruct-2507
        # @ :11435 (SGLang container healthy, 65 tok/s verified). The
        # MOA gateway at :8642 is left untouched and the system defaults
        # MOA_LOCAL_URL=/MODEL are bumped, so a future SGLang outage can
        # still flip to MOA via env vars without code changes.
        #
        # History:
        #   - Sep 19: SGLang qwen38-27b-abliterated @ :11434
        #   - Sep 21 AM: SGLang qwen3-30b-a3b-instruct-2507 @ :11435
        #   - Sep 21 PM: MOA hermes-agent @ :8642 (REVERTED below)
        # Cheap domains route here FIRST and degrade sideways into the
        # cloud cheap tiers if SGLang is down. Never reachable from a
        # frontier or trading primary (CHEAP_TIERS membership only).
        "name": "SGLang qwen3-30b-a3b-instruct-2507 (Free / Local :11435)",
        "models": ["local/qwen3-30b-a3b-instruct-2507"],
        "plan": "local_free",
        "domains": ["general_easy", "coding_routine", "personal"],
    },
}

# Domain -> primary tier. Any domain not listed here falls back to
# tier_general (cheap-first) per the .get(domain, ...) default.
DOMAIN_TO_TIER = {
    "trading_emergency": "tier_trading_critical",
    "trading_decision": "tier_trading_critical",
    "trading_info": "tier_general",        # info/lookup queries: cheap-first (don't burn Opus)
    "vision": "tier_vision",              # Sep 9 2026 NEW: multimodal-input prompts
    "coding_hard": "tier_complex",
    "research": "tier_complex",
    "analysis": "tier_complex",
    "system_admin": "tier_quota_burn",
    "creative": "tier_quota_burn",
    "general": "tier_general",            # Sep 9 2026: cheap-first catch-all
    "general_easy": "tier_local",
    "coding_routine": "tier_local",
    # personal: everyday-life how-to/planning/chat
    "personal": "tier_local",
}
