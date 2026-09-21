#!/usr/bin/env python3
"""Unit tests for router_escalation.py — run WITHOUT touching a live proxy.

    ~/.venvs/ibkr/bin/python3 test_escalation_advisor.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_escalation as R

P, F = 0, 0


def ok(name, cond, extra=""):
    global P, F
    if cond:
        P += 1
        print(f"[PASS] {name}")
    else:
        F += 1
        print(f"[FAIL] {name} {extra}")


def body(content, tool_calls=None, usage=(10, 20)):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return json.dumps({
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "model": "test",
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }).encode()


print("=" * 70)
print("1. CAPABILITY FLAGS (fail-closed AND across chain)")
print("=" * 70)
ok("all-Claude tier => vision True",
   R.tier_capabilities(["cc/claude-opus-5", "cc/claude-fable-5-1"])["vision"] is True)
ok("GLM chain => vision False",
   R.tier_capabilities(["glm-cn/glm-5.3", "minimax/MiniMax-M3"])["vision"] is False)
ok("MIXED chain fails CLOSED (Claude+GLM => no vision)",
   R.tier_capabilities(["cc/claude-opus-5", "glm-cn/glm-5.3"])["vision"] is False)
ok("unknown model fails closed",
   R.tier_capabilities(["who/knows"])["vision"] is False)
ok("empty chain fails closed",
   R.tier_capabilities([])["vision"] is False)

TS = {
    "tier_primary_critical": {"models": ["cc/claude-opus-5"], "plan": "claude_max"},
    "tier_quota_burn": {"models": ["glm-cn/glm-5.3", "minimax/MiniMax-M3"], "plan": "glm"},
}
payload = R.build_models_payload(TS, {})
ids = [m["id"] for m in payload["data"]]
ok("models payload has auto + tiers", "auto" in ids and "tier_quota_burn" in ids, ids)
crit = next(m for m in payload["data"] if m["id"] == "tier_primary_critical")
ok("Claude tier advertises image input",
   crit["input_modalities"] == ["text", "image"], crit["input_modalities"])
burn = next(m for m in payload["data"] if m["id"] == "tier_quota_burn")
ok("GLM tier advertises text only",
   burn["input_modalities"] == ["text"], burn["input_modalities"])
auto = next(m for m in payload["data"] if m["id"] == "auto")
ok("auto folds across ALL tiers => text only (fail closed)",
   auto["input_modalities"] == ["text"], auto["input_modalities"])

print()
print("=" * 70)
print("2. SESSION IDENTITY (stable as conversation grows)")
print("=" * 70)
m1 = [{"role": "system", "content": "You are Hermes." * 200},
      {"role": "user", "content": "help me plan a trip to Shanghai"}]
m2 = m1 + [{"role": "assistant", "content": "Sure, when?"},
           {"role": "user", "content": "March 13-22"}]
m3 = m2 + [{"role": "assistant", "content": "Got it"},
           {"role": "user", "content": "and hotels?"}]
ok("session key stable across turns",
   R.session_key(m1) == R.session_key(m2) == R.session_key(m3))
other = [{"role": "system", "content": "You are Hermes." * 200},
         {"role": "user", "content": "flatten all my positions now"}]
ok("different first user msg => different session",
   R.session_key(m1) != R.session_key(other))
ok("long shared system prompt does NOT collide (no truncation)",
   R.session_key(m1) != R.session_key(other))
nudge = [{"role": "system", "content": "You are Hermes." * 200},
         {"role": "user", "content": "You just executed tool calls but returned an empty response"},
         {"role": "user", "content": "help me plan a trip to Shanghai"}]
ok("boilerplate-first conversation skips to real msg",
   R.session_key(nudge) == R.session_key(m1))
ok("explicit client session header honored",
   R.session_key(m1, "abc") == R.session_key(m2, "abc")
   and R.session_key(m1, "abc") != R.session_key(m1))

print()
print("=" * 70)
print("3. VERDICT PARSING (None != decline)")
print("=" * 70)
ok("clean json escalate",
   R.parse_verdict(body('{"verdict":"escalate","reason":"looping"}')) == "escalate")
ok("clean json decline",
   R.parse_verdict(body('{"verdict":"decline","reason":"fine"}')) == "decline")
ok("json in prose",
   R.parse_verdict(body('Sure: {"verdict":"escalate","reason":"x"}')) == "escalate")
ok("bare word escalate", R.parse_verdict(body("escalate")) == "escalate")
ok("garbage => None (NOT decline)", R.parse_verdict(body("I think maybe?")) is None)
ok("empty content => None", R.parse_verdict(body("")) is None)
ok("malformed body => None", R.parse_verdict(b"not json at all") is None)

print()
print("=" * 70)
print("4. ESCALATION STREAK + FAIL-OPEN + BUDGET")
print("=" * 70)
st = R.StatsTracker(port=9999)
tr = R.EscalationTracker(stats=st)
sk = "s1"
s, l = tr.record_verdict(sk, "escalate")
ok("1st escalate: streak 1, no latch", s == 1 and not l, (s, l))
s, l = tr.record_verdict(sk, "escalate")
ok("2nd escalate: latches at confirmations=2", s == 2 and l, (s, l))
ok("is_latched true after latch", tr.is_latched(sk))

sk2 = "s2"
tr.record_verdict(sk2, "escalate")
s, l = tr.record_verdict(sk2, "decline")
ok("decline RESETS streak", s == 0 and not l, (s, l))

sk3 = "s3"
tr.record_verdict(sk3, "escalate")
s, l = tr.record_verdict(sk3, None)
ok("judge failure HOLDS streak (no reset)", s == 1, s)
ok("judge failure NEVER latches", not l and not tr.is_latched(sk3))

# budget
tr2 = R.EscalationTracker(stats=st)
R.ESCALATION_BUDGET_PER_HOUR_orig = R.ESCALATION_BUDGET_PER_HOUR
granted = 0
for i in range(R.ESCALATION_BUDGET_PER_HOUR + 5):
    k = f"burst{i}"
    tr2.record_verdict(k, "escalate")
    _, latched = tr2.record_verdict(k, "escalate")
    if latched:
        granted += 1
ok(f"budget caps latches at {R.ESCALATION_BUDGET_PER_HOUR}/hr",
   granted == R.ESCALATION_BUDGET_PER_HOUR, granted)
ok("budget denial counted in stats",
   st.snapshot(0)["counters"]["escalation_budget_denied"] == 5,
   st.snapshot(0)["counters"]["escalation_budget_denied"])

print()
print("=" * 70)
print("5. ESCALATION GATE (containment invariant)")
print("=" * 70)
ok("cheap lane + general => eligible",
   R.should_escalate("general", False, "tier_quota_burn", "cheap"))
ok("personal eligible", R.should_escalate("personal", False, "tier_quota_burn", "cheap"))
ok("DEMOTED never escalates (invariant)",
   not R.should_escalate("general", True, "tier_quota_burn", "cheap"))
ok("frontier lane never escalates (already Claude)",
   not R.should_escalate("coding_hard", False, "tier_primary_complex", "frontier"))
ok("trading_decision not in escalation set (starts on Opus)",
   not R.should_escalate("trading_decision", False, "tier_primary_critical", "frontier"))
ok("unknown domain not eligible",
   not R.should_escalate("weird_domain", False, "tier_quota_burn", "cheap"))

jr = tr.build_judge_request(m3, "here is my answer")
ok("judge uses cheap model", jr["model"] == R.JUDGE_MODEL, jr["model"])
ok("judge max_tokens >= 1500 (thinking-model floor)", jr["max_tokens"] >= 1500)
ok("judge sees completed turn",
   "JUST_COMPLETED" in jr["messages"][1]["content"])
long_msgs = [{"role": "user", "content": "x" * 5000}]
jr2 = tr.build_judge_request(long_msgs, "y" * 5000)
ok("judge truncates per-message",
   len(jr2["messages"][1]["content"]) < 3000, len(jr2["messages"][1]["content"]))

print()
print("=" * 70)
print("6. ADVISOR GATE (Fable 5.1)")
print("=" * 70)
ok("advisor model is Fable 5.1", R.ADVISOR_MODEL == "cc/claude-fable-5-1", R.ADVISOR_MODEL)
ag = R.AdvisorGate(stats=st)

tool_convo = [{"role": "system", "content": "sys"},
              {"role": "user", "content": "check my position risk"}]
for i in range(4):
    tool_convo.append({"role": "assistant", "content": None,
                       "tool_calls": [{"function": {"name": "get_pos"}}]})
    tool_convo.append({"role": "tool", "content": "AAPL 100sh"})

d, why = ag.should_review(domain="trading_decision", executor_model="glm-cn/glm-5.3",
                          messages=tool_convo, reply_has_tool_calls=False)
ok("gated domain + no_tool_call + enough tool results => review", d, why)

d, why = ag.should_review(domain="general", executor_model="glm-cn/glm-5.3",
                          messages=tool_convo, reply_has_tool_calls=False)
ok("ungated domain (general) => NO review", not d and why == "domain_not_gated", why)

d, why = ag.should_review(domain="personal", executor_model="glm-cn/glm-5.3",
                          messages=tool_convo, reply_has_tool_calls=False)
ok("personal => NO review (latency/cost)", not d, why)

d, why = ag.should_review(domain="trading_decision", executor_model="cc/claude-opus-5",
                          messages=tool_convo, reply_has_tool_calls=False)
ok("frontier executor => skip (no lift, double bill)",
   not d and why == "executor_already_frontier", why)

short = [{"role": "user", "content": "hi"}]
d, why = ag.should_review(domain="trading_decision", executor_model="glm-cn/glm-5.3",
                          messages=short, reply_has_tool_calls=False)
ok("early chatty turn => skip", not d and why == "early_chatty_turn", why)

d, why = ag.should_review(domain="trading_decision", executor_model="glm-cn/glm-5.3",
                          messages=tool_convo, reply_has_tool_calls=True)
ok("turn WITH tool calls => not terminal, skip", not d, why)

stall = [{"role": "user", "content": "go"}] + \
        [{"role": "assistant", "content": f"working {i}"} for i in range(35)]
d, why = ag.should_review(domain="coding_hard", executor_model="glm-cn/glm-5.3",
                          messages=stall, reply_has_tool_calls=True)
ok("stall checkpoint fires past 30 turns", d and why == "stall", why)

sk_t = R.session_key(tool_convo)
for _ in range(R.ADVISOR_MAX_REVIEWS):
    ag.note(sk_t, used=1)
d, why = ag.should_review(domain="trading_decision", executor_model="glm-cn/glm-5.3",
                          messages=tool_convo, reply_has_tool_calls=False)
ok("budget exhausts after max_reviews", not d and why == "budget_spent", why)

sk_f = "failsess"
ag2 = R.AdvisorGate(stats=st)
for _ in range(3):
    ag2.note(sk_f, failed=1)
print()
print("   -- review parsing --")
ok("APPROVE parsed", R.AdvisorGate.parse_review(body("APPROVE"))[0] == "approve")
v, plan = R.AdvisorGate.parse_review(body("REDO: you never ran the backtest. Run it first."))
ok("REDO parsed with plan", v == "redo" and "backtest" in plan, (v, plan))
ok("lowercase approve", R.AdvisorGate.parse_review(body("approve, looks good"))[0] == "approve")
ok("garbage => (None,None) => fail open",
   R.AdvisorGate.parse_review(body("hmm not sure")) == (None, None))
ok("empty => fail open", R.AdvisorGate.parse_review(body("")) == (None, None))

rr = ag.build_review_request(tool_convo, "I have verified everything, done.")
ok("review request uses Fable 5.1", rr["model"] == "cc/claude-fable-5-1")
ok("review request is non-streaming", rr["stream"] is False)
ok("review request carries the draft",
   "TURN AWAITING REVIEW" in rr["messages"][1]["content"])

print()
print("=" * 70)
print("7. TRANSCRIPT RENDERING")
print("=" * 70)
big = [{"role": "user", "content": "TASK_MARKER_START " + "a" * 100000},
       {"role": "assistant", "content": "b" * 100000},
       {"role": "user", "content": "c" * 100000 + " RECENT_MARKER_END"}]
t = R.render_transcript(big, max_chars=5000)
ok("middle-out truncation applied", "truncated" in t and len(t) < 6000, len(t))
ok("keeps task at start", "TASK_MARKER_START" in t)
ok("keeps recent at end", "RECENT_MARKER_END" in t)
ok("tool_calls rendered",
   "tool_calls" in R.render_transcript(tool_convo))

print()
print("=" * 70)
print("8. STATS TRACKER")
print("=" * 70)
s2 = R.StatsTracker(port=8898)
s2.record(domain="personal", tier="tier_quota_burn", model="glm-cn/glm-5.3",
          latency_ms=1200, ok=True, raw=body("hi", usage=(100, 50)), lane="cheap")
s2.record(domain="personal", tier="tier_quota_burn", model="glm-cn/glm-5.3",
          latency_ms=800, ok=True, raw=body("hi", usage=(60, 40)), lane="cheap")
s2.record(domain="trading_decision", tier="tier_primary_critical",
          model="cc/claude-opus-5", latency_ms=5000, ok=True,
          raw=body("x", usage=(500, 300)), lane="frontier")
s2.record_overhead("judge", R.JUDGE_MODEL, 300, True, body("v", usage=(30, 10)))
s2.record_overhead("advisor", R.ADVISOR_MODEL, 4000, True, body("APPROVE", usage=(2000, 20)))
snap = s2.snapshot()
ok("per-domain aggregation", snap["by_domain"]["personal"]["calls"] == 2)
ok("token accounting", snap["by_domain"]["personal"]["tokens_total"] == 250,
   snap["by_domain"]["personal"]["tokens_total"])
ok("avg latency", snap["by_domain"]["personal"]["avg_latency_ms"] == 1000.0,
   snap["by_domain"]["personal"]["avg_latency_ms"])
ok("judge overhead in OWN bucket",
   snap["routing_overhead"]["judge"][R.JUDGE_MODEL]["calls"] == 1)
ok("advisor overhead in OWN bucket (Fable 5.1)",
   snap["routing_overhead"]["advisor"]["cc/claude-fable-5-1"]["tokens_total"] == 2020)
ok("advisor overhead NOT mixed into served models",
   "cc/claude-fable-5-1" not in snap["by_model"])
ok("recent ring populated", len(snap["recent"]) == 3)
prom = s2.prometheus_text()
ok("prometheus exposes domain counters", 'domain="personal"' in prom)
ok("prometheus exposes overhead", 'kind="advisor"' in prom)

print()
print("=" * 70)
print("9. HELPERS")
print("=" * 70)
ok("extract text", R.extract_reply_text(body("hello world")) == "hello world")
ok("extract from block list",
   R.extract_reply_text(json.dumps({"choices": [{"message": {
       "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}]}}]}).encode()) == "abcd")
ok("detect tool calls",
   R.reply_has_tool_calls(body(None, tool_calls=[{"function": {"name": "f"}}])))
ok("no tool calls", not R.reply_has_tool_calls(body("plain")))
ok("count tool results", R.count_tool_results(tool_convo) == 8,
   R.count_tool_results(tool_convo))
ok("count assistant turns", R.count_assistant_turns(stall) == 35)

print()
print("=" * 70)
print(f"=== {P} passed, {F} failed (of {P+F}) ===")
print("=" * 70)
sys.exit(1 if F else 0)

print()
print("=" * 70)
print("10. LOCAL-FIRST JUDGE (sglang @ 127.0.0.1:11434)")
print("=" * 70)

import os as _os
import importlib
_os.environ["ROUTER_JUDGE_LOCAL_URL"] = "http://127.0.0.1:11434/v1"
_os.environ["ROUTER_JUDGE_LOCAL_MODEL"] = "qwen38-27b-abliterated"
import router_escalation as RR
importlib.reload(RR)
ok("local prepended to chain", RR.JUDGE_MODELS[0] == "qwen38-27b-abliterated", RR.JUDGE_MODELS)
ok("cloud chain kept AFTER local", RR.JUDGE_MODELS[1:] == ["glm-cn/glm-5.3-flash","glm-cn/glm-5.2","minimax/MiniMax-M3"], RR.JUDGE_MODELS[1:])
ok("JUDGE_USE_LOCAL flags true when env is set", RR.JUDGE_USE_LOCAL is True)

import urllib.request, json
jr = RR.EscalationTracker(None).build_judge_request(
    [{"role":"user","content":"what is 2+2"}, {"role":"assistant","content":"4"}], "4")
ok, body = RR.call_judge_local(jr)
ok("local judge reachable (live probe)", ok is True, body[:120] if not ok else "OK")
if ok:
    v = RR.parse_verdict(body)
    ok(f"local returns parseable verdict (got {v!r})", v in ("escalate", "decline"), v)
else:
    print(f"  unreachable reason: {body[:120]}")

del _os.environ["ROUTER_JUDGE_LOCAL_URL"]
del _os.environ["ROUTER_JUDGE_LOCAL_MODEL"]
importlib.reload(RR)
ok("with env unset, local NOT in chain", "qwen38-27b-abliterated" not in RR.JUDGE_MODELS)
ok("with env unset, JUDGE_USE_LOCAL is False", RR.JUDGE_USE_LOCAL is False)
