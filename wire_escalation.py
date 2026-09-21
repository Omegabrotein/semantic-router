#!/usr/bin/env python3
"""Idempotently wire router_escalation.py into all three router proxies.

Adds, per proxy:
  * import of the shared layer + a per-port StatsTracker/EscalationTracker/AdvisorGate
  * GET  /v1/stats     -> per-domain/per-model spend accounting
  * GET  /v1/models    -> honest capability flags (vision/tool_calling/reasoning)
  * GET  /metrics      -> now also emits Prometheus text when ?format=prom
  * POST path: escalation latch + judge, advisor gate, stats.record()

Re-running is safe: every insertion is guarded by a marker string.

    ~/.venvs/ibkr/bin/python3 wire_escalation.py [--check]
"""
import re
import shutil
import sys
import time

PROXIES = {
    "smart-router-proxy.py": 8898,
    "jimmy-router-proxy.py": 8901,
    "mila-router-proxy.py": 8902,
}

MARK = "ROUTER_ESCALATION_WIRED"

IMPORT_BLOCK = '''
# ─── {mark} (Sep 7 2026) ────────────────────────────────────────────
# Switchyard-derived layer: stats, escalation, advisor gate (Fable 5.1).
# See router_escalation.py for design notes and fail-open rules.
import router_escalation as _RE

_stats = _RE.StatsTracker(port={port})
_escalation = _RE.EscalationTracker(stats=_stats)
_advisor = _RE.AdvisorGate(stats=_stats)
# ─── end {mark} ─────────────────────────────────────────────────────
'''

GET_BLOCK = '''        if self.path.startswith('/v1/stats'):
            payload = json.dumps(_stats.snapshot(), indent=2).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(payload))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.startswith('/v1/models'):
            payload = json.dumps(
                _RE.build_models_payload(TIER_SYSTEM, DOMAIN_TO_TIER),
                indent=2).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(payload))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.startswith('/metrics') and 'format=prom' in self.path:
            payload = _stats.prometheus_text().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.send_header('Content-Length', len(payload))
            self.end_headers()
            self.wfile.write(payload)
            return
'''


def patch_imports(src, port):
    if MARK in src:
        return src, False
    anchor = "from router_tiers import TIER_SYSTEM, DOMAIN_TO_TIER"
    if anchor not in src:
        raise SystemExit(f"anchor not found: {anchor}")
    return src.replace(
        anchor, anchor + "\n" + IMPORT_BLOCK.format(mark=MARK, port=port), 1), True


def patch_get(src):
    if "/v1/stats" in src:
        return src, False
    anchor = "    def do_GET(self):\n"
    i = src.index(anchor) + len(anchor)
    # skip the docstring line
    j = src.index("\n", i) + 1
    return src[:j] + GET_BLOCK + src[j:], True


def patch_post(src):
    """Insert escalation + advisor + stats around the served response."""
    if "ESCALATION LATCH" in src:
        return src, False

    # 1. capture session key + header sid right after messages are parsed
    a1 = "            start_time = time.time()\n"
    if a1 not in src:
        raise SystemExit("anchor a1 not found")
    add1 = (a1 +
            "            _sid_hdr = (self.headers.get('x-session-id')\n"
            "                        or self.headers.get('x-switchyard-session-id'))\n"
            "            _skey = _RE.session_key(messages, _sid_hdr)\n"
            "            _escalated = False\n"
            "            _advised = None\n")
    src = src.replace(a1, add1, 1)

    # 2. latched sessions skip the cheap primary entirely
    a2 = ("            if _demoted:\n"
          "                primary_tier = \"tier_quota_burn\"\n"
          "            else:\n"
          "                primary_tier = DOMAIN_TO_TIER.get(domain, \"tier_primary_fast\")\n")
    if a2 not in src:
        raise SystemExit("anchor a2 not found")
    add2 = (a2 +
            "            # ESCALATION LATCH: a session the judge already ruled stuck\n"
            "            # routes straight to the strong tier with NO judge call.\n"
            "            # Only reachable for non-demoted cheap-lane domains, and\n"
            "            # only after ESCALATION_CONFIRMATIONS consecutive verdicts\n"
            "            # inside the hourly budget -- see router_escalation.py.\n"
            "            if (not _demoted and _escalation.is_latched(_skey)\n"
            "                    and _RE.should_escalate(domain, _demoted, primary_tier, 'cheap')):\n"
            "                primary_tier = 'tier_primary_complex'\n"
            "                _escalated = True\n"
            "                logger.info(f'ESCALATION LATCHED session -> {primary_tier} '\n"
            "                            f'(domain={domain})')\n")
    src = src.replace(a2, add2, 1)

    # 3. after a successful response: advisor gate, then judge, then stats
    a3 = ("            # Cache the response\n"
          "            # FABLE-REVIEW H1: streaming bodies are not cached (see above).\n"
          "            if not _is_stream:\n"
          "                cache.set_response(_conv_key, resp_body)\n")
    if a3 not in src:
        raise SystemExit("anchor a3 not found")
    add3 = ("            # ── ADVISOR GATE (cc/claude-fable-5-1) ──────────────────\n"
            "            # Reviews only TERMINAL turns in gated domains. On REDO the\n"
            "            # draft is discarded (client never sees it) and the executor\n"
            "            # is re-invoked with the reviewer's plan appended. Fail-open:\n"
            "            # any advisor failure releases the original turn unchanged.\n"
            "            if (not _is_stream) and response_success and resp_body:\n"
            "                try:\n"
            "                    _draft = _RE.extract_reply_text(resp_body)\n"
            "                    _has_tc = _RE.reply_has_tool_calls(resp_body)\n"
            "                    _do_rev, _why = _advisor.should_review(\n"
            "                        domain=domain, executor_model=model_used,\n"
            "                        messages=messages, reply_has_tool_calls=_has_tc)\n"
            "                    if _do_rev:\n"
            "                        _stats.bump('advisor_reviews')\n"
            "                        _rev_req = _advisor.build_review_request(messages, _draft)\n"
            "                        _t0 = time.time()\n"
            "                        _rok, _rbody, _ = self._try_model(\n"
            "                            _rev_req, _RE.ADVISOR_MODEL, retries=1)\n"
            "                        _rms = (time.time() - _t0) * 1000\n"
            "                        _stats.record_overhead('advisor', _RE.ADVISOR_MODEL,\n"
            "                                               _rms, _rok, _rbody if _rok else None)\n"
            "                        _verdict, _plan = (_advisor.parse_review(_rbody)\n"
            "                                           if _rok else (None, None))\n"
            "                        if _verdict == 'redo' and _plan:\n"
            "                            _advisor.note(_skey, used=1)\n"
            "                            _stats.bump('advisor_redo')\n"
            "                            logger.info(f'ADVISOR REDO ({_why}): {_plan[:120]}')\n"
            "                            _redo_req = dict(request_data)\n"
            "                            _redo_req['messages'] = list(messages) + [\n"
            "                                {'role': 'assistant', 'content': _draft},\n"
            "                                {'role': 'user', 'content': _RE._REDO_PREFIX + _plan},\n"
            "                            ]\n"
            "                            _t1 = time.time()\n"
            "                            _ok2, _body2, _ = self._try_model(_redo_req, model_used)\n"
            "                            if _ok2 and _body2:\n"
            "                                resp_body = _body2\n"
            "                                latency_ms += (time.time() - _t1) * 1000\n"
            "                                _advised = 'redo'\n"
            "                            else:\n"
            "                                _advised = 'redo_failed'\n"
            "                        elif _verdict == 'approve':\n"
            "                            _advisor.note(_skey, used=1)\n"
            "                            _stats.bump('advisor_approve')\n"
            "                            _advised = 'approve'\n"
            "                        else:\n"
            "                            _advisor.note(_skey, failed=1)\n"
            "                            _stats.bump('advisor_failed')\n"
            "                            _advised = 'fail_open'\n"
            "                except Exception as _ae:\n"
            "                    logger.warning(f'ADVISOR ERROR (fail-open): {_ae}')\n"
            "                    _advised = 'error_fail_open'\n"
            "\n"
            "            # ── ESCALATION JUDGE ────────────────────────────────────\n"
            "            # Judges the turn the cheap model ACTUALLY produced. Never\n"
            "            # blocks the response: the buffered reply is served either\n"
            "            # way; the verdict only moves the streak for NEXT turn.\n"
            "            if ((not _is_stream) and response_success and not _escalated\n"
            "                    and _RE.should_escalate(domain, _demoted, primary_tier, 'cheap')\n"
            "                    and _escalation.budget_available()):\n"
            "                try:\n"
            "                    _jreq = _escalation.build_judge_request(\n"
            "                        messages, _RE.extract_reply_text(resp_body))\n"
            "                    _t2 = time.time()\n"
            "                    _jok, _jbody, _ = self._try_model(\n"
            "                        _jreq, _RE.JUDGE_MODEL, retries=1)\n"
            "                    _jms = (time.time() - _t2) * 1000\n"
            "                    _stats.record_overhead('judge', _RE.JUDGE_MODEL,\n"
            "                                           _jms, _jok, _jbody if _jok else None)\n"
            "                    _stats.bump('escalation_judged')\n"
            "                    _v = _RE.parse_verdict(_jbody) if _jok else None\n"
            "                    _streak, _latched_now = _escalation.record_verdict(_skey, _v)\n"
            "                    if _latched_now:\n"
            "                        _stats.bump('escalation_latched')\n"
            "                        logger.info(\n"
            "                            f'ESCALATION LATCH domain={domain} streak={_streak} '\n"
            "                            f'-> next turn uses tier_primary_complex')\n"
            "                except Exception as _ee:\n"
            "                    logger.warning(f'JUDGE ERROR (fail-open): {_ee}')\n"
            "\n" + a3)
    src = src.replace(a3, add3, 1)

    # 4. stats.record + headers on the served path
    a4 = "            self.send_header('x-cache-hit-rate', f\"{cache.get_cache_hit_rate():.1f}%\")\n"
    if a4 not in src:
        raise SystemExit("anchor a4 not found")
    add4 = (a4 +
            "            self.send_header('x-escalated', '1' if _escalated else '0')\n"
            "            if _advised:\n"
            "                self.send_header('x-advisor', str(_advised))\n"
            "            try:\n"
            "                _stats.record(\n"
            "                    domain=domain, tier=primary_tier, model=model_used,\n"
            "                    latency_ms=latency_ms, ok=True,\n"
            "                    raw=None if _is_stream else resp_body,\n"
            "                    demoted=_demoted, lane=_lane,\n"
            "                    cache_type='classification' if class_hit else 'miss',\n"
            "                    escalated=_escalated, advised=_advised)\n"
            "            except Exception:\n"
            "                pass\n")
    src = src.replace(a4, add4, 1)
    return src, True


def main():
    check = "--check" in sys.argv
    ts = time.strftime("%Y%m%d-%H%M%S")
    for fname, port in PROXIES.items():
        src = open(fname).read()
        orig = src
        src, i1 = patch_imports(src, port)
        src, i2 = patch_get(src)
        src, i3 = patch_post(src)
        changed = src != orig
        print(f"{fname:26} port={port} imports={i1} get={i2} post={i3} changed={changed}")
        if changed and not check:
            shutil.copy2(fname, f".bak-{ts}-{fname}")
            open(fname, "w").write(src)
    print("\nDone." + (" (check only, nothing written)" if check else ""))


if __name__ == "__main__":
    main()
