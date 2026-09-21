#!/usr/bin/env python3
"""
Anthropic /v1/messages -> OpenAI /v1/chat/completions shim.

Purpose: let Claude Code (which speaks the Anthropic Messages API) route through
the semantic auto-router on 127.0.0.1:8898, which only speaks chat-completions.

Design constraint: smart-router-proxy.py (8898) is LIVE for default+youtube
profiles and is shared infrastructure. This shim is a SEPARATE process and does
not modify it. Traffic flow:

    claude CLI --(Anthropic Messages)--> :8903 shim --(chat/completions)--> :8898 semantic router --> :20128 9router --> upstream

Listens on 127.0.0.1:8903. Forwards the client's Authorization header verbatim,
exactly as the proxies at 8901/8902 do.
"""
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

UPSTREAM = "http://127.0.0.1:8898/v1/chat/completions"
UPSTREAM_MODELS = "http://127.0.0.1:8898/v1/models"
LISTEN = ("127.0.0.1", 8903)
UPSTREAM_TIMEOUT = 600

# The Claude CLI validates model names client-side and rejects anything that is
# not a real Anthropic id ("unrecognized_model"), so it must be configured with
# e.g. claude-sonnet-4-5-*. The semantic router, conversely, has its own tier
# table and 503s "All models unavailable" on those Anthropic ids. We therefore
# rewrite the model to ROUTER_MODEL on the way out: the router picks the tier
# semantically from the prompt, which is the whole point of routing here.
# Set SHIM_PASSTHROUGH_MODELS=1 to forward the client's model name instead.
ROUTER_MODEL = os.environ.get("SHIM_ROUTER_MODEL", "auto")
PASSTHROUGH = os.environ.get("SHIM_PASSTHROUGH_MODELS") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shim] %(levelname)s %(message)s",
)
logger = logging.getLogger("anthropic-shim")


# --------------------------------------------------------------------------
# Anthropic -> OpenAI request translation
# --------------------------------------------------------------------------

def _text_from_blocks(content):
    """Flatten an Anthropic content value to plain text (best effort)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for b in content:
        if isinstance(b, str):
            out.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "".join(out)


def anthropic_to_openai(body):
    """Translate an Anthropic Messages request dict into a chat-completions dict."""
    msgs = []

    system = body.get("system")
    if system:
        sys_text = _text_from_blocks(system)
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})

    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        text_parts = []
        tool_calls = []
        tool_results = []

        for blk in content:
            if not isinstance(blk, dict):
                if isinstance(blk, str):
                    text_parts.append(blk)
                continue
            btype = blk.get("type")
            if btype == "text":
                text_parts.append(blk.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": blk.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": blk.get("name", ""),
                        "arguments": json.dumps(blk.get("input", {})),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": blk.get("tool_use_id", ""),
                    "content": _text_from_blocks(blk.get("content", "")),
                })
            elif btype == "image":
                # The router advertises vision:false; drop image bytes rather
                # than shipping a payload the upstream will reject.
                text_parts.append("[image omitted by anthropic-shim]")

        # tool results must precede the assistant/user turn they answer
        msgs.extend(tool_results)

        if text_parts or tool_calls:
            om = {"role": role, "content": "".join(text_parts)}
            if tool_calls:
                om["tool_calls"] = tool_calls
                if not om["content"]:
                    om["content"] = None
            msgs.append(om)

    out = {
        "model": body.get("model", "auto"),
        "messages": msgs,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]

    tools = body.get("tools")
    if tools:
        oa_tools = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            oa_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object"}),
                },
            })
        if oa_tools:
            out["tools"] = oa_tools

    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "auto":
            out["tool_choice"] = "auto"
        elif ctype == "any":
            out["tool_choice"] = "required"
        elif ctype == "tool" and choice.get("name"):
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": choice["name"]},
            }

    return out


# --------------------------------------------------------------------------
# OpenAI -> Anthropic response translation
# --------------------------------------------------------------------------

_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def openai_to_anthropic(oa, req_model):
    choices = oa.get("choices") or [{}]
    msg = (choices[0] or {}).get("message") or {}

    blocks = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})

    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args)
        except Exception:
            parsed = {"__raw": raw_args}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": fn.get("name", ""),
            "input": parsed,
        })

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    usage = oa.get("usage") or {}
    finish = (choices[0] or {}).get("finish_reason") or "stop"

    return {
        "id": oa.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": oa.get("model") or req_model,
        "content": blocks,
        "stop_reason": _STOP_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def anthropic_sse(resp):
    """Render a complete Anthropic message as a well-formed SSE event stream.

    The upstream router collapses SSE into a single JSON object, so there is no
    incremental stream to relay. Claude Code accepts a single-chunk stream as
    long as the event sequence is complete and correctly ordered.
    """
    def ev(name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    blocks = resp["content"]
    shell = dict(resp)
    shell["content"] = []
    shell["stop_reason"] = None
    shell["usage"] = {"input_tokens": resp["usage"]["input_tokens"], "output_tokens": 0}

    yield ev("message_start", {"type": "message_start", "message": shell})

    for i, blk in enumerate(blocks):
        if blk["type"] == "text":
            yield ev("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""},
            })
            if blk.get("text"):
                yield ev("content_block_delta", {
                    "type": "content_block_delta", "index": i,
                    "delta": {"type": "text_delta", "text": blk["text"]},
                })
        elif blk["type"] == "tool_use":
            yield ev("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {
                    "type": "tool_use", "id": blk["id"],
                    "name": blk["name"], "input": {},
                },
            })
            yield ev("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(blk.get("input", {})),
                },
            })
        yield ev("content_block_stop", {"type": "content_block_stop", "index": i})

    yield ev("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": resp["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": resp["usage"]["output_tokens"]},
    })
    yield ev("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

class ShimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "anthropic-shim/1.0"

    def log_message(self, fmt, *args):  # quieter default access log
        return

    def _upstream_auth(self):
        """Build the Authorization header for the router.

        Anthropic clients (including Claude Code) send credentials as
        `x-api-key`, NOT `Authorization: Bearer`. The router at :8898 forwards
        whatever Authorization it receives, so an x-api-key-only request arrives
        upstream with NO credential and comes back 503 "All models unavailable".
        Accept either form and always emit a Bearer header.
        """
        auth = self.headers.get("Authorization", "")
        if auth:
            return auth
        api_key = self.headers.get("x-api-key", "")
        if api_key:
            return f"Bearer {api_key}"
        return ""

    def _send(self, code, payload, ctype="application/json"):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload).encode()
        elif isinstance(payload, str):
            payload = payload.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _err(self, code, msg):
        logger.error("%s -> %s", self.path, msg)
        self._send(code, {"type": "error", "error": {"type": "api_error", "message": msg}})

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok", "upstream": UPSTREAM})
            return
        if self.path.startswith("/v1/models"):
            try:
                req = urllib.request.Request(UPSTREAM_MODELS, headers={
                    "Authorization": self._upstream_auth()})
                with urllib.request.urlopen(req, timeout=30) as r:
                    self._send(200, r.read())
            except Exception as e:
                self._err(502, f"upstream models failed: {e}")
            return
        self._err(404, "not found")

    def do_POST(self):
        if not self.path.startswith("/v1/messages"):
            self._err(404, f"unsupported path {self.path}")
            return

        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._err(400, f"bad request body: {e}")
            return

        # /v1/messages/count_tokens — cheap local estimate, never hits upstream
        if self.path.startswith("/v1/messages/count_tokens"):
            chars = len(json.dumps(body.get("messages", [])))
            chars += len(json.dumps(body.get("system", "")))
            self._send(200, {"input_tokens": max(1, chars // 4)})
            return

        wants_stream = bool(body.get("stream"))
        req_model = body.get("model", "auto")
        oa_req = anthropic_to_openai(body)
        oa_req["stream"] = False  # the router collapses SSE anyway
        if not PASSTHROUGH:
            oa_req["model"] = ROUTER_MODEL

        t0 = time.time()
        try:
            req = urllib.request.Request(
                UPSTREAM,
                data=json.dumps(oa_req).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._upstream_auth(),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            logger.error(
                "upstream %s | model=%s->%s msgs=%d tools=%d payload=%dB | %s",
                e.code, req_model, oa_req.get("model"),
                len(oa_req.get("messages", [])), len(oa_req.get("tools", [])),
                len(json.dumps(oa_req)), detail,
            )
            self._err(e.code, f"upstream {e.code}: {detail}")
            return
        except Exception as e:
            self._err(502, f"upstream unreachable: {e}")
            return

        try:
            oa = json.loads(raw)
        except Exception:
            # defensive: upstream returned SSE despite stream:false
            collected = []
            for line in raw.decode(errors="replace").split("\n"):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if not p or p == "[DONE]":
                    continue
                try:
                    ev = json.loads(p)
                except Exception:
                    continue
                d = ((ev.get("choices") or [{}])[0] or {}).get("delta") or {}
                if isinstance(d.get("content"), str):
                    collected.append(d["content"])
            if not collected:
                self._err(502, "unparseable upstream response")
                return
            oa = {"choices": [{"message": {"role": "assistant",
                                           "content": "".join(collected)},
                               "finish_reason": "stop"}]}

        resp = openai_to_anthropic(oa, req_model)
        logger.info("%s -> %s  %.1fs  in=%d out=%d",
                    req_model, resp["model"], time.time() - t0,
                    resp["usage"]["input_tokens"], resp["usage"]["output_tokens"])

        if not wants_stream:
            self._send(200, resp)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in anthropic_sse(resp):
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("client disconnected mid-stream")


def main():
    server = ThreadingHTTPServer(LISTEN, ShimHandler)
    server.daemon_threads = True
    logger.info("Anthropic Messages shim on http://%s:%d", *LISTEN)
    logger.info("  POST /v1/messages  -> %s", UPSTREAM)
    logger.info("  POST /v1/messages/count_tokens (local estimate)")
    logger.info("  GET  /v1/models, /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
