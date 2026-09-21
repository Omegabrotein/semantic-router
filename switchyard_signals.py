#!/usr/bin/env python3
"""tool_signals: severity scoring + tool activity counts.

Ported from NVIDIA-NeMo/Switchyard @ d5cfe5b
  crates/libsy/src/algorithms/util/tool_signals.rs
to replace our C1/C2 static fallback chain with scored severity.

Three jobs in one module so callers reach only for one import:

  classify_text(text) -> (severity, [pattern_names, ...])
      Pattern-match a tool-result string. severity ∈ {0.0, 0.3, 0.7, 1.0}.
      Anchored substrings (e.g. "file does not exist") prefer FN over FP —
      if a substring is too eager it's narrowed here.

  signals_from_messages(messages, recent_window=3) -> dict
      Walk OpenAI / Anthropic chat messages, return ToolSignals-shaped dict:
        severity, repeated_failure, no_error_streak, edit_count, write_count,
        read_count, todowrite_count, new_count, recent_*, pure_bash_streak,
        tests_passed, tool_result_count, assistant_turn_count, turn_depth,
        compacted.
      Used by stage_score and any future escalator that wants a windowed view.

  tool_semantic_name(name, command=None) -> str
      Classify a tool call as Write / Edit / Read / Plan / New / Unknown,
      applying the Bash subcommand inference. Used only to inflate
      write/edit/read counters.

Severity ladder (kept identical to Switchyard so shadow metrics align):
  0.0  clean                      — no pattern matched
  0.3  soft (exit_nonzero)        — non-zero exit without recognisable trace
  0.7  hard                       — concrete exception, traceback, timeout, ...
  1.0  critical                   — OOM (unambiguous; forces capable tier)
"""

from __future__ import annotations

import re
from typing import Iterable


# ─── severity ladder ────────────────────────────────────────────────────────
SOFT = 0.3
HARD = 0.7
CRITICAL = 1.0
DEFAULT_RECENT_WINDOW = 3


# ─── stage-score constants (mirror Switchyard util/stage.rs) ───────────────
# Hosted here, NOT in switchyard_score, so callers that want only severity
# don't pull in tanh and the RLock.
#
# Each maxed signal contributes SIGNAL_UNIT to a "raw" weighted sum,
# divided by SCORE_GAIN so confidence spans (-1, +1) without clipping.
SCORE_GAIN = 5.0
SIGNAL_UNIT = 0.10
# Spinning/Exploring only fire once turn_depth is at least this large,
# because early no-write turns are normal exploration, not a stall.
STALL_MIN_TURN_DEPTH = 8

# Backwards-friendly alias — Switchyard calls it SEVERITY_CRITICAL.
SEVERITY_CRITICAL = CRITICAL


# ─── pattern table ──────────────────────────────────────────────────────────
# Lower-case substrings. (name, severity, [substrings]). An OR inside
# substrings. First hit per name fires once.
#
# Anchored strings ("file does not exist" / "\nok ") are load-bearing — they
# are trace-mined from real trajectories. Do not generalise them to bare
# substrings.
ERROR_PATTERNS: list[tuple[str, float, tuple[str, ...]]] = [
    ("oom", CRITICAL,
        ("out of memory", "memoryerror", "cannot allocate memory")),
    ("connection_refused", HARD,
        ("connection refused", "connectionrefusederror", "econnrefused")),
    ("traceback", HARD, ("traceback (most recent call last)",)),
    ("import_error", HARD,
        ("modulenotfounderror:", "importerror:", "no module named ")),
    ("cmd_not_found", HARD,
        ("command not found", "not found\n", "/usr/bin/env: ")),
    ("assertion", HARD, ("assertionerror",)),
    ("value_error", HARD, ("valueerror:",)),
    ("syntax_error", HARD, ("syntaxerror:",)),
    ("timeout", HARD,
        ("timed out", "timeouterror", "timeout expired", "deadline exceeded")),
    ("no_such_file", HARD,
        # Claude Code Read-tool miss. Anchored (not bare "does not exist")
        # because of `ls` output and prose false-positive risk.
        ("filenotfounderror:", "no such file or directory", "file does not exist")),
]


# Plain non-zero exit phrases. Matched in has_nonzero_exit_status — only with
# an integer in front so "0 failed" doesn't fire.
NONZERO_EXIT_PHRASES = (
    "exit code", "exit status", "exited with code", "exited with status",
)
NONZERO_EXIT_PHRASE_RE = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in NONZERO_EXIT_PHRASES) + r")\D{0,8}(\d+)"
)

# count-prefixed failure keywords — fired only with a non-zero integer.
NUMERIC_FAILURE_KEYWORDS = ("failed", "failure", "failures", "errors", "error")
NUMERIC_FAILURE_RE = re.compile(
    r"(?:^|\s)(\d+)\s+(?:" + "|".join(NUMERIC_FAILURE_KEYWORDS) + r")\b",
    re.MULTILINE,
)

# Literal failure tokens. Always-trip; safe because they cannot appear in a
# clean test run.
TEST_FAILURE_LITERAL = ("✗ ", "fatal:", "assertionerror", "error:")


# Test-pass phrases. Conservative — false positives here would clear a
# capable-hold too early. Each phrase is anchored so prose like "passed by
# your account" doesn't trip.
TEST_PASS_PHRASES = (
    " passed",        # "5 passed" / "tests passed"
    "passed in",      # "passed in 3.2s"
    "tests passed",
    "all tests passed",
    "test ok",
    "test result: ok",
    "passed.\n",
    "tests pass",
    "\nok ",           # go test; newline-anchored
    "✓ ",             # Unicode checkmark + space
)


# ─── Tool-call name recognition (built-ins first, Bash infer last) ──────────
WRITE_TOOL_NAMES = {"write", "create_file", "new_file", "write_file"}
EDIT_TOOL_NAMES = {
    "edit", "multiedit", "notebookedit", "str_replace",
    "str_replace_based_edit_tool", "apply_patch", "text_editor", "patch",
}
READ_TOOL_NAMES = {"read", "view", "read_file", "search_files"}
PLAN_TOOL_NAMES = {"todowrite", "todo_write", "todo", "update_plan"}
NEW_TOOL_NAMES = {"new", "create", "add"}        # forward activity demos
BASH_TOOL_NAMES = {
    "bash", "shell_command", "shell", "local_shell_call", "terminal",
    "exec_command",
}

# Bash subcommand inference. Order matters: writes override reads.
BASH_WRITE_PATTERNS = (
    "cat >", "cat >>", "echo >", "echo >>", "tee ", "printf >", "printf >>",
    "> /", ">> /", "<< 'eof'", "<<eof", "<<'eof'", "<< eof",
)
PYTHON_WRITE_PATTERNS = ("write_text(", "writelines(", ".write(")
JS_WRITE_PATTERNS = (
    "writefilesync(", "writefile(", "appendfilesync(", "appendfile(",
)
BASH_EDIT_PATTERNS = (
    "sed -i", "sed --in-place", "awk -i inplace", "awk 'inplace=1'",
    "patch ", "patch -p", "perl -i", "perl -p -i", "perl -pi",
)
BASH_READ_PATTERNS = (
    "cat /", "cat ./", "cat ../", "grep ", "ls ", "ls -", "find ",
    "head ", "tail ", "wc ", "diff ", "which ", "ps ", "df ", "du ",
    "stat ", "file ", "less ", "more ",
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _has_compiler_diagnostic(lower: str) -> bool:
    """error[E0001]: ... — Rust/Clippy diagnostic, OR 'compilation failed'."""
    for raw in lower.splitlines():
        line = raw.lstrip()
        if line in (
            "compilation failed", "error: compilation failed",
            "error: could not compile",
        ):
            return True
        if line.startswith("error: could not compile "):
            return True
        rest = line[len("error[e"):] if line.startswith("error[e") else None
        if rest is None:
            continue
        if "]: " not in rest:
            continue
        code, _ = rest.split("]: ", 1)
        if code and code.isascii() and code.isdigit():
            return True
    return False


def _has_runtime_exception(lower: str) -> bool:
    """Node/JS style runtime exception: typeerror: + stack line."""
    prefixes = (
        "typeerror:", "referenceerror:", "rangeerror:", "runtimeerror:",
        "keyerror:", "attributeerror:",
    )
    has_exc_line = any(
        line.lstrip().startswith(prefixes)
        for line in lower.splitlines()
    )
    return has_exc_line and ("\n    at " in lower or "\n  at " in lower)


def _has_runtime_panic(lower: str) -> bool:
    """Go panic: 'panic: runtime error:' + goroutine / signal signature."""
    has_panic = any(
        line.lstrip().startswith("panic: runtime error:")
        for line in lower.splitlines()
    )
    return has_panic and ("\ngoroutine " in lower or "[signal sig" in lower)


def _has_patch_failure(lower: str) -> bool:
    for raw in lower.splitlines():
        line = raw.lstrip()
        if line.startswith("error: patch failed:"):
            return True
        if line.startswith("patch failed:"):
            return True
        if ": patch does not apply" in line:
            return True
        if line.startswith("invalid context"):
            return True
    return False


def _has_nonzero_failure_count(lower: str) -> bool:
    """cargo/go style 'N failed' / 'N errors' with N > 0."""
    for m in NUMERIC_FAILURE_RE.finditer(lower):
        n = int(m.group(1))
        if n > 0:
            return True
    return False


def _has_nonzero_exit_status(lower: str) -> bool:
    return bool(NONZERO_EXIT_PHRASE_RE.search(lower))


def _has_test_failure_literal(lower: str) -> bool:
    return any(tok in lower for tok in TEST_FAILURE_LITERAL)


def _has_tests_passed(text: str) -> bool:
    return any(p in text for p in TEST_PASS_PHRASES)


# ─── Public API ─────────────────────────────────────────────────────────────

def classify_text(text: str) -> tuple[float, list[str]]:
    """Score one tool result.

    Returns (severity, [pattern_names]). Severity ladder mirrors Switchyard:
        0.0, 0.3 (exit_nonzero), 0.7 (hard), 1.0 (critical).

    Pure / total — no side effects, no I/O. Deterministic for a given input.
    """
    lower = text.lower()
    severity = 0.0
    patterns: list[str] = []

    # Phase 1: anchored substrings.
    for name, sev, subs in ERROR_PATTERNS:
        if any(sub in lower for sub in subs):
            patterns.append(name)
            severity = max(severity, sev)

    # Phase 2: counted failures (cargo/go style).
    if _has_nonzero_failure_count(lower):
        if "exit_nonzero" not in patterns:
            patterns.append("exit_nonzero")
        severity = max(severity, SOFT)

    # Phase 3: nonzero exit status, only counts when exit_nonzero isn't already.
    if "exit_nonzero" not in patterns and _has_nonzero_exit_status(lower):
        patterns.append("exit_nonzero")
        severity = max(severity, SOFT)

    # Phase 4: tier-specific detectors.
    for name, hit in (
        ("compile_error", _has_compiler_diagnostic(lower)),
        ("runtime_exception", _has_runtime_exception(lower)),
        ("runtime_panic", _has_runtime_panic(lower)),
        ("patch_error", _has_patch_failure(lower)),
    ):
        if hit and name not in patterns:
            patterns.append(name)
            severity = max(severity, HARD)

    # Phase 5: test pass / fail tokens (only emit both if both trip).
    if _has_test_failure_literal(lower):
        if "exit_nonzero" not in patterns:
            patterns.append("exit_nonzero")
        severity = max(severity, SOFT)
    if _has_tests_passed(text):
        # tests_passed is a positive signal, not an error — surfaced via
        # signals_from_messages(), not via classify_text().
        pass

    return severity, patterns


# ─── Tool-call classifier ───────────────────────────────────────────────────

def tool_semantic(name: str, command: str | None = None) -> str:
    """Return one of: write / edit / read / plan / new / unknown."""
    lower = (name or "").lower()
    if lower in WRITE_TOOL_NAMES:
        return "write"
    if lower in EDIT_TOOL_NAMES:
        return "edit"
    if lower in READ_TOOL_NAMES:
        return "read"
    if lower in PLAN_TOOL_NAMES:
        return "plan"
    if lower in NEW_TOOL_NAMES:
        return "new"
    if lower in BASH_TOOL_NAMES and command:
        c = command.lower()
        # write patterns trump edit trump read.
        if any(p in c for p in BASH_WRITE_PATTERNS) \
                or any(p in c for p in PYTHON_WRITE_PATTERNS):
            return "write"
        if any(p in c for p in BASH_EDIT_PATTERNS):
            return "edit"
        if any(p in c for p in BASH_READ_PATTERNS):
            return "read"
    return "unknown"


# ─── Walk messages (OpenAI + Anthropic shape) ───────────────────────────────

def _content_text(content) -> str:
    """Best-effort string-coerce for OpenAI str / list / Anthropic blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                t = block.get("type")
                if t in ("text", "output_text", "input_text"):
                    out.append(str(block.get("text") or ""))
                elif t == "tool_use":
                    inp = block.get("input") or {}
                    cmd = inp.get("command") or inp.get("cmd")
                    if isinstance(cmd, str):
                        out.append(cmd)
                elif t == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        out.append(inner)
                    elif isinstance(inner, list):
                        for x in inner:
                            if isinstance(x, dict):
                                out.append(str(x.get("text") or ""))
        return "\n".join(out)
    return str(content)


def _iter_tool_calls(message: dict):
    """Yield (tool_name, command_or_input_str, tool_use_id_or_None)."""
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("tool_use", "tool_call"):
                name = block.get("name") or block.get("function", {}).get("name", "")
                inp = block.get("input") or block.get("function", {}).get("arguments", "")
                if isinstance(inp, dict):
                    cmd = inp.get("command") or inp.get("cmd")
                    if isinstance(cmd, str):
                        yield name, cmd, block.get("id")
                        continue
                    yield name, json_dumps(inp), block.get("id")
                else:
                    yield name, str(inp or ""), block.get("id")
    if message.get("role") == "assistant":
        tc = message.get("tool_calls")
        if isinstance(tc, list):
            for t in tc:
                fn = (t or {}).get("function", {}) if isinstance(t, dict) else {}
                yield fn.get("name", ""), json_dumps(fn.get("arguments") or ""), (
                    t or {}).get("id") if isinstance(t, dict) else None


def _iter_tool_results(message: dict):
    """Yield strings from a tool-role result message (OpenAI + Anthropic)."""
    role = message.get("role")
    if role not in ("tool", "tool_result", "function"):
        return
    content = message.get("content")
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    yield block.get("text", "")
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        yield inner
                    elif isinstance(inner, list):
                        for x in inner:
                            if isinstance(x, dict):
                                yield str(x.get("text") or "")
            elif isinstance(block, str):
                yield block


def json_dumps(obj) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def signals_from_messages(
    messages: Iterable[dict],
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> dict:
    """Walk a conversation and return a ToolSignals-shaped dict.

    Same keys as Switchyard's ToolSignals; pure-Python values so the JSON in
    `/v1/stats` (or our shadow log) carries a stable contract.
    """
    msg_list = list(messages) if not isinstance(messages, list) else messages
    tool_result_count = 0
    assistant_turn_count = 0
    user_turn_count = 0
    tool_results_text: list[str] = []
    tool_calls_meta: list[tuple[str, str]] = []   # (name, command_or_input)
    plan_count = 0
    write_count = 0
    edit_count = 0
    read_count = 0
    new_count = 0
    todowrite_count = 0

    for m in msg_list:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            assistant_turn_count += 1
        elif role == "user":
            user_turn_count += 1

        for name, cmd, _tid in _iter_tool_calls(m):
            tool_calls_meta.append((name, cmd))
            kind = tool_semantic(name, cmd)
            if kind == "write":
                write_count += 1
            elif kind == "edit":
                edit_count += 1
            elif kind == "read":
                read_count += 1
            elif kind == "plan":
                plan_count += 1
                todowrite_count += 1
            elif kind == "new":
                new_count += 1

        for txt in _iter_tool_results(m):
            tool_result_count += 1
            tool_results_text.append(txt)

    # Window: last N tool results and tool calls.
    recent_results = tool_results_text[-recent_window:]
    recent_calls = tool_calls_meta[-recent_window:]
    recent_write = sum(1 for n, _ in recent_calls if tool_semantic(n, _) == "write")
    recent_edit = sum(1 for n, _ in recent_calls if tool_semantic(n, _) == "edit")
    recent_read = sum(1 for n, _ in recent_calls if tool_semantic(n, _) == "read")
    recent_todowrite = sum(1 for n, _ in recent_calls
                          if tool_semantic(n, _) == "plan")
    recent_new = sum(1 for n, _ in recent_calls
                     if tool_semantic(n, _) == "new")

    # Severity — max across the recent window; sticky across recovery turns.
    severities = [classify_text(t) for t in recent_results]
    severity = max((s for s, _ in severities), default=0.0)
    named = [p for _, ps in severities for p in ps]

    # repeated_failure: same hard-or-critical pattern ≥ 2 times in window.
    repeated_failure = False
    hard_terms = ("traceback", "timeout", "no_such_file", "import_error",
                  "cmd_not_found", "compile_error", "runtime_exception",
                  "runtime_panic", "patch_error", "assertion", "value_error",
                  "syntax_error", "connection_refused")
    hits = {p for p in named if p in hard_terms or p in ("oom",)}
    if severity >= HARD:
        for p in hits:
            if sum(1 for _, ps in severities if p in ps) >= 2:
                repeated_failure = True
                break

    # no_error_streak: consecutive clean tool results back from the latest.
    no_error_streak = 0
    for s, _ps in reversed(severities):
        if s <= SOFT and s == 0.0:
            no_error_streak += 1
        else:
            break

    # tests_passed: most recent clean result says tests passed.
    tests_passed = False
    if recent_results and severity == 0.0 and no_error_streak >= 1:
        tests_passed = _has_tests_passed(recent_results[-1])

    # pure_bash_streak: trailing unknown tool calls.
    pure_bash_streak = 0
    for n, cmd in reversed(recent_calls):
        if tool_semantic(n, cmd) == "unknown":
            pure_bash_streak += 1
        else:
            break

    # Compacted — when the conversation carries an explicit context-compaction
    # marker. We don't know the harness's exact marker; the proxy can be told
    # via the request (out of scope for the helper).
    compacted = False

    return {
        "severity": float(severity),
        "repeated_failure": repeated_failure,
        "no_error_streak": no_error_streak,
        "edit_count": edit_count,
        "write_count": write_count,
        "read_count": read_count,
        "todowrite_count": todowrite_count,
        "new_count": new_count,
        "recent_edit_count": recent_edit,
        "recent_write_count": recent_write,
        "recent_read_count": recent_read,
        "recent_todowrite_count": recent_todowrite,
        "recent_new_count": recent_new,
        "pure_bash_streak": pure_bash_streak,
        "tests_passed": tests_passed,
        "tool_result_count": tool_result_count,
        "assistant_turn_count": assistant_turn_count,
        "turn_depth": len(msg_list),
        "compacted": compacted,
        "user_turn_count": user_turn_count,
        "named_patterns": list(hits),
    }


__all__ = [
    "SOFT", "HARD", "CRITICAL", "DEFAULT_RECENT_WINDOW",
    "SCORE_GAIN", "SIGNAL_UNIT", "STALL_MIN_TURN_DEPTH",
    "ERROR_PATTERNS",
    "classify_text",
    "tool_semantic",
    "signals_from_messages",
]
