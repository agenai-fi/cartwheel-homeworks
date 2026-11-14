"""Read an openai-agents SQLiteSession store: sessions, turns, tool calls.

This module knows about the SDK and the model provider. It knows nothing about
any particular application -- no roles, no record schema, no homework. Give it
a session database and it gives you turns.

This is a library, not a command. It has no CLI: the application that knows
what a session means owns that. `turn_records` yields one record per turn --
the shape any application-specific tool consumes:

    {"session_id": str, "turn": int, "turns_in_session": int,
     "status": "final" | <reason>, "request": str, "response": str,
     "context": [str],   # the requests that preceded this one in the session
     "tool_calls": [{"name": str, "arguments": {...}, "result": ...}]}

Facts about the store, each verified against the SDK source and a real database:

* One SQLiteSession per process; the process loops on input(), so a session
  holds one turn per request the user typed. Turns split at each user message.
* Writes are committed once per model loop iteration, not once per turn. A turn
  therefore arrives in several batches seconds apart, and text can be persisted
  long before the answer is.
* A tool's result is stored as ``str(result)`` -- a Python repr with single
  quotes and True/None, which ``json.loads`` cannot read.
* The Responses API stamps assistant messages
  ``phase: Literal["commentary", "final_answer"]``.
* ``agent_messages.id`` is AUTOINCREMENT, so a single integer watermark is
  enough to follow the store, and ids are never reused.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sqlite3
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

WIDTH_CAP = 100
POLL_SECONDS = 0.5


# --- the store --------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open read-write but never create: a reader that creates an empty store
    leaves an artifact that makes the next run fail differently."""
    conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_sessions(db_path: Path, since_id: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Messages grouped by session, in insertion order.

    ``since_id`` restricts which SESSIONS are returned, not which messages: a
    turn spans many rows, so a session touched by a new row is returned whole.
    """
    conn = _connect(db_path)
    try:
        touched = [
            row["session_id"]
            for row in conn.execute(
                "SELECT session_id, MIN(id) AS first FROM agent_messages "
                "WHERE id > ? GROUP BY session_id ORDER BY first",
                (since_id,),
            )
        ]
        sessions: dict[str, list[dict[str, Any]]] = {}
        for session_id in touched:
            sessions[session_id] = [
                json.loads(row["message_data"])
                for row in conn.execute(
                    "SELECT message_data FROM agent_messages "
                    "WHERE session_id = ? ORDER BY id",
                    (session_id,),
                )
            ]
        return sessions
    finally:
        conn.close()


def current_watermark(db_path: Path) -> int:
    """The highest message id in the store right now, or 0 if unreadable."""
    if not db_path.exists():
        return 0
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return 0
    try:
        return conn.execute("SELECT MAX(id) FROM agent_messages").fetchone()[0] or 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# --- turns ------------------------------------------------------------------


def split_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Every turn of a session, sliced at each user message."""
    starts = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    bounds = starts + [len(messages)]
    return [messages[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def _parse_arguments(raw: Any) -> Any:
    """Tool arguments are stored as a JSON string."""
    if not isinstance(raw, str):
        return raw if raw is not None else {}
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_result(raw: Any) -> Any:
    """Tool results are stored as a Python repr, not JSON."""
    if not isinstance(raw, str):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def parse_turn(turn: list[dict[str, Any]]) -> dict[str, Any]:
    """One turn's request, its tool calls paired by call_id, and its reply."""
    responses: list[str] = []
    by_call_id: dict[Any, dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []

    for message in turn[1:]:
        kind = message.get("type")
        if kind == "function_call":
            call = {
                "name": message.get("name"),
                "arguments": _parse_arguments(message.get("arguments")),
                "result": None,
            }
            call_id = message.get("call_id")
            # Only index a usable call_id: indexing None would let a second
            # unidentified call absorb the first one's result.
            if call_id:
                by_call_id[call_id] = call
            tool_calls.append(call)
        elif kind == "function_call_output":
            call = by_call_id.get(message.get("call_id"))
            if call is not None:
                call["result"] = _parse_result(message.get("output"))
        elif message.get("role") == "assistant":
            text = "\n".join(
                part.get("text") or part.get("refusal") or ""
                for part in message.get("content") or []
                if isinstance(part, dict) and (part.get("text") or part.get("refusal"))
            )
            if text:
                responses.append(text)

    return {
        "request": turn[0]["content"] if turn else "",
        "tool_calls": tool_calls,
        "response": responses[-1] if responses else "",
    }


def turn_status(turn: list[dict[str, Any]]) -> str:
    """ "final", or the reason this turn is not finished.

    One definition, used by every caller. Text is NOT proof of completion: the
    SDK commits per model loop iteration, so a model that speaks alongside its
    tool call persists a preamble a whole inference before the answer.

    * A provider that stamps ``phase`` is authoritative -- anything other than
      "final_answer" means the closing message has not arrived.
    * A provider that never stamps it is judged by shape: the turn is over when
      no tool call is missing its result and the last stored message is
      assistant text. A batch ending in a function_call_output is mid-flight.
    """
    if not turn:
        return "no request"
    assistants = [
        message
        for message in turn
        if message.get("role") == "assistant" and message.get("content")
    ]
    parsed = parse_turn(turn)
    # Checked before the phase field: a turn with an unanswered call is not
    # finished however the model stamped its message.
    if any(call["result"] is None for call in parsed["tool_calls"]):
        return "a tool call has no stored result: the run ended mid-turn"
    if any(message.get("phase") == "final_answer" for message in assistants):
        return "final"
    if any("phase" in message for message in assistants):
        return "the closing message has not arrived yet"

    if not parsed["response"]:
        return "the run ended before the agent replied"
    last = turn[-1]
    if last.get("role") == "assistant" and last.get("content"):
        return "final"
    return "the closing message has not arrived yet"


def turn_records(db_path: Path, since_id: int = 0) -> Iterator[dict[str, Any]]:
    """Every turn in the store, as the exported record shape."""
    for session_id, messages in load_sessions(db_path, since_id).items():
        turns = split_turns(messages)
        parsed = [parse_turn(slice_) for slice_ in turns]
        for index, slice_ in enumerate(turns, start=1):
            yield {
                "session_id": session_id,
                "turn": index,
                "turns_in_session": len(turns),
                "status": turn_status(slice_),
                # A later turn read alone is unintelligible -- "and 3980?" means
                # nothing without what preceded it. Carry the antecedents.
                "context": [earlier["request"] for earlier in parsed[: index - 1]],
                **parsed[index - 1],
            }


# --- labels -----------------------------------------------------------------


def label(session_id: str, pattern: re.Pattern[str] | None) -> str:
    """A human label for a session.

    The store has no notion of who was talking; that lives in whatever naming
    convention the application gave its sessions. Supply it as a regex with
    named groups and its captures become the label -- so this module can show
    "role=shopper user=1" without knowing what a shopper is.
    """
    if pattern is None:
        return session_id
    match = pattern.search(session_id)
    if match is None or not match.groupdict():
        return session_id
    fields = " ".join(
        f"{name}={value}" for name, value in match.groupdict().items() if value
    )
    return f"{session_id}  {fields}" if fields else session_id


# --- rendering --------------------------------------------------------------


def _terminal_width(default: int = 80) -> int:
    return min(shutil.get_terminal_size((default, 24)).columns, WIDTH_CAP)


def _scalar(value: Any) -> str:
    """JSON spelling, so the view matches the stored data."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _wrap(text: str, indent: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                paragraph, width=width, initial_indent=indent, subsequent_indent=indent
            )
        )
    return lines


def _render(value: Any, indent: str, width: int) -> list[str]:
    """Any JSON value as an indented tree.

    Deliberately generic: no tool knows how to render itself, so a new tool or a
    changed result shape stays readable without touching this code.
    """
    out: list[str] = []
    if isinstance(value, dict):
        if not value:
            return [f"{indent}(empty)"]
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                out.append(f"{indent}{key}:")
                out.extend(_render(item, indent + "  ", width))
                continue
            rendered = (
                _scalar(item) if not isinstance(item, (dict, list)) else "(empty)"
            )
            line = f"{indent}{key}: {rendered}"
            if len(line) <= width and "\n" not in rendered:
                out.append(line)
            else:
                out.append(f"{indent}{key}:")
                out.extend(_wrap(rendered, indent + "  ", width))
    elif isinstance(value, list):
        if not value:
            return [f"{indent}(empty)"]
        for position, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                out.append(f"{indent}[{position}]")
                out.extend(_render(item, indent + "  ", width))
            else:
                out.extend(_wrap(f"[{position}] {_scalar(item)}", indent, width))
    else:
        out.extend(_wrap(_scalar(value), indent, width))
    return out


def render_turn(
    record: dict[str, Any], width: int | None = None, heading: str | None = None
) -> str:
    """One turn, laid out for reading."""
    width = width or _terminal_width()
    out: list[str] = []
    if heading is not None:
        out.extend([heading + "-" * max(0, width - len(heading)), ""])
    if record.get("status") != "final":
        out.extend(_wrap(f"! {record['status']}", "   ", width))
        out.append("")

    out.append("REQUEST")
    out.extend(_wrap(record.get("request", ""), "  ", width))
    out.append("")

    calls = record.get("tool_calls") or []
    if not calls:
        out.extend(["TOOLS", "  (none called)", ""])
    for position, call in enumerate(calls, start=1):
        out.append(f"TOOL {position}  {call.get('name')}")
        out.append("  arguments:")
        out.extend(_render(call.get("arguments"), "    ", width))
        out.append("  result:")
        if call.get("result") is None:
            out.append("    (none stored -- the run ended before the tool returned)")
        else:
            out.extend(_render(call.get("result"), "    ", width))
        out.append("")

    out.append("RESPONSE")
    response = record.get("response", "")
    if response:
        out.extend(_wrap(response, "  ", width))
    else:
        out.append("  (none -- the run ended before the agent replied)")
    out.append("")
    return "\n".join(out)


def heading_for(
    record: dict[str, Any], pattern: re.Pattern[str] | None, position: int
) -> str:
    who = label(record["session_id"], pattern)
    total = record.get("turns_in_session", 1)
    turn = f"turn {record['turn']} of {total}" if total > 1 else "1 turn"
    return f"-- {position}. {who}  ({turn}) "


# --- following --------------------------------------------------------------


def tail_step(
    db_path: Path, watermark: int, seen: dict[Any, Any]
) -> tuple[list[dict[str, Any]], int]:
    """One poll: newly announced requests and newly finished turns.

    ``seen`` maps (session_id, turn) to "announced" or "emitted" and is updated
    in place, so the caller holds no other state. Because completion is
    deterministic, a turn always finishes on a poll touched by its closing
    batch -- no timing, no re-scanning of quiet sessions.
    """
    events: list[dict[str, Any]] = []
    # Connecting to a missing path would create an empty database.
    if not db_path.exists():
        return events, watermark
    try:
        highest = current_watermark(db_path)
        identity = db_path.stat().st_ino
        # Comparing ids alone only catches a SMALLER replacement store; a
        # restored backup with more rows would silently skip its own history.
        if seen.get("#inode") not in (None, identity) or highest < watermark:
            seen.clear()
            watermark = 0
            events.append({"kind": "reset"})
        seen["#inode"] = identity

        for record in turn_records(db_path, watermark):
            key = (record["session_id"], record["turn"])
            if key not in seen:
                seen[key] = "announced"
                events.append({"kind": "request", **record})
            if record["status"] == "final" and seen[key] != "emitted":
                seen[key] = "emitted"
                events.append({"kind": "turn", **record})
    except (sqlite3.Error, OSError):
        # A store being recreated exists for a moment without its tables.
        return events, watermark
    return events, max(highest, watermark)


def render_event(
    event: dict[str, Any], pattern: re.Pattern[str] | None, width: int
) -> str:
    if event["kind"] == "reset":
        return "! the store was replaced; following it from the beginning\n"
    tag = f"[{label(event['session_id'], pattern)} turn {event['turn']}]"
    if event["kind"] == "request":
        request = " ".join(str(event["request"]).split())
        room = max(20, width - len(tag) - 5)
        if len(request) > room:
            request = request[: room - 1] + "…"
        return f"{tag} {request}"
    return "\n".join(
        [f"-- {tag} " + "-" * max(0, width - len(tag) - 4), ""]
        + render_turn(event, width).split("\n")
    )


# --- cli --------------------------------------------------------------------
