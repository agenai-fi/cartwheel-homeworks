"""Turn Cartwheel CLI conversations into the Homework 1 record file.

Everything homework- and Cartwheel-specific lives here: the ten hw1.md fields,
the cli-{role}-{user}-{ts} naming convention, the store_id lookup, SPEC.md ids
and the >=10 gate. Reading the session store is scripts/sessionreader.py's job
and it knows none of this.

    ./session-reader dump hw1-session.jsonl    # write / refresh the records
    ./session-reader tail                      # read the conversations
    ./session-reader tail -f                   # follow new ones live

`dump` rebuilds the eight mechanical fields from the store on every run and
carries your four judgment fields through untouched, so a record can never go
stale and a hand-written verdict is never lost. It then reports what is left:
unfilled judgments, sessions whose turns are not all present, and whether ten
conversations have been recorded. With no store reachable it validates the
named file instead of writing it, so a fresh clone can still be judged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent import db
from scripts import sessionreader

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "hw1-session.jsonl"
SPEC_PATH = REPO_ROOT / "SPEC.md"
SESSIONS_DB = REPO_ROOT / ".sessions.db"

# The ten fields hw1.md:139-148 requires, in the order it lists them, plus the
# two keys that identify which exported turn a record came from.
HOMEWORK_FIELDS = (
    "role",
    "user_id",
    "store_id",
    "request",
    "tool_calls",
    "response",
    "expected",
    "requirement",
    "met_requirement",
    "problem_source",
)
# Not homework fields: two identify which exported turn a record came from,
# two make a record judgeable and auditable on its own.
EXTRA_FIELDS = ("session_id", "turn", "context", "turns_in_session")
JUDGMENT_FIELDS = ("expected", "requirement", "met_requirement", "problem_source")
PROBLEM_SOURCES = ("prompt", "tool", "specification")
ROLES = ("shopper", "merchant", "support")
MINIMUM_CONVERSATIONS = 10

# agent/cli.py names its sessions f"cli-{role}-{user_id}-{timestamp}".
SESSION_ID_RE = re.compile(rf"^cli-({'|'.join(ROLES)})-(\d+)-\d+$")
# The same convention handed to sessionreader, which knows none of it.
LABEL_PATTERN = rf"^cli-(?P<role>{'|'.join(ROLES)})-(?P<user>\d+)-"

# SPEC.md declares requirements in bold prose and as the leading column of the
# tools table.
SPEC_BOLD_RE = re.compile(r"\*\*([A-Z]+-\d+)\.?\*\*")
SPEC_TABLE_RE = re.compile(r"^\|\s*([A-Z]+-\d+)\s*\|", re.MULTILINE)


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def store_id_for(role: str, user_id: int, warn: Callable[[str], None]) -> int | None:
    """Merchants carry a store; other roles do not. Mirrors cli.resolve_auth.

    This is the one thing that cannot come from the session store: it is a
    lookup in Cartwheel's world database.
    """
    if role != "merchant":
        return None
    try:
        conn = db.connect()
    except FileNotFoundError:
        warn(f"user {user_id}: world database missing, store_id left null")
        return None
    try:
        user = db.get_user(conn, user_id)
    finally:
        conn.close()
    if user is None:
        warn(f"user {user_id}: not in the users table, store_id left null")
        return None
    return user.store_id


def to_record(
    turn: dict[str, Any], warn: Callable[[str], None] = _stderr
) -> dict[str, Any] | None:
    """One exported turn as a homework record, or None if it is not evidence."""
    session_id = turn["session_id"]
    match = SESSION_ID_RE.match(session_id)
    if match is None:
        warn(f"{session_id}: not an agent.cli session, skipped")
        return None
    if turn["status"] != "final":
        warn(f"{session_id} turn {turn['turn']}: {turn['status']}; skipped")
        return None

    role, user_id = match.group(1), int(match.group(2))
    return {
        "session_id": session_id,
        "turn": turn["turn"],
        "role": role,
        "user_id": user_id,
        "store_id": store_id_for(role, user_id, warn),
        "request": turn["request"],
        # A later turn read alone is unintelligible: "4455" is not a question.
        "context": turn.get("context") or [],
        "turns_in_session": turn.get("turns_in_session", turn["turn"]),
        "tool_calls": turn["tool_calls"],
        "response": turn["response"],
        "expected": "",
        "requirement": None,
        "met_requirement": None,
        "problem_source": None,
    }


def record_key(record: dict[str, Any]) -> tuple[Any, Any]:
    return (record.get("session_id"), record.get("turn"))


def refresh(
    existing: list[dict[str, Any]], fresh: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rebuild the transcript, keep the judgment.

    The eight mechanical fields are always re-derived, so a reader improvement
    reaches every record and no record can go stale. The four judgment fields
    are yours and are carried through untouched -- note the test is against
    "" and None, not truthiness, or a recorded `met_requirement: false` would
    be erased.
    """
    unkeyed = [i for i, r in enumerate(existing, 1) if None in record_key(r)]
    if unkeyed:
        raise SystemExit(
            f"records {unkeyed} carry no session_id/turn, so they cannot be matched "
            "against the store. Restore those keys, or move the file aside and dump "
            "again."
        )
    by_key = {record_key(record): record for record in fresh}
    merged: list[dict[str, Any]] = []
    for old_record in existing:
        replacement = by_key.pop(record_key(old_record), None)
        if replacement is None:
            # The store no longer holds this turn. Keep the record rather than
            # silently drop work; check() reports the gap.
            merged.append(old_record)
            continue
        combined = dict(replacement)
        for field in JUDGMENT_FIELDS:
            if old_record.get(field) not in ("", None):
                combined[field] = old_record[field]
        merged.append(combined)
    merged.extend(by_key.values())
    return merged


def spec_requirement_ids(spec_path: Path = SPEC_PATH) -> set[str]:
    text = spec_path.read_text()
    return set(SPEC_BOLD_RE.findall(text)) | set(SPEC_TABLE_RE.findall(text))


def validate(
    records: list[dict[str, Any]], spec_ids: set[str] | None = None
) -> list[str]:
    """Every problem found. Empty means the file is ready to submit."""
    if spec_ids is None:
        spec_ids = spec_requirement_ids()
    problems: list[str] = []

    seen: set[tuple[Any, Any]] = set()
    for index, record in enumerate(records, start=1):
        where = f"record {index}"
        key = record_key(record)
        if key[0] is not None:
            if key in seen:
                problems.append(f"{where}: duplicate {key}")
            seen.add(key)

        missing = [field for field in HOMEWORK_FIELDS if field not in record]
        if missing:
            problems.extend(f"{where}: missing '{field}'" for field in missing)
            continue

        if record["role"] not in ROLES:
            problems.append(f"{where}: role {record['role']!r} is not one of {ROLES}")
        if not isinstance(record["user_id"], int):
            problems.append(f"{where}: user_id must be an integer")
        store = record["store_id"]
        if record["role"] == "merchant":
            if not isinstance(store, int):
                problems.append(f"{where}: a merchant record needs a store_id")
        elif store is not None:
            problems.append(f"{where}: store_id must be null for {record['role']}")
        if not str(record["request"]).strip():
            problems.append(f"{where}: request is empty")
        if not str(record["response"]).strip():
            problems.append(f"{where}: response is empty")

        calls = record["tool_calls"]
        if not isinstance(calls, list):
            problems.append(f"{where}: tool_calls must be a list")
        else:
            for position, call in enumerate(calls, start=1):
                if not isinstance(call, dict) or not {
                    "name",
                    "arguments",
                    "result",
                } <= set(call):
                    problems.append(
                        f"{where}: tool_calls[{position}] needs name, arguments, result"
                    )
                elif call["result"] is None:
                    problems.append(
                        f"{where}: tool_calls[{position}] ({call['name']}) has no result"
                    )

        if not str(record["expected"] or "").strip():
            problems.append(f"{where}: expected is not filled in")
        requirement = record["requirement"]
        if requirement is not None and requirement not in spec_ids:
            problems.append(f"{where}: requirement {requirement!r} is not in SPEC.md")

        met, source = record["met_requirement"], record["problem_source"]
        if not isinstance(met, bool):
            problems.append(f"{where}: met_requirement is not filled in")
        if source is not None and source not in PROBLEM_SOURCES:
            problems.append(
                f"{where}: problem_source {source!r} is not one of {PROBLEM_SOURCES}"
            )
        elif met is True and source is not None:
            problems.append(f"{where}: problem_source must be null when met is true")
        elif met is False and source is None:
            problems.append(f"{where}: problem_source is required when met is false")

    return problems


def missing_turns(records: list[dict[str, Any]]) -> list[str]:
    """Sessions whose records do not cover every turn the store held.

    A turn skipped at build time -- a run that died mid-answer -- leaves a gap
    no other check can see: the records present are all well formed, and verify
    only looks from the file to the store, never the other way.
    """
    seen: dict[Any, set[int]] = {}
    totals: dict[Any, int] = {}
    for record in records:
        session = record.get("session_id")
        if session is None:
            continue
        seen.setdefault(session, set()).add(record.get("turn"))
        totals[session] = max(
            totals.get(session, 0), record.get("turns_in_session") or 0
        )
    lines = []
    for session, total in sorted(totals.items()):
        absent = sorted(set(range(1, total + 1)) - seen[session])
        if absent:
            lines.append(
                f"{session} held {total} turns; no record for turn(s) "
                f"{', '.join(str(t) for t in absent)}"
            )
    return lines


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: not valid JSON -- {error}") from error
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write via a temporary file: the judgments live only here."""
    body = "".join(json.dumps(record) + "\n" for record in records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body)
    os.replace(temporary, path)


def report(records: list[dict[str, Any]], spec_path: Path) -> int:
    """Say what is left to do. Returns 0 when the file is ready to submit."""
    for line in missing_turns(records):
        print(f"note: {line}")
    problems = validate(records, spec_requirement_ids(spec_path))
    for problem in problems:
        print(problem)
    conversations = len({r.get("session_id") for r in records})
    short = conversations < MINIMUM_CONVERSATIONS
    if short:
        print(
            f"homework 1 asks for at least {MINIMUM_CONVERSATIONS} conversations; "
            f"{conversations} present"
        )
    print(
        f"{len(records)} records from {conversations} conversations, "
        f"{len(problems)} problems"
    )
    return 1 if problems or short else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session-reader", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser("dump", help="write the conversation records to a file")
    dump.add_argument("filename", type=Path, help="where to write the records")
    dump.add_argument("--db", type=Path, default=SESSIONS_DB)
    dump.add_argument("--spec", type=Path, default=SPEC_PATH)

    tail = commands.add_parser("tail", help="read the conversations")
    tail.add_argument("-f", "--follow", action="store_true", help="stream new turns")
    tail.add_argument("--db", type=Path, default=SESSIONS_DB)
    tail.add_argument("--turn", default=None, metavar="SESSION:N", help="show one turn")

    args = parser.parse_args(argv)

    if args.command == "tail":
        return _tail(args)
    return _dump(args)


def _dump(args: argparse.Namespace) -> int:
    existing = read_jsonl(args.filename)
    if not args.db.exists():
        # Without the store there is nothing to write, but the file on disk can
        # still be judged -- and saying so beats failing on a fresh clone.
        print(f"no session store at {args.db}; validating {args.filename} only")
        return report(existing, args.spec)

    turns = list(sessionreader.turn_records(args.db))
    fresh = [r for r in (to_record(turn) for turn in turns) if r is not None]
    skipped = len(turns) - len(fresh)
    records = refresh(existing, fresh)
    write_jsonl(args.filename, records)
    print(
        f"wrote {len(records)} records to {args.filename} ({skipped} turn(s) skipped)"
    )
    return report(records, args.spec)


def _tail(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"no session store at {args.db}", file=sys.stderr)
        return 1
    pattern = re.compile(LABEL_PATTERN)
    width = sessionreader._terminal_width()

    if not args.follow:
        shown = 0
        for record in sessionreader.turn_records(args.db):
            name = f"{record['session_id']}:{record['turn']}"
            if args.turn and args.turn != name:
                continue
            heading = (
                f"-- {sessionreader.label(record['session_id'], pattern)} "
                f"turn {record['turn']} of {record['turns_in_session']} "
            )
            print(sessionreader.render_turn(record, width, heading))
            shown += 1
        if not shown:
            print(f"no turn {args.turn}" if args.turn else f"{args.db} holds no turns")
            return 1
        return 0

    seen: dict[Any, Any] = {}
    watermark = sessionreader.current_watermark(args.db)
    print(f"following {args.db} from id {watermark}; Ctrl-C to stop", file=sys.stderr)
    try:
        while True:
            events, watermark = sessionreader.tail_step(args.db, watermark, seen)
            for event in events:
                print(sessionreader.render_event(event, pattern, width), flush=True)
            time.sleep(sessionreader.POLL_SECONDS)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
