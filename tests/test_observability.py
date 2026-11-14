"""Authentication tests for the Homework 2 endpoints.

Offline by design: no Langfuse, no Docker, no model provider key. The only
component that would need a key is `Runner.run`, which is stubbed. Spans are
created and discarded, so nothing has to reach a collector.

Both tests attack the trust boundary rather than the happy path: identity is
established once, from the database, and a credential is authority over one
session only.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException
from opentelemetry.sdk.trace import TracerProvider

from server import app as server_app


class _StubResult:
    """Stands in for the SDK's RunResult; only final_output is read."""

    final_output = "stubbed reply"


@pytest.fixture
def agent_calls(world: dict, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Make post_message runnable without a model, and record whether it ran.

    Returns the list the stub appends to, so a test can assert the agent did
    or did not run.

    monkeypatch, rather than plain assignment: `server_app.Runner.run = stub`
    rebinds an attribute on an imported module and that rebinding outlives the
    test, so every later test in the session would get the stub. monkeypatch
    restores each attribute it sets when the test ends.
    """
    calls: list[Any] = []

    async def stub_run(*args: Any, **kwargs: Any) -> _StubResult:
        calls.append((args, kwargs))
        return _StubResult()

    monkeypatch.setattr(server_app, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(server_app.Runner, "run", stub_run)
    # A provider with no exporter attached. In a full-suite run another test
    # may already have configured global tracing, and without this the stub
    # replies would be exported to Langfuse as real-looking
    # cartwheel.session_message traces, polluting the ones Part E selects from.
    monkeypatch.setattr(server_app, "_tracer", TracerProvider().get_tracer("test"))
    # _SESSIONS is defined at module level, created once on import, and lives
    # for the whole pytest run. Without this, sessions leak between tests.
    server_app._SESSIONS.clear()
    return calls


def test_create_session_rejects_a_role_the_user_does_not_hold(
    agent_calls: list[Any],
) -> None:
    """Claiming a role is not holding one.

    User 9002 is a merchant in the seeded database. A request claiming
    `shopper` for that user is refused, and no session is created — the
    caller never obtains a credential to reuse.
    """
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(
            server_app.SessionCreate(user_id=9002, role="shopper")
        )

    # pytest.raises alone would pass on a 500; the code is the contract.
    assert exc_info.value.status_code == 403
    assert server_app._SESSIONS == {}


def test_token_from_one_session_cannot_authorize_another(
    agent_calls: list[Any],
) -> None:
    """A valid signature is authority over one session, not over any session.

    Both sessions belong to the same user, so identity is held constant and
    the session binding is the only difference between the two tokens.
    """
    first = server_app.create_session(
        server_app.SessionCreate(user_id=9002, role="merchant")
    )
    second = server_app.create_session(
        server_app.SessionCreate(user_id=9002, role="merchant")
    )
    assert first["session_id"] != second["session_id"]
    # Authentication succeeds: the HMAC is genuine and unmodified. What fails
    # below is the binding, which is why this is 403 and not 401.
    assert server_app.verify_token(first["token"]) is not None

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server_app.post_message(
                second["session_id"],
                server_app.MessageIn(message="Show my recent orders."),
                authorization=f"Bearer {first['token']}",
            )
        )

    assert exc_info.value.status_code == 403
    # The endpoint must authorize before it runs the agent. A 403 on its own is
    # equally consistent with running the agent, executing its tools, and only
    # then noticing the token was wrong.
    assert agent_calls == []

    # Positive control: the same stub does record a call when the token matches
    # its session. Without this, `agent_calls == []` above could pass because
    # the recorder never works, rather than because the agent never ran.
    asyncio.run(
        server_app.post_message(
            second["session_id"],
            server_app.MessageIn(message="Show my recent orders."),
            authorization=f"Bearer {second['token']}",
        )
    )
    assert len(agent_calls) == 1
