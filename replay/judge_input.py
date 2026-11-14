"""The judge input contract, shared by Module 2 (analysis) and the Harbor verifier.

A frozen judge's TPR/TNR hold only for the text it was validated on, so HW5
(analysis.helpers.normalization) and HW6 (replay.rollout.judge_trace_text, run
inside each Harbor task, where the analysis package is not shipped) render
traces with these same functions and settings.
"""

from __future__ import annotations

import json
from typing import Any


def _data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if hasattr(value, "dict"):
        return value.dict(by_alias=True)
    return value


def _text(value: Any) -> str:
    value = _data(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                for p in item.get("parts") or []:
                    if isinstance(p, dict) and p.get("content"):
                        parts.append(str(p["content"]))
        if parts:
            return "\n".join(parts)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    """One line per message, `role: content`. Tool lines name the tool
    (`tool_call get_policy: {...}`), since a judge reads only this text and
    the arguments alone do not say which lookup ran."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "step")
        if role == "tool_call":
            content = _text(message.get("arguments"))
        else:
            content = _text(message.get("text", message.get("content")))
        if role in {"tool_call", "tool_result"} and message.get("name"):
            role = f"{role} {message['name']}"
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def judge_completion_kwargs(model: str) -> dict[str, Any]:
    """Temperature 0 halves verdict flips between identical runs
    (analysis/report/spike-v1-dev-noise.json). Reasoning models (gpt-5*, o*)
    accept only the default temperature."""
    return {} if model.startswith(("gpt-5", "o")) else {"temperature": 0}
