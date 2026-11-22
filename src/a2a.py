"""Small A2A-style handoff contract for analyst/reviewer delegation.

The contract is transport-neutral and serializable.  It gives the analyst and
critic an explicit agent-card identity, task, input, and acceptance status,
which can later be moved to an HTTP A2A transport without changing the graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentCard(BaseModel):
    name: str
    version: str = "1.0"
    skills: list[str] = Field(default_factory=list)


class AgentHandoff(BaseModel):
    protocol: Literal["a2a"] = "a2a"
    task_id: str
    sender: AgentCard
    recipient: AgentCard
    task: str
    payload: dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    accepted: bool = False


ANALYST_CARD = AgentCard(name="wealthpilot-committee-analyst", skills=["underwriting-draft"])
CRITIC_CARD = AgentCard(name="wealthpilot-risk-compliance-critic", skills=["citation-check", "dscr-check", "bias-check"])


def create_analyst_handoff(task_id: str, payload: dict[str, Any]) -> AgentHandoff:
    return AgentHandoff(
        task_id=task_id,
        sender=ANALYST_CARD,
        recipient=CRITIC_CARD,
        task="review_underwriting_draft",
        payload=payload,
    )


def accept_critic_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    parsed = AgentHandoff.model_validate(handoff)
    if parsed.recipient.name != CRITIC_CARD.name:
        raise ValueError("Handoff recipient is not the WealthPilot critic")
    parsed.accepted = True
    return parsed.model_dump(mode="json")


__all__ = ["AgentCard", "AgentHandoff", "accept_critic_handoff", "create_analyst_handoff"]
