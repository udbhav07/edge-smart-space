"""Email, against a mock outbox (FR-54, FR-55, NFR-06).

The node makes no outbound connection, so nothing here can send anything:
a message is recorded, given a reference, and kept in a bounded outbox the
console shows as "Sent". Every sentence this produces says it was a mock,
and ``simulated`` is True, which the registry copies onto every result, so
the mock is identifiable on the blackboard without reading the text.

Mail commits the occupant to someone outside the system, so the tool is
declared ``commit`` and reaches this only after an explicit yes (FR-74);
that gate is the registry's, not this class's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from src.common.tools import SEND_MAIL, ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "mock_mail"

#: Longest body echoed back in the result's detail, so the console can show
#: what was "sent" without the result growing without bound.
_BODY_SHOWN = 600


class MockMail:
    """Records a message and sends nothing."""

    def __init__(self) -> None:
        self._next_reference = 1

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def simulated(self) -> bool:
        return True

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """Pretend to send, and say plainly that it was pretend.

        :raises ValueError: for a tool this provider does not serve, or a
            message with nobody to send it to.
        """
        if tool != SEND_MAIL.name:
            raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")
        recipient = " ".join(str(arguments["to"]).split())
        if not recipient:
            raise ValueError("Who should it go to?")
        subject = " ".join(str(arguments.get("subject") or "").split()) or "(no subject)"
        body = str(arguments.get("body") or "").strip()
        reference = f"MAIL-{self._next_reference:04d}"
        self._next_reference += 1
        LOGGER.info("MOCK mail %s to %s: %s", reference, recipient, subject)
        return ProviderOutcome(
            message=(
                f"Mock email {reference} to {recipient} recorded: “{subject}”. "
                f"This is a mock: nothing was sent."
            ),
            detail={"reference": reference, "to": recipient, "subject": subject, "body": body[:_BODY_SHOWN]},
        )
