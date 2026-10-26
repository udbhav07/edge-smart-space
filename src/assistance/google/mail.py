"""Mail through Gmail when an account is connected; the mock otherwise (FR-85).

``send_mail`` is still a COMMIT tool: the registry holds it until the
occupant says yes (FR-54), whichever provider sends it. Once confirmed, a
Gmail message is queued in the outbox and sent at once if Google can be
reached; if not, it waits and the reply says so plainly (FR-86).

With no account connected the mock does exactly what it did before, and
says it is a mock (FR-55).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from src.assistance.google.account import GoogleAccount
from src.assistance.google.api import raw_message
from src.assistance.google.outbox import SEND, Outbox, OutboxFull
from src.assistance.providers.mock_mail import MockMail
from src.common.tools import SEND_MAIL, ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "gmail"

#: Longest body echoed back in the detail, as the mock does.
_BODY_SHOWN = 600


class GmailOrMock:
    """Sends through Gmail when connected, records a mock otherwise."""

    def __init__(self, account: GoogleAccount, outbox: Outbox, clock, flush=None, mock: MockMail | None = None) -> None:
        self._account = account
        self._outbox = outbox
        self._clock = clock
        #: Try to send now, so a reachable Google gets it at once.
        self._flush = flush
        self._mock = mock or MockMail()

    @property
    def name(self) -> str:
        return PROVIDER_NAME if self._account.connected else self._mock.name

    @property
    def simulated(self) -> bool:
        return not self._account.connected

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if tool != SEND_MAIL.name:
            raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")
        if not self._account.connected:
            return self._mock.invoke(tool, arguments)
        recipient = " ".join(str(arguments["to"]).split())
        if not recipient:
            raise ValueError("Who should it go to?")
        subject = " ".join(str(arguments.get("subject") or "").split()) or "(no subject)"
        body = str(arguments.get("body") or "").strip()
        return send_or_queue(self._outbox, self._clock, self._flush, recipient, subject, body)


def send_or_queue(outbox: Outbox, clock, flush, recipient: str, subject: str, body: str) -> ProviderOutcome:
    """Queue one mail for Gmail and try it at once; say which happened."""
    try:
        op_id = outbox.add(SEND, {"raw": raw_message(recipient, subject, body), "to": recipient, "subject": subject}, clock.now())
    except OutboxFull as exc:
        raise ValueError(f"Too much is already waiting to send ({exc}); try again later.") from exc
    if flush is not None:
        flush()
    queued = outbox.contains(op_id)
    message = (
        f"Queued the email to {recipient}: “{subject}”. It will be sent when the system is back online."
        if queued
        else f"Sent the email to {recipient}: “{subject}”."
    )
    LOGGER.info("gmail %s to %s: %s", op_id, recipient, "queued" if queued else "sent")
    return ProviderOutcome(
        message=message,
        detail={"reference": op_id.upper(), "to": recipient, "subject": subject, "body": body[:_BODY_SHOWN], "queued": queued},
    )
