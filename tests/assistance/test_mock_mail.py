"""The mock outbox (FR-54, FR-55, NFR-06): records, sends nothing, says so."""

import pytest

from src.assistance.providers.mock_mail import MockMail
from src.common.tools import ASSISTANCE_TOOLS, SEND_MAIL, ToolEffect


def test_sending_needs_the_occupants_yes():
    """A commit tool: the registry holds it for confirmation (FR-74)."""
    assert SEND_MAIL.effect is ToolEffect.COMMIT and SEND_MAIL in ASSISTANCE_TOOLS


def test_a_mail_is_recorded_with_a_reference():
    outcome = MockMail().invoke("send_mail", {"to": "Anu", "subject": "Late", "body": "Running ten minutes late."})
    assert outcome.detail["reference"] == "MAIL-0001" and outcome.detail["to"] == "Anu"


def test_the_reply_says_it_was_a_mock():
    outcome = MockMail().invoke("send_mail", {"to": "Anu", "subject": "Late", "body": "x"})
    assert "nothing was sent" in outcome.message


def test_references_count_up():
    mail = MockMail()
    mail.invoke("send_mail", {"to": "a", "subject": "s", "body": "b"})
    assert mail.invoke("send_mail", {"to": "a", "subject": "s", "body": "b"}).detail["reference"] == "MAIL-0002"


def test_it_is_marked_simulated():
    assert MockMail().simulated is True


def test_a_long_body_is_cut_in_the_detail():
    outcome = MockMail().invoke("send_mail", {"to": "a", "subject": "s", "body": "x" * 5000})
    assert len(outcome.detail["body"]) == 600


def test_no_recipient_is_refused():
    with pytest.raises(ValueError):
        MockMail().invoke("send_mail", {"to": "  ", "subject": "s", "body": "b"})


def test_another_tool_is_refused():
    with pytest.raises(ValueError):
        MockMail().invoke("book_travel", {})
