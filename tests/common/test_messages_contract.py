"""The contract for messages home (FR-95): words and a file name, never audio, never a path."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config import load_config
from src.common.schemas import HomeMessage, MessageState, MessageStatus


def _message(**overrides):
    fields = {"ts": 1.0, "message_id": "msg_1793_1", "sender": "Ravi", "transcript": "Call me back, Amma.", "duration_s": 3.2, "audio_file": "msg_1793_1.wav"}
    return HomeMessage(**{**fields, **overrides})


class TestHomeMessage:
    def test_a_message_names_its_file(self):
        assert _message().audio_file == "msg_1793_1.wav"

    @pytest.mark.parametrize("name", ["../secret.wav", "msg_1/../../x.wav", "C:\\\\x.wav", "msg_1.mp3", "other.wav", ""])
    def test_a_file_is_a_name_in_the_directory_never_a_path(self, name):
        with pytest.raises(ValidationError):
            _message(audio_file=name)

    @pytest.mark.parametrize("message_id", ["msg_", "MSG_1", "msg_1/2", "x" * 70])
    def test_an_id_is_a_plain_token(self, message_id):
        with pytest.raises(ValidationError):
            _message(message_id=message_id)

    def test_a_sender_is_needed(self):
        with pytest.raises(ValidationError):
            _message(sender="")

    def test_there_is_no_field_for_audio(self):
        assert not any("audio" in name and name != "audio_file" for name in HomeMessage.model_fields)


class TestMessageStatus:
    def _status(self, **overrides):
        fields = {"ts": 2.0, "message_id": "msg_1793_1", "sender": "Ravi", "duration_s": 3.2, "state": MessageState.SENT, "sent_ts": 1.0}
        return MessageStatus(**{**fields, **overrides})

    def test_a_reply_names_its_own_file(self):
        assert self._status(state=MessageState.REPLIED, reply_file="msg_1793_1-reply.wav").reply_file.endswith("-reply.wav")

    def test_no_reply_is_an_empty_name(self):
        assert self._status().reply_file == ""

    def test_a_reply_file_is_never_a_path(self):
        with pytest.raises(ValidationError):
            self._status(reply_file="../x-reply.wav")

    def test_it_survives_the_wire(self):
        status = self._status(state=MessageState.PLAYED, played_ts=3.0)
        assert MessageStatus.model_validate_json(status.model_dump_json()) == status


def test_the_messages_config_is_present():
    messages = load_config(Path("config/default.yaml")).messages
    assert messages.directory == "state/messages" and messages.reply_wait_s > 0 and messages.announce
