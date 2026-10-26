"""The connected Google account: its token, kept fresh, never published (FR-84).

The console writes ``token.json`` (and ``account.json``, the address) when
the occupant signs in; this reads them on every use, so signing in again, or
disconnecting, takes effect without a restart. The token never leaves the
file: what the rest of the system learns is :class:`GoogleStatus`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.config import GoogleConfig
from src.common.schemas import GOOGLE_SCOPES

LOGGER = logging.getLogger(__name__)

SCOPES = GOOGLE_SCOPES

ACCOUNT_FILE = "account.json"


class GoogleAccount:
    """Credentials from the token file, refreshed when they expire."""

    def __init__(self, config: GoogleConfig) -> None:
        self._config = config
        self._token_path = Path(config.token_path)
        self._refused_mtime: float | None = None

    @property
    def configured(self) -> bool:
        """Whether a client secret is in place, so signing in is possible."""
        return Path(self._config.client_secret_path).is_file()

    @property
    def connected(self) -> bool:
        """Whether a token is held that Google has not refused since it was written."""
        if not self._token_path.is_file():
            return False
        return self._refused_mtime != self._mtime()

    @property
    def needs_reconnect(self) -> bool:
        return self._token_path.is_file() and self._refused_mtime == self._mtime()

    @property
    def email(self) -> str:
        try:
            return str(json.loads((self._token_path.parent / ACCOUNT_FILE).read_text(encoding="utf-8")).get("email", ""))
        except (OSError, ValueError):
            return ""

    def refused(self) -> None:
        """Google turned the token down; it stays refused until a new one is written."""
        self._refused_mtime = self._mtime()

    def credentials(self) -> Any:
        """Current credentials, refreshed if expired; None if not connected.

        :raises GoogleAuthError: the refresh was refused.
        :raises GoogleOffline: the refresh could not reach Google.
        """
        from google.auth.exceptions import RefreshError, TransportError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        from src.assistance.google.api import GoogleAuthError, GoogleOffline

        if not self.connected:
            return None
        try:
            credentials = Credentials.from_authorized_user_file(str(self._token_path), list(SCOPES))
        except (OSError, ValueError) as exc:
            LOGGER.warning("the Google token at %s is unreadable: %s", self._token_path, exc)
            return None
        if credentials.valid:
            return credentials
        if not credentials.refresh_token:
            self.refused()
            raise GoogleAuthError("the sign-in has no refresh token; connect again")
        try:
            credentials.refresh(Request())
        except RefreshError as exc:
            self.refused()
            raise GoogleAuthError("the sign-in has expired; connect again") from exc
        except TransportError as exc:
            raise GoogleOffline(f"Google is unreachable: {exc}") from exc
        self._save(credentials.to_json())
        return credentials

    def _save(self, token_json: str) -> None:
        temporary = self._token_path.with_suffix(".tmp")
        temporary.write_text(token_json, encoding="utf-8")
        temporary.replace(self._token_path)

    def _mtime(self) -> float | None:
        try:
            return self._token_path.stat().st_mtime
        except OSError:
            return None
