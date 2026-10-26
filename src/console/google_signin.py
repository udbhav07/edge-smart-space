"""Signing in to Google from the console (FR-84).

The occupant presses "Connect Google": the console sends the browser to
Google's consent page, and Google sends it back to ``/api/google/callback``
with a code, which is exchanged here for a token. The token is written to
``google.token_path`` -- the only place it ever goes -- and the address to
``account.json`` beside it. The assistance service reads both from there;
nothing about the token is published.

OAuth 2.0 for a web client whose redirect URI is this console's own
``http://localhost`` address, with PKCE. HTTP is allowed for that loopback
address only; anything else is refused.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from src.common.config import GoogleConfig
from src.common.schemas import GOOGLE_SCOPES

LOGGER = logging.getLogger(__name__)

#: Sign-ins started and not yet finished, by their state value. Bounded.
_PENDING_KEPT = 8


class SignInError(ValueError):
    """Signing in could not be started or finished, in words for the occupant."""


class GoogleSignIn:
    """Starts and finishes the OAuth exchange; writes the token file."""

    def __init__(self, config: GoogleConfig) -> None:
        self._config = config
        self._pending: dict[str, object] = {}

    @property
    def configured(self) -> bool:
        return Path(self._config.client_secret_path).is_file()

    def start(self) -> str:
        """The URL of Google's consent page.

        :raises SignInError: no client secret is in place yet.
        """
        if not self.configured:
            raise SignInError(
                f"Google is not set up yet: put the OAuth client JSON from Google Cloud at "
                f"{self._config.client_secret_path} (see the README), then try again."
            )
        flow = self._flow()
        url, state = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="true")
        self._pending[state] = flow
        while len(self._pending) > _PENDING_KEPT:
            self._pending.pop(next(iter(self._pending)))
        return url

    def finish(self, state: str, code: str) -> str:
        """Exchange the code for a token and keep it. Returns the address.

        :raises SignInError: an unknown or expired sign-in, or Google refused.
        """
        flow = self._pending.pop(state, None)
        if flow is None:
            raise SignInError("That sign-in has expired; press Connect Google again.")
        self._allow_loopback_http()
        try:
            flow.fetch_token(code=code)
        except Exception as exc:
            # oauthlib raises a family of its own; any of them means the same
            # thing to the occupant, and the detail goes to the log.
            LOGGER.warning("Google sign-in failed: %s", exc)
            raise SignInError("Google did not accept the sign-in; please try again.") from exc
        credentials = flow.credentials
        email = _email_of(getattr(credentials, "id_token", None))
        token_path = Path(self._config.token_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        _write(token_path, credentials.to_json())
        _write(token_path.parent / "account.json", json.dumps({"email": email}))
        LOGGER.info("connected Google account %s", email or "(address unknown)")
        return email

    def disconnect(self) -> bool:
        """Forget the account: the token and the address. Returns whether there was one."""
        removed = False
        token_path = Path(self._config.token_path)
        for path in (token_path, token_path.parent / "account.json"):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
        return removed

    def _flow(self):
        from google_auth_oauthlib.flow import Flow

        return Flow.from_client_secrets_file(
            self._config.client_secret_path,
            scopes=list(GOOGLE_SCOPES),
            redirect_uri=self._config.redirect_uri,
            autogenerate_code_verifier=True,
        )

    def _allow_loopback_http(self) -> None:
        """oauthlib insists on HTTPS; a loopback redirect is the documented exception."""
        host = urlparse(self._config.redirect_uri).hostname or ""
        if host not in ("localhost", "127.0.0.1", "::1"):
            return
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        # Google adds openid scopes of its own to the grant; that is not an error.
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"


def _email_of(id_token: str | None) -> str:
    """The address in an ID token Google just handed over directly, over TLS.

    Read, not verified: it came straight from Google's token endpoint in the
    exchange above, and it is only shown to the occupant, never trusted for
    anything.
    """
    if not id_token:
        return ""
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return str(json.loads(base64.urlsafe_b64decode(payload)).get("email", ""))
    except (IndexError, ValueError):
        return ""


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
