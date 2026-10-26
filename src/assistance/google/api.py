"""The edge of Google: the calls the room makes, and how they fail.

Behind :class:`GoogleApi` so the outbox, the sync and every provider can be
tested with a fake -- no network, no account -- and so the three ways a call
fails are named rather than inferred from exception text:

* :class:`GoogleOffline` -- Google could not be reached, or answered with a
  server error. Retried later; nothing is lost (FR-86).
* :class:`GoogleAuthError` -- Google refused the token. The occupant must
  sign in again; retried after they do.
* :class:`GoogleRejected` -- Google understood and said no (a malformed
  request). Retrying would never help, so the operation is dropped and the
  reason shown.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Callable, Protocol

LOGGER = logging.getLogger(__name__)

#: Statuses Google uses for "try again later".
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})


class GoogleError(Exception):
    """A call to Google did not complete."""


class GoogleOffline(GoogleError):
    """Google could not be reached; the call should be retried."""


class GoogleAuthError(GoogleError):
    """Google refused the credentials; the occupant must sign in again."""


class GoogleRejected(GoogleError):
    """Google refused the request itself; retrying will not help."""


class GoogleAlreadyExists(GoogleRejected):
    """An insert whose id is already taken: an earlier attempt landed."""


class GoogleNotFound(GoogleRejected):
    """The thing addressed is gone: a delete has nothing left to do."""


class GoogleApi(Protocol):
    def insert_event(self, calendar_id: str, body: dict) -> str: ...
    def patch_event(self, calendar_id: str, event_id: str, body: dict) -> None: ...
    def delete_event(self, calendar_id: str, event_id: str) -> None: ...
    def list_events(self, calendar_id: str, time_min: str, time_max: str) -> list[dict]: ...
    def send_mail(self, raw: str) -> str: ...


def raw_message(to: str, subject: str, body: str) -> str:
    """An email as Gmail's API takes it: RFC 2822, base64url."""
    from email.message import EmailMessage

    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body or "")
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


class RealGoogleApi:
    """Calendar v3 and Gmail v1 through google-api-python-client.

    Built per call from whatever credentials are current, so a token renewed
    or replaced by signing in again is picked up without a restart. Discovery
    documents are the ones shipped with the library: nothing is fetched to
    learn the API's shape.
    """

    def __init__(self, credentials: Callable[[], Any]) -> None:
        self._credentials = credentials

    def _service(self, name: str, version: str):
        from googleapiclient.discovery import build

        credentials = self._credentials()
        if credentials is None:
            raise GoogleAuthError("no Google account is connected")
        return build(name, version, credentials=credentials, cache_discovery=False, static_discovery=True)

    def insert_event(self, calendar_id: str, body: dict) -> str:
        created = self._call(lambda: self._service("calendar", "v3").events().insert(calendarId=calendar_id, body=body).execute())
        return str(created.get("id", body.get("id", "")))

    def patch_event(self, calendar_id: str, event_id: str, body: dict) -> None:
        self._call(lambda: self._service("calendar", "v3").events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute())

    def delete_event(self, calendar_id: str, event_id: str) -> None:
        self._call(lambda: self._service("calendar", "v3").events().delete(calendarId=calendar_id, eventId=event_id).execute())

    def list_events(self, calendar_id: str, time_min: str, time_max: str) -> list[dict]:
        events: list[dict] = []
        token = None
        while True:
            page = self._call(lambda: self._service("calendar", "v3").events().list(
                calendarId=calendar_id, timeMin=time_min, timeMax=time_max,
                singleEvents=True, orderBy="startTime", pageToken=token, maxResults=250,
            ).execute())
            events.extend(page.get("items", []))
            token = page.get("nextPageToken")
            if not token:
                return events

    def send_mail(self, raw: str) -> str:
        sent = self._call(lambda: self._service("gmail", "v1").users().messages().send(userId="me", body={"raw": raw}).execute())
        return str(sent.get("id", ""))

    @staticmethod
    def _call(request: Callable[[], Any]) -> Any:
        """Run one request, turning every failure into one of the three kinds."""
        from google.auth.exceptions import RefreshError, TransportError
        from googleapiclient.errors import HttpError

        try:
            return request()
        except HttpError as exc:
            status = int(getattr(exc.resp, "status", 0) or 0)
            if status == 401:
                raise GoogleAuthError("Google refused the sign-in; connect again") from exc
            if status == 409:
                raise GoogleAlreadyExists(str(exc)) from exc
            if status in (404, 410):
                raise GoogleNotFound(str(exc)) from exc
            if status in _RETRYABLE:
                raise GoogleOffline(f"Google answered {status}") from exc
            raise GoogleRejected(f"Google refused it ({status})") from exc
        except RefreshError as exc:
            raise GoogleAuthError("the sign-in has expired; connect again") from exc
        except (TransportError, OSError, TimeoutError) as exc:
            raise GoogleOffline(f"Google is unreachable: {exc}") from exc
