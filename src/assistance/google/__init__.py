"""Google Calendar and Gmail, once the occupant connects an account (FR-84 to FR-86).

Everything here is behind the existing tool surface: the reasoning layer
still invokes ``schedule_event`` or ``send_mail`` by name, and which provider
fulfils it is wiring in ``src/assistance/__main__.py`` (FR-71, FR-72).

The room is **offline-first**. The local calendar stays the source of truth,
and every call to Google goes through a persisted outbox that is drained when
Google can be reached (FR-86). Nothing here makes a request until a client
secret is in place and the occupant has signed in (FR-84, NFR-06).
"""
