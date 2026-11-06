# Plan: web search, news and live cricket scores

Branch: `feat/web-search`. Status: implemented. What changed against this plan is listed at the end.
Requirements: FR-101 (web search), FR-102 (news), FR-103 (live cricket scores), and a revision of NFR-06.

Jarvis can answer questions about the room, the calendar and the devices, but
nothing about the world outside. This plan adds three read-only assistant
tools: a general web search, a news search, and live cricket scores. Each
answer comes from what the tool returned, with its source and age, and never
from the model's memory.

---

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Web search first, then news, then cricket scores, all on this branch | Each phase builds on the previous one's provider and prompt work |
| 2 | On by default; each feature has its own switch in config | The feature is the point of the branch; the switches keep NFR-06 enforceable |
| 3 | SearXNG is an outside service that `start.py` starts and stops, using Podman | One command still brings the whole room up |
| 4 | Only requests addressed to Jarvis may search, enforced in the executor, not just the prompt | Overheard speech must never leave the house |
| 5 | No confirmation before a search. Every query is visible on the MQTT board and in the chat trace | Searching is read-only and changes nothing |
| 6 | Answers come from result snippets only. Reading a full article is a later step | Article pages are slow and paywalled, and too long for a 7B model on CPU |
| 7 | Every answer names its source and age. With nothing relevant, Jarvis says "I couldn't find that" | A stale or invented answer must never sound current |
| 8 | Default region and language `en-IN`, set in config | Indian news and sources first |
| 9 | A swappable provider interface, with only SearXNG and Cricinfo built | Tavily, Brave or CricketData.org can each be added later as one file |
| 10 | Three narrow tools: `web_search`, `news_search`, `cricket_score` | A 7B model chooses correctly more often when each tool does one obvious thing |
| 11 | Live scores come from Cricinfo's live-scores RSS feed | Free, needs no key, and returned 15 live matches when checked on 2026-11-06 |
| 12 | The SearXNG image is pinned and downloaded automatically on first run | The project pins every dependency |
| 13 | No clickable links in the chat. Queries and results appear as plain text in the trace | Keeps the frontend unchanged |
| 14 | NFR-06 is rewritten to list every outbound exception, including YouTube | The YouTube exception (FR-96) was never recorded there |
| 15 | Commits use `<area>: <what> (FR-xx)` with no AI trailer | Project rule in `.claude/skills/smart-space-code` |

### Options considered

- **Search provider.** SearXNG, self-hosted, is free and needs no key or account
  on the node. Tavily, Brave, Exa and Ollama's web search are all hosted, need a
  key, and are mostly paid; Brave dropped its free tier in February 2026. Hosted
  providers return cleaner results, which is why the provider is swappable.
- **Live scores.** Web search cannot give a live score reliably: search
  snippets are snapshots, and score pages fill in the numbers with JavaScript. A
  dedicated feed is needed. Cricinfo's feed is free with no key but carries
  scores only. CricketData.org has more detail but needs a key, allows 100
  requests a day, and runs a few minutes behind.

---

## How it fits together

```
"Jarvis, what's today's news?"
  → reasoning (qwen2.5 7B) chooses news_search ──► space/assist/proposed   (query visible on the MQTT board and chat trace)
  → assistance service runs the provider
       web_search, news_search ──► SearXNG on 127.0.0.1:8888 (Podman, started by start.py)
       cricket_score ────────────► static.cricinfo.com/rss/livescores.xml
  → result ──────────────────────────────────────► space/assist/result
  → reasoning phrases the reply from the results only: answer, source and age, or "I couldn't find that"
```

Everything travels on the existing `space/assist/*` topics. There are no new
topics, so DESIGN §6.1 and `tests/common/test_design_conformance.py` are
unchanged.

### Facts from the code that shape the design

- Tools are declared as `ToolSpec` in `src/common/tools.py` and bound to providers in `build_registry()` in `src/assistance/__main__.py`.
- A READ tool never needs confirmation; only COMMIT tools do.
- Only a result's `message` reaches the model. `detail` is flat key/value pairs and is never shown to it.
- The reply is phrased by the model only when `_needs_phrasing()` (`src/reasoning/assistant.py`) says so. Otherwise the provider's message is spoken as written.
- `max_tool_rounds` is 1, so each tool is called at most once per request.
- The assistant waits `assistance.result_timeout_s` (14 s) for a result, so a search has to finish well within that.
- Overheard speech (`src/reasoning/ambient.py`) calls the model with no tools at all.
- The current time reaches the assistant prompt from the injected clock (`local_now`, `site.utc_offset_h`).
- Layer 1 code (`src/assistance/`) may import `src/common` and the standard library only.
- `src/` has no HTTP client of its own, so providers use `urllib`.

---

## Phase 0: foundations

### 0.1 Config

`WebSearchConfig` in `src/common/config.py`, defaulted on `Config`, with matching entries in `config/default.yaml`:

```yaml
web_search:
  enabled: true
  provider: searxng
  url: http://127.0.0.1:8888
  region: en-IN
  max_results: 5
  snippet_chars: 300          # keeps the prompt small for a 7B model on CPU
  timeout_s: 6.0              # validator: must be below assistance.result_timeout_s
  searxng:
    manage: true              # start.py runs it
    image: docker.io/searxng/searxng:<pinned release tag>
    port: 8888
  cricket:
    enabled: true
    provider: cricinfo
    feed_url: https://static.cricinfo.com/rss/livescores.xml
    cache_s: 30
```

A `Config` validator requires `web_search.timeout_s < assistance.result_timeout_s`.

### 0.2 SearXNG settings

`deploy/searxng/settings.yml`:
- JSON output enabled (SearXNG serves HTML only by default)
- bound to localhost, rate limiter off (this machine is the only client)
- default language `en-IN`

The secret key is generated on first run into `state/` and passed in as an
environment variable. It is never committed.

### 0.3 `start.py`

- **Starts SearXNG:** when `web_search.enabled` and `searxng.manage` are both on and nothing answers on the port, it runs `podman run --rm --name edge-searxng -p 127.0.0.1:8888:8080 -v deploy/searxng:/etc/searxng:Z,ro …`.
- **Reuses** an instance that is already answering.
- **Stops** it on `Ctrl-C` with the other services.
- **First run** pulls the pinned image, which needs internet.
- **Podman missing, or the pull fails:** a warning, and everything else still starts.
- **`--check`** gets a third dependency line, e.g. `ok searxng at localhost:8888`.
- **Docstring:** update the line saying outside services are "never started".
- **Jetson:** a systemd unit in `deploy/` under `smart-space.target`.

### 0.4 Design document (`docs/DESIGN.md`)

- **New requirements in §3.6:**
  - FR-101 web search, including the rule that only addressed requests may search
  - FR-102 news, including what "today", "this morning" and "yesterday" mean
  - FR-103 live cricket scores
- **NFR-06** rewritten to list each outbound exception and the switch that turns it off:
  - Google after sign-in (FR-84)
  - YouTube (FR-96)
  - SearXNG's upstream engines (FR-101, FR-102)
  - Cricinfo (FR-103)
- **Other updates:**
  - the §5.7.6 tool table, which is already out of date
  - the traceability matrix and revision history
  - a terms-of-use caveat for Cricinfo next to YouTube's

---

## Phase 1: web search (FR-101)

1. **Provider.** `src/assistance/providers/web_search.py` holds a `SearchBackend`
   interface (`search(query, category, time_range) → hits`) and a SearXNG
   backend. HTTP goes through `urllib`, so there is no new dependency. The fetch
   function is injected, so tests use a fake.
2. **Tool.**
   - `WEB_SEARCH` in `src/common/tools.py`, effect READ, bound in `build_registry()`.
   - Its result message is a numbered list: `title — site — age — snippet — link`.
   - A provider failure such as a timeout or SearXNG being down raises `ValueError` with a sentence the occupant can understand.
3. **Reasoning.**
   - `_needs_phrasing()` covers the search tools, so results are phrased by the model and never read out raw.
   - A search answer prompt tells the model to use the results only, name the source and its age, and say it couldn't find the answer when the results don't contain it.
   - `ASSISTANT_PROMPT` sends questions about the outside world to the tools; questions about the room still use room data.
4. **Addressed requests only.** The executor refuses any search tool whose
   requester is overheard speech. This guards against future changes, since
   overheard speech has no tools today.

## Phase 2: news (FR-102)

- `NEWS_SEARCH` uses SearXNG's news category, with a `when` argument: `today | this_morning | yesterday | this_week`.
- SearXNG can only narrow results to the past day or week. The provider filters further by each article's publish time, with day boundaries from the injected clock and `site.utc_offset_h`.
- Ages are given in plain words ("2 hours ago"). An article without a date is said to have none rather than given a guessed one.

## Phase 3: live cricket scores (FR-103)

- `CRICKET_SCORE` takes an optional team, e.g. "India" or "RCB".
- A `ScoreSource` interface with a Cricinfo provider:
  - reads the feed with the standard library's XML parser
  - matches the team named
  - caches the feed for `cricket.cache_s`
- Example replies:
  - "Bangladesh 88 for 3, batting, against West Indies' 356 (Cricinfo, just now)."
  - "No live match for India right now."
- The feed carries scores only, with no overs or ball-by-ball detail. CricketData.org can be added later as another `ScoreSource`.

---

## Tests

Each test ships with the change it covers. None of them touch the network.

| Area | Cases |
|---|---|
| SearXNG backend | JSON results, timeout, empty result, malformed response, region passed through |
| Cricinfo provider | A saved copy of the real feed as test data; a team found, a team not playing, the cache honoured |
| News filtering | A fake clock across "this morning", "yesterday" and articles with no date |
| Executor | A search tool requested by overheard speech is refused |
| Reasoning | Search results go through the answer prompt; no results gives "couldn't find" |
| `start.py` | The Podman command built, Podman missing, an already-running instance reused, the `--check` line |
| Config | `timeout_s` must be below `result_timeout_s` |
| CI smoke test | Runs with web search off, so CI needs neither Podman nor internet |

## Documentation

The README gets a "Web search" section covering:
- Podman
- the first-run image download
- how to turn each feature off
- the new `--check` output

## Commit order

1. `common: web search config ... (FR-101)`
2. `deploy: SearXNG settings ... (FR-101)`
3. `build: start.py starts SearXNG and checks it ... (FR-101)`
4. `docs: FR-101 to FR-103 and NFR-06 listing every outbound exception (FR-101, FR-102, FR-103, NFR-06)`
5. `assistance: web_search through SearXNG ... (FR-101)`
6. `reasoning: search results phrased with source and age ... (FR-101)`
7. `assistance: overheard speech can never search ... (FR-101)`
8. `assistance: news_search with today, this morning and yesterday ... (FR-102)`
9. `assistance: cricket_score from Cricinfo's live feed ... (FR-103)`
10. `docs: web search in the README ... (FR-101, FR-102, FR-103)`

Each commit passes the full test suite on its own.

## Not in this plan

- Clickable source links in the chat
- Hosted search providers (Tavily, Brave, Exa) and CricketData.org
- Ball-by-ball commentary

## What changed during implementation

- **SearXNG's time filter is not used.** Measured on the pinned image,
  `time_range=day` drops Brave News and DuckDuckGo News and leaves results with
  no dates. News is fetched unfiltered and narrowed by each result's own date,
  read from `publishedDate`, a relative `pubdate` ("11 hours ago") or the head
  of `metadata` ("48 minutes ago | News18").
- **Readiness is `/healthz`, not the port.** Podman's port forwarder accepts
  connections as soon as the container starts, so a TCP probe passed before
  SearXNG could answer and the first searches were reset.
- **SearXNG is stopped by name.** Ending the `podman run` client left the
  container running; `start.py` runs `podman stop`, and `--rm` removes it.
- **`searxng.port` became part of `web_search.url`**, so the two cannot disagree.
- **Cricket gained `max_matches` and `aliases`** (RCB, CSK and the rest of the
  IPL), because the feed writes every team's full name.
- **The Orin unit is `deploy/searxng.service`**, beside `llama-server.service`,
  installed and given its secret by `provision.sh`.
- **Web and news search landed in one commit**: one provider serves both tools.

## Added after: deep answers (FR-104)

Asked for after the first version: reading the top pages, and a second
search, for questions the snippets cannot answer.

- **Who decides.** The model, given the snippets, may reply `MORE:` (with
  better words) instead of an answer. A question the snippets answer costs no
  extra call. Headlines and scores are never offered it.
- **Who chooses the pages.** The assistant, from the search's own results;
  `read_pages` is never offered to the model, and the reader refuses anything
  a search did not return, and any address inside the network, redirects
  included.
- **Bounded.** Three pages read at once, 5 s and 2 MiB each, 1,500 characters
  kept per page, trimmed to the question. One more search at most.
- **Measured live.** Three real pages in 0.7 s, about 3,850 characters to the
  model. One site answered 200 with "Access Denied", which is why a page with
  almost no text is not counted as read.
- **Not checked live:** the 7B model's use of `MORE:`, since Ollama was kept
  off during this work.
- **Fixed after review.** The name was looked up twice, once to check and
  once to connect, so a hostile name could answer public and then
  127.0.0.1 (DNS rebinding; reproduced). Now each connection connects to the
  address it checked. A bad charset or an answer cut short no longer fails
  the whole read. A page still arriving is told to stop. The text extraction
  reads divs and table cells, and a menu left unclosed no longer hides the
  page. `MORE:` is heard after a sentence, and only in capitals. An undated
  cricket feed says when it was read. The reader's numbers are config.
