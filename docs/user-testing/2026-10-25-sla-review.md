# Customer acceptance check: promised features vs. what was delivered

**Date:** 2026-10-25, 22:20–00:10 local (IST).
**Tested by:** an end user who bought the product, with no technical knowledge. Only what the vendor documents was used: the README's "Running" instructions, the console at http://127.0.0.1:8077, and the terminal commands the README tells customers to run (`tools.overhear`, `tools.inject`, `tools.demo`). No source code was read to decide a verdict.
**Build:** branch `fix/live-run-defects`, head `23f4507`. Windows 11, mosquitto, Ollama `qwen2.5:7b` (loaded), CUDA. Started from a clean stop with `python start.py --check` and then `python start.py`. Layer 1 was the simulator. The real microphone was attached, and speech ran in its default mode (always listening).

**How the console was driven.** Each console action was made through the same HTTP calls its buttons make (`/api/say`, `/api/calendar/*`, `/api/bookings/{id}`, `/api/mail`, `/api/memory/settle`, `/api/operator/*`), and results were read from `/api/state`. That kept the timings honest. All 11 pages were also opened in a browser and looked at, at desktop and phone width. The scripts and raw transcripts are listed under "Evidence" at the end.

## Status, 2026-10-26

Every finding is addressed on `fix/live-run-defects`; `git log --grep S-02` shows the work for any one. Each was then replayed live, as the customer did, on a clean restart (`sla_verify.py`, in the evidence folder). Notes give what was seen live.

| ID | Status | Commits | Live re-check |
|---|---|---|---|
| S-01 | Fixed | 9d0f88f | About 30 min with the microphone live: no new tentative entries; only the four deliberate test plans remain. Overheard speech keeps only the segments Whisper is confident in, a looping transcript is dropped, and a plan needs a day or time said. |
| S-02 | Fixed | 82f3808 | `tools.inject temp_01 drift 0.01` gave `D4_DRIFT on temp_01` at 450 s, `DEGRADED_SENSOR`, explained as drift; the AC stayed in service. The meter shows whether the unit ran, and humidity whether the room really warmed. E3 seeds 1–3: drift → D4, dead unit → D5; five healthy one-hour runs, no faults. |
| S-03 | Fixed | 0c51c14, 397e540 | The first replay found two entries named "plumber" at 11:00, which made the move ask "which one?". Follow-ups now name the entry by id. Unit tests replay both live variants and the duplicate-name case. The room goal was unchanged in the replay (24.0 → 24.0). |
| S-04 | Fixed | 0c51c14, 397e540 | As S-03: "cancel it" cancels the entry just put in. |
| S-05 | Fixed | f135ff0 | After "forget that I'm vegetarian", "what do you know about me?" listed the profile as it stands, without "vegetarian". |
| S-06 | Fixed | f135ff0 | A guest: "I don't know who you are, so I keep nothing about you." |
| S-07 | Fixed, one open item | 767a39d | Stale retained points are no longer charted. The tile says "cooling toward 24.0°" while the room is above it. **Open:** after a restart the estimator's first published `t_in` was 23.2 °C with a current timestamp, while every raw reading was 29.0. It is one point (a 5 s jump, not a climb), and its source is not yet found. Nothing is retained on `space/sensor/+/state`. |
| S-08 | Fixed | 0cbf137 | `space/system/mode` with 2 messages: 0.0/min, not 19.6. |
| S-09 | Fixed | 0cbf137 | "A mock email to anu@example.com: Dinner". |
| S-10 | Fixed | 0cbf137 | The inbox is kept in `state/console/inbox.json`, bounded at 100. |
| S-11 | Fixed | 9c61d1d | "next Tuesday", said on Monday: "… on Tuesday 27 October (that's tomorrow; say if you meant the week after)". |
| S-12 | Fixed | f135ff0 | Typed as Ravi: "(You're chatting as Ravi …)". |
| S-13 | Fixed | 9c61d1d, a967a0a | The first replay read "The system is the temperature sensor is not working", which was broken grammar. It now reads "Not quite. The temperature sensor is not working, so the room is being controlled on an estimate. …" |
| S-14 | Fixed | 0cbf137, a967a0a | The first replay's inbox note began "D5_ACTUATOR_NO_RESPONSE on ac". It now reads "The air conditioner isn't cooling the room, so it is on hold. When the air conditioner has been checked, press Reset on Faults & control." Overview shows the same while the hold lasts. |

## Verdict scale

- **BREACH-High:** the product does the wrong thing to the room or the calendar, or tells the customer something false.
- **BREACH-Medium:** a promised feature fails or misleads, but nothing is harmed.
- **BREACH-Low:** polish: confusing wording or display.
- **PASS:** works as promised. **NOT TESTED:** outside what an unassisted customer could check here.

IDs are stable (`S-xx`); refer to them in commit messages, as was done for the `U-xx` findings in [2026-10-08-findings.md](2026-10-08-findings.md).

---

## Summary

| | Count |
|---|---|
| Promises checked | 41 |
| PASS | 32 (two of them with a wording finding) |
| Promises breached | 9 rows of the checklist, giving the 14 findings below |
| BREACH-High | 3 (S-01, S-02, S-03) |
| BREACH-Medium | 4 (S-04 to S-07) |
| BREACH-Low | 7 (S-08 to S-14) |
| NOT TESTED | 5 |

## Promise checklist

| # | Promise (where it is made) | Verdict | Finding |
|---|---|---|---|
| 1 | `start.py --check` reports the broker and the AI server (README, Running) | PASS | `ok mqtt broker`, `ok inference server`, `qwen2.5:7b is loaded` |
| 2 | `start.py` starts everything; the console is at :8077 (README) | PASS | Console up within seconds, all services live |
| 3 | Every page the README lists exists and can be bookmarked | PASS | All 11 routes return 200 and render |
| 4 | Pages load nothing from the internet (README, NFR-06) | PASS | No external URLs in `index.html` or the bundles |
| 5 | Works on a phone-width screen | PASS | 520 px overview renders cleanly |
| 6 | Overview shows indoor, humidity, outside, presence, AC, tariff, live | PASS | All present and updating |
| 7 | The room is held at the setpoint shown | BREACH-Medium | S-07 |
| 8 | Services page judges each process by its own output and rate | BREACH-Low | S-08 |
| 9 | "Make it 22 degrees" sets 22 | PASS | "Done. The room is now set to 22 degrees." The goal showed 22.0 at once |
| 10 | An unsafe request is clamped and explained | PASS | 10 °C: "outside the safe range of 18 to 30 … heading there and is 20 degrees for now" |
| 11 | "I'm a bit cold" warms the room | PASS | Goal 20 → 21 |
| 12 | Calendar entry by chat | PASS | "Added meeting with Ravi at 15:00 on Thursday 29 October." |
| 13 | Asking what is on the calendar | PASS | Listed both Thursday entries |
| 14 | Add and remove entries on the Calendar page | PASS | Added, appeared, removed, gone |
| 15 | A follow-up about an entry just made ("make it half past …", "cancel it") | BREACH-High / Medium | S-03, S-04 |
| 16 | A booking asks first and is clearly a mock (README, Bookings page) | PASS | "That would be a mock booking … Shall I go ahead?" |
| 17 | Saying No books nothing | PASS | 0 new bookings, nothing left waiting |
| 18 | Saying Yes records a mock booking | PASS | `MOCK-0001` hotel in Pune, "This is a mock" |
| 19 | Mail asks first and is only a mock (Mail page) | PASS | Held, then `MAIL-0001` in Sent (mock) after Yes |
| 20 | What is waiting for a yes is described clearly | BREACH-Low | S-09 |
| 21 | Inbox: "the room writes here when something happens" | BREACH-Low | S-10 (lost on restart), and see S-01 (flooded) |
| 22 | "Remember …" keeps a fact (README) | PASS | "I'll remember that, Ravi." Fact in profile |
| 23 | "Forget …" removes it | PASS | "Forgotten: Is vegetarian." Fact gone from profile |
| 24 | Jarvis tells you what it remembers, truthfully | BREACH-Medium (intermittent) | S-05 |
| 25 | A guest gets no one else's memory (README) | BREACH-Medium | S-06 |
| 26 | Recognition only personalises; it unlocks nothing | PASS, with a wording issue | Stated on the first reply; see S-12 |
| 27 | Always listening: overheard speech is never answered (README) | PASS | Nothing overheard ever got a reply in the conversation |
| 28 | An overheard plan is pencilled in as model tentative, with the words heard | PASS | "Eye test, Wednesday 17:00 … tentatively", hatched on Calendar |
| 29 | An overheard trip is pencilled in, never booked | PASS | "Train to Mysore, Saturday 09:00"; no booking |
| 30 | Small talk produces nothing | PASS | "that film last night …" → nothing |
| 31 | An overheard fact waits on Chat; kept only on Keep | PASS | Appeared under Overheard; moved to facts on Keep |
| 32 | Keep makes a pencilled plan firm | PASS | Eye test `tentative: false` after Keep |
| 33 | Overhearing with the real microphone picks up useful things only | BREACH-High | S-01 |
| 34 | A thermometer that drops out is found in about 15 s (Faults page) | PASS | `DEGRADED_SENSOR` after 16 s, explained |
| 35 | A stuck thermometer is found in about 5 min (Faults page) | PASS | After 313 s, explained; room still controlled |
| 36 | A dead AC is found in about 10 min (Faults page) | PASS | After 602 s, explained; Reset returns to normal (see S-14) |
| 37 | A slowly drifting thermometer is found (README, `tools.inject temp_01 drift 0.01`) | BREACH-High | S-02 |
| 38 | "Is everything OK?" during a fault tells the truth | PASS, with a wording issue | S-13 |
| 39 | The room keeps being controlled with the AI gone (README) | PASS | AC commands kept flowing; AI back within seconds |
| 40 | Close the console and the room runs the same (README) | PASS | Commands unaffected; console restarted itself |
| 41 | `tools.demo` and `--scenario actuator` show the system working without a broker (README) | PASS | Both ran to completion, narrated |

**NOT TESTED:**
- **Speaking to Jarvis** with the wake word into the microphone. This needs a person; the microphone itself was live and transcribing.
- **`Ctrl-K`**, the command palette.
- **Opening the console from a phone on the LAN.**
- **Real hardware**, through `devices.source: esphome`.
- **The "Impossible reading (999°)"** button.

---

## Findings

### S-01 (BREACH-High): the microphone fills the calendar with junk "model tentative" entries

**Promise.** README, "Always listening": overheard speech is read for plans and facts, and a plan is pencilled in. Calendar page: "pencils in plans it overhears".

**Steps.** Start the product and leave it running in a normal room with the microphone attached. Open Calendar after an hour or so.

**Expected.** Nothing, unless someone actually talked about a plan.

**Observed.** Nine tentative entries made from background audio. Nobody discussed any of these plans. Each also posted a "Pencilled in from conversation" message to the Inbox.

| Entry | Time | What was "heard" |
|---|---|---|
| Grab a second | 16:51 | "and the pilot me, everyone is okay, I'll grab you a second." |
| Run | 17:00 | "then log, log 1 need to run there." |
| Going home | 17:00 | "We are going to go home We are going home No we are going home …" |
| Check water | 17:31 | "I'm going to go to the water and see if there's any water in the water …" |
| Going to the clock | 18:00 | "for 2 power 1. Until the remaining ending you will go to the clock." |
| Put in a minute | 19:28 | "So that I'm going to put all of those in a minute." |
| Good night | 22:30 | "to be safe. So, good night. Thanks. I'll see you in a minute. Bye. Bye. Bye …" |
| Meeting | 23:00 | "Okay. All right. Okay. We'll get to them. There we go. All r…" |
| Grab information | 23:00 | "Let's grab a few information." |

Several of them are for *today*, minutes after the words were heard. The typed stand-in (`tools.overhear`) behaves well, as in checklist rows 28–31. The breach comes from real room audio: TV, video or garbled transcription.

**Fixed when.** An hour of ordinary background audio adds no tentative entries. A plan needs a named day or time *and* a recognisable event. Garbled or repetitive transcripts are rejected.

### S-02 (BREACH-High): a drifting thermometer is blamed on the working air conditioner, which is put on hold

**Promise.** README, "Breaking it on purpose": `python -m tools.inject temp_01 drift 0.01` gives "0.01 C per second, invisible per sample", and every fault the detector bank can find is triggerable.

**Steps.**
1. In normal mode, run `python -m tools.inject temp_01 drift 0.01`.
2. Watch the Faults & control page.

**Expected.** A sensor fault on the indoor thermometer, the room run on its model, and an explanation.

**Observed.**
- **0 to 480 s:** NORMAL. The indoor reading rose 22.7 → 26.7 °C with no fault, while the AC was working.
- **At 510 s:** `DEGRADED_ACTUATOR`, with fault `D5_ACTUATOR_NO_RESPONSE on ac`. The **healthy air conditioner was put on hold** because of the thermometer's drift.
- **Explanation:** none shown (the diagnosis was empty when checked).
- **Afterwards:** clearing the injection and pressing Reset returned the system to NORMAL.

This matches the open item recorded earlier (drift blamed on the AC). It is still present.

**Fixed when.** The same command produces a sensor fault on `temp_01`, never an actuator fault, and the AC keeps working.

### S-03 (BREACH-High): a time-only follow-up about an entry changes the room or claims a change it did not make

**Promise.** Chat page and How it works: Jarvis understands follow-ups (FR-52 session context); a temperature change and a calendar change are different things.

**Steps and observations.** These were two runs of the same idea.
1. **First run, as Ravi:**
   - "put the dentist in my calendar on Wednesday at two" gave "Added Dentist appointment at 14:00 on Wednesday 28 October."
   - "actually make it half past two" gave "Sure, I'll adjust the target temperature to 21.5 °C." The **room setpoint changed** and the dentist stayed at 14:00.
2. **Second run, as Ravi:**
   - "put the plumber in my calendar on Friday at eleven" gave "Added Plumber appointment at 11:00 on Friday 30 October."
   - "actually make it half past eleven" gave "Sure, I'll update the plumber appointment to 11:30 on Friday 30 October."
   - The **entry stayed at 11:00**. The room was not touched, but the reply **claims an action that never happened**.

"Move it to four" (whole hours) works, as verified on 2026-10-25 earlier in the day. "Make it half past …" does not.

**Fixed when.** "Actually make it half past two" after creating an entry moves that entry to 14:30. It never changes the room, and the reply only states what the calendar confirms.

### S-04 (BREACH-Medium): "cancel it" right after discussing an entry is not understood

**Steps.** As Ravi, straight after S-03 run 1, say "cancel it".
**Observed.** "Sure, which entry would you like to cancel? Could you please provide more details …"
**Expected.** The entry just discussed (the dentist) is cancelled, or Jarvis asks "the dentist on Wednesday?"
**Fixed when.** "Cancel it" refers to the last entry this person created or moved.

### S-05 (BREACH-Medium, intermittent): Jarvis recites a fact that was just forgotten

**Steps.**
1. As Ravi: "remember that I'm vegetarian", then "forget that I'm vegetarian". The reply was "Forgotten: Is vegetarian." and the profile no longer has it.
2. On the Chat page, keep the overheard fact "Can never work if the room is noisy."
3. Ask "what do you know about me?"

**Observed (first run).** "… prefer standing desks, and are vegetarian." The forgotten fact was recited, and the newly kept one was missing.
**Second run (badminton).** The answer was correct. It happens 1 in 2.
**Fixed when.** The answer only ever lists facts in the profile at that moment.

### S-06 (BREACH-Medium): a guest asking "what do you know about me?" is told someone else's plans

**Steps.** As a guest (no profile chosen), ask "what do you know about me?"
**Observed.** "I know that you prefer the room temperature around 21 degrees Celsius and we have some events planned … including a meeting with Ravi on Thursday at 3:00 PM."
**Expected.** "I don't know who you are, so I keep nothing about you." Room-wide facts can be offered only as room facts, not as "about you".
**Fixed when.** A guest is never told that a setting or another person's entry is something known about *them*.

### S-07 (BREACH-Medium): just after start, the room warmed 24 → 29 °C while the screen said "All good, holding 24.0°"

**Steps.** `python start.py` from a clean stop, then open Overview.
**Observed.**
- **First two minutes:** the "Measured" line rose from about 24 °C to about 29 °C. AC power read 0 W for the first minute, then the AC switched to COOL at 1,186 W.
- **While that happened:** the status read "All good" and the setpoint "holding 24.0°".
- **Recovery:** it took about 10 more minutes to come back below 28 °C.

**Expected.** The room starts near where it was and is held there, or the screen says the room is still settling or warming up.
**Needs a developer to judge** whether the simulated room really starts that warm, or whether the first readings are wrong. To a customer it looks like the product let the room overheat while saying all is well.

### S-08 (BREACH-Low): Services page overstates how busy a service is

**Observed.** Fault detectors showed **99 messages/min**, with 12 messages seen in total. `space/sensor/pwr_01/health` and `space/system/mode` showed **19.6/min** each, with a message count of **2**. Two messages close together are reported as a sustained rate.
**Fixed when.** A rate is shown only once enough messages exist to measure one, or it is shown as "on change".

### S-09 (BREACH-Low): a mail waiting for a yes is labelled "A mock send_mail"

**Observed.** The waiting card for a mail reads "A mock send_mail" (also for one written in Chat to "Ravi").
**Fixed when.** It reads like the booking cards: "A mock email to anu@example.com: Dinner".

### S-10 (BREACH-Low): the Inbox is emptied whenever the product restarts

**Observed.** Before the restart, the Mail badge showed 18 messages. After `python start.py`, the inbox read "Inbox is empty".
**Fixed when.** The inbox survives a restart, or the page says it only shows messages since the last start.

### S-11 (BREACH-Low): "next Monday" said on a Sunday is booked for tomorrow

**Observed.** On Sunday 25 October, "book me a flight to Delhi next Monday" gave "a flight to Delhi on Monday 26 October". The booking asked first, so nothing went wrong, but many people mean 2 November.
**Fixed when.** An ambiguous "next <weekday>" falling within two days is confirmed ("tomorrow, the 26th?").

### S-12 (BREACH-Low): typed messages are answered "I recognised your voice as Ravi"

**Observed.** A message *typed* in Chat as Ravi got "(I recognised your voice as Ravi. That only personalises my replies …)".
**Fixed when.** For typed messages it says "you're chatting as Ravi".

### S-13 (BREACH-Low): "Everything seems fine" during an active sensor fault

**Observed.** While `DEGRADED_SENSOR` was active, "is everything ok?" gave "Everything seems fine, just a note that the temperature sensor isn't working properly …".
**Fixed when.** An active fault is never described as "everything fine".

### S-14 (BREACH-Low): after the AC is repaired nothing tells the customer to press Reset

**Observed.** The AC fault was cleared, but the mode stayed `DEGRADED_ACTUATOR` 181 s later; Reset on Faults & control fixed it. The explanation did say "on hold … until we can check it", but neither the Inbox nor the Overview says that the next step is to press Reset.
**Fixed when.** While the AC is on hold, Overview or the Inbox says "Checked the air conditioner? Press Reset on Faults & control."

---

## Test data left behind (for whoever checks this)

These were created by this check in the real calendar and profile files (`state/`):
- **Firm calendar entries:** meeting with Ravi (Thu 15:00), Dentist appointment (Wed 14:00), Plumber appointment (Fri 11:00), Eye test (kept; Wed 17:00).
- **Tentative calendar entries:** Train to Mysore, plus all the S-01 junk entries.
- **Mock records:** booking `MOCK-0001` and mail `MAIL-0001`.
- **Ravi's profile:** kept fact "Can never work if the room is noisy."; "vegetarian" and "badminton" were remembered and then forgotten.

The system was left running after the check, back in NORMAL mode, with the setpoint at 23 °C.

## Evidence

All in the session scratchpad, `C:\Users\goran\AppData\Local\Temp\claude\c--Users-goran-Desktop-FEAT-control-core\f04c3d1d-3600-4ec0-947a-2d90b3d14687\scratchpad\`:

| File | What |
|---|---|
| `sla_user.py` | The console actions, made as the buttons make them |
| `sla_round1.py` / `.txt` | Comfort, calendar, bookings, mail (rows 9–21) |
| `sla_round2.py` / `.txt` | Memory, sessions, always listening (rows 22–33; S-03 run 1, S-04, S-05, S-06) |
| `sla_round3.py` / `.txt` | Faults, AI and console stopped (rows 34–40) |
| `sla_drift.py` / `.txt` | S-02 |
| `sla_repro.py` / `.txt` | S-03 run 2, S-05 second run, S-01 list |
| `sla_tour.sh`, `sla-*.png` | Every page, desktop and phone |
| `sla_run.log` | The output of every service for the whole check |
