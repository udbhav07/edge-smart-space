# Roadmap

What is left to build, in the order it should happen. Written in terms of what
the system will be able to *do*, not which files change — the week-by-week
milestone table in `DESIGN.md` §9.1 stays the reference for that.

| Field | Value |
|---|---|
| Project start | 19 August 2026 |
| Last updated | 3 September 2026 |
| Position | Calendar week 3, roughly 1.5 weeks ahead of §9.1 |

---

## Done

| Goal |
|---|
| Every component talks over one message bus, so any of them can be killed and restarted independently and the whole system state can be watched from a terminal. |
| A simulated room runs against the same topics the real hardware will use, so hardware bring-up is a swap rather than a rewrite. |
| **The room identifies its own thermal model from operating data** — no offline training, no fixed gains — and keeps it across restarts. |
| Nothing can command the plant without passing a hard safety gate first. A request for 5 °C becomes 18 °C and the audit trail says why. |
| Setpoint tracking runs deterministically at its own cadence, holding its last valid target if everything above it dies. |
| Spoken requests become preferences the system weighs, not commands it obeys. |

Along the way, experiment E1 found that the model as originally specified could
not identify itself at our sensor's precision. §5.2 changed as a result
(DESIGN.md v1.2). That is a result worth reporting, not merely a fix.

---

## Pending

| Week | Dates | Goal |
|---|---|---|
| **3** | 2–8 Sep | **The room notices when a sensor is lying to it** — gone quiet, stuck, or reporting nonsense — and says so with evidence. Faults triggerable on demand, so an examiner can break it live. |
| **4** | 9–15 Sep | **The room keeps working while a sensor is broken.** It detects a drifting sensor from the model's own expectation, spots an air conditioner that isn't actually cooling, and switches to controlling on prediction instead of collapsing to open loop. This is the whole fault-tolerance claim. *Also: real hardware in hand.* |
| **5–6** | 16–29 Sep | The reasoning layer proposes goals in plain language and the gate refuses the unsafe ones — demonstrably, not by trust. |
| **7** | 30 Sep – 6 Oct | Runs on real sensors and a real air conditioner, not a simulation. |
| **8** | 7–13 Oct | **Buffer.** Hardware overruns, and every detector threshold gets re-derived from measured noise rather than the placeholders currently in config. |
| **9** | 14–20 Oct | **A person can use it without a terminal.** A local page carries the comfort band, standing prompts, confirmations, and a calendar. Asking to book a flight produces a prompt and nothing happens until it is answered; asking to schedule a meeting writes a real entry to our own calendar. Same page shows the live view. |
| **10** | 21–27 Oct | **Proof it beats a thermostat.** The same faults injected into both; ours holds the comfort bound where the baseline does not. Every experiment replayable offline. |
| **11–12** | 28 Oct – 10 Nov | The three demonstration scenarios run end to end without intervention, and the report explains why. |

---


