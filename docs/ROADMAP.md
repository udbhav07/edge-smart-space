# Roadmap

What is left to build, in the order it should happen. Written in terms of what
the system will be able to *do*, not which files change — the week-by-week
milestone table in `DESIGN.md` §9.1 stays the reference for that.



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
| **The reasoning layer can ask for an action, not just a temperature** — a calendar entry, a booking — through one declared surface that does not know which service fulfils it. Adding a tool, or moving from our calendar to a hosted one, touches nothing else. |
| The system knows which of its own actions it is allowed to take alone. It writes to a calendar on its own authority and stops dead at anything that would commit you to an outsider. |


---

## Pending

| Week | Dates | Goal |
|---|---|---|
| **3** | 2–8 Sep | **The room notices when a sensor is lying to it** — gone quiet, stuck, or reporting nonsense — and says so with evidence. Faults triggerable on demand, so an examiner can break it live. |
| **4** | 9–15 Sep | **The room keeps working while a sensor is broken.** It detects a drifting sensor from the model's own expectation, spots an air conditioner that isn't actually cooling, and switches to controlling on prediction instead of collapsing to open loop. This is the whole fault-tolerance claim. *Also: hardware in hand.* |
| **5** | 16–22 Sep | **It runs on the Jetson Orin, not a laptop.** The stack is provisioned on the Orin and comes up on boot; temperature, humidity and power sensors are wired and publishing to the same topics the simulator already uses, and the air conditioner takes real commands. Simulated and real sources are selectable by config, so every existing test runs unchanged against either. |
| **6** | 23–29 Sep | **The reasoning layer both proposes and acts.** It proposes goals in plain language and the gate refuses the unsafe ones — demonstrably, not by trust. Asking for a meeting on Thursday puts it in the calendar and it says so; asking about Thursday reads it back. Asking for a flight gets you a question, not a booking. |
| **7** | 30 Sep – 6 Oct | **Closed loop on real data.** Identification, control and fault detection all run on live sensor readings against a real air conditioner, and every detector threshold is re-derived from measured noise rather than the placeholders currently in config. |
| **8** | 7–13 Oct | **A person can use it without a terminal — and it beats a thermostat.** A local page carries the comfort band, standing prompts, the calendar the system has been writing to, and the confirmations that bookings have been waiting on — answer one and the mock endpoint runs, and the page says plainly that it was a mock. Same page shows the live view. In parallel, the same faults are injected into ours and a baseline thermostat; ours holds the comfort bound where the baseline does not, and every experiment replays offline. |
| **9** | 14–15 Oct | The three demonstration scenarios run end to end on the hardware without intervention, and the report explains why. |

---


