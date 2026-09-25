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
| **The room notices when a sensor is lying to it** — gone quiet, stuck, or reporting nonsense — and says so with the evidence that produced the finding. Any of those is triggerable from a terminal while it runs, so an examiner can break it live; the fault is applied at the sensor, so nothing above can tell an injected one from a real one. Drift and actuator faults are Week 4. |
| **The room keeps working while a sensor is broken.** It catches a drifting sensor from the model's own expectation and an air conditioner that is not actually cooling from the room failing to respond — the two faults a thermostat cannot detect at all. When a sensor fails it controls on the model's prediction instead of collapsing to open loop, and stops when that has run long enough to stop being trustworthy. *Hardware still to arrive.* |
| **The hardware swap is a configuration line.** `io.source` selects the room simulator or the ESPHome bridge; both publish the same topics, so the rest of the system never learns which is running and every test applies to either. The broker, the systemd units and the node definition are written. *What still needs the Orin: flashing a board, wiring the sensors, and re-deriving every detector threshold against a real room.* |
| **The system acts, not just advises.** Asking for a meeting puts it in a real calendar and reads it back; asking for a flight returns a question rather than a booking, and confirming it runs a mock that says so in every result. A spoken “it is too warm in here” becomes a setpoint the safety gate still clamps — asking for 5 °C gets 21 °C and an audit record saying why. *The supervisor that chooses goals arrived in Week 6, below.* |
| **It is ready to run on the Jetson Orin, not a laptop.** One script provisions the Orin — MAXN and pinned clocks, the broker, `llama-server` built with CUDA and the model it serves, a service user, every unit — and the whole stack comes up on boot. Temperature, humidity and now the air conditioner's power draw publish on the topics hardware will use, and the simulated and real Layer 1 are selected by configuration, never both. A bring-up check reports every sensor's rate, jitter and measured noise, and proves the air conditioner takes commands by asking the gate for cooling and watching the power meter. *What still needs hands on the board: flashing the ESP32, wiring the sensors and the meter, and settling the IR protocol — then the bring-up check says whether it worked.* |
| **The reasoning layer both proposes and acts.** The supervisor reads the room through its tools and proposes goals in plain language, on a cadence and whenever occupancy, the tariff or a fault changes; the gate refuses the unsafe ones where everyone can see — a proposal of 5 °C comes back `CLAMPED`, and the model is told so. Asking for a meeting on Thursday puts it in the calendar and it says so; asking about Thursday reads it back; asking for a flight gets a question, not a booking. Every fault is explained in words after the mode has already changed, and every reasoning call is on the audit topic with its latency. Killing the reasoning process costs reasoning and nothing else. Any of it is askable from a terminal. *What still needs a model: E6 is written and its scoring tested, but its numbers come from the Orin's server.* |


---

## Pending

| Week | Dates | Goal |
|---|---|---|
| **7** | 30 Sep – 6 Oct | **Closed loop on real data.** Identification, control and fault detection all run on live sensor readings against a real air conditioner, and every detector threshold is re-derived from measured noise rather than the placeholders currently in config. |
| **8** | 7–13 Oct | **A person can use it without a terminal — and it beats a thermostat.** A local page carries the comfort band, standing prompts, the calendar the system has been writing to, and the confirmations that bookings have been waiting on — answer one and the mock endpoint runs, and the page says plainly that it was a mock. Same page shows the live view. In parallel, the same faults are injected into ours and a baseline thermostat; ours holds the comfort bound where the baseline does not, and every experiment replays offline. |
| **9** | 14–15 Oct | The three demonstration scenarios run end to end on the hardware without intervention, and the report explains why. |

---


