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
| **The reasoning layer both proposes and acts (Week 6).** A supervisor proposes setpoints from what it reads and the gate refuses the unsafe ones, visibly. "Make it 5 degrees" is answered with the safe range; a meeting on Thursday lands in the calendar on Thursday and is read back; a flight gets a question that names what would be booked and that it is a mock. Every fault is explained to the occupant after the mode has already changed. |
| **It is ready for the Jetson, in software (Week 5).** Simulated and real devices are one config switch apart, every service has a systemd unit that restarts it, and the launcher does the same on a laptop. *Nothing has run on the hardware yet.* |
| **The thresholds are the room's, not the simulator's (Week 7, software).** A recording of a healthy room becomes a calibration: every noise-derived detector threshold is re-derived from what was measured, written as a configuration layer that states beside each value what it came from. Under thresholds calibrated from two healthy simulated hours every fault class is still found, drift sooner, with no false alarm in ten healthy hours. *It has been calibrated from the simulator only.* |
| **The unit can say whether it obeyed (Week 6, software).** A meter on the air conditioner's supply is the readback an IR blaster cannot give: a command that landed draws, or stops drawing, and the bridge acknowledges on it. The bring-up check passes a working unit and fails a dead one. A fault on an instrument the loop does not control on — the meter, the humidity sensor — is reported and no longer degrades the room. *No meter has been wired.* |
| **It beats a thermostat, measured (E5).** With the indoor sensor gone silent, this system holds the comfort band for the whole half hour while the same thermostat is outside it 69 % of the time. Getting there meant fixing the experiment before the system: the simulated unit could not hold its own room, and the comparison scored against a setpoint neither side was given. |
| **The room keeps working while a sensor is broken.** It catches a drifting sensor from the model's own expectation and an air conditioner that is not actually cooling from the room failing to respond — the two faults a thermostat cannot detect at all. When a sensor fails it controls on the model's prediction instead of collapsing to open loop, and stops when that has run long enough to stop being trustworthy. *Hardware still to arrive.* |


---

## Pending

| Week | Dates | Goal |
|---|---|---|
| **5–6** | 16–29 Sep | **It runs on the Jetson Orin, not a laptop — the hardware half.** The software is done (above): provisioning, units, the device bridge, the meter's readback. What remains is provisioning the Orin, flashing the node, filling its topics into `site.yaml`, and `tools.bringup` passing on the real instruments. |
| **7** | 30 Sep – 6 Oct | **Closed loop on real data — the hardware half.** The re-derivation itself is built (above); what remains is a healthy recording from the real room, `tools.calibrate` run on it, and identification, control and detection closed on live readings against the real unit. |
| **8** | 7–13 Oct | **A person can use it without a terminal — and it beats a thermostat.** A local page carries the comfort band, standing prompts, the calendar the system has been writing to, and the confirmations that bookings have been waiting on — answer one and the mock endpoint runs, and the page says plainly that it was a mock. Same page shows the live view. In parallel, the same faults are injected into ours and a baseline thermostat; ours holds the comfort bound where the baseline does not, and every experiment replays offline. |
| **9** | 14–15 Oct | The three demonstration scenarios run end to end on the hardware without intervention, and the report explains why. |

---


