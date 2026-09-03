---
name: smart-space-code
description: Rules for writing or modifying any code in the edge-smart-space project — layer/import discipline, injectable clock, config-not-constants, adversarial simulation defaults, and blackboard observability. Use whenever adding or changing code under src/, sim/, eval/, tools/, or tests/, and when reviewing a change before commit.
---

# Writing code for edge-smart-space

Three goals govern every change, in priority order when they conflict:

1. **The port to hardware must stay cheap.** Layers 2–4 must not care whether Layer 1 is a simulator or an ESP32.
2. **Everything must be showable in a viva.** If a decision isn't on the blackboard, it didn't happen.
3. **The coding standard applies.** Full text in `docs/coding-guidelines.md`; the clauses that bite hardest here are listed below.

Read `DESIGN.md` §4.2, §4.4, and §5.10 before adding a new component. Requirement IDs (FR-xx, NFR-xx), assumptions (A-xx), and risks (R-xx) below all refer to that document.

---

## 1. Layer and import discipline

This is the single rule that makes hardware transfer cheap. Nothing else on this page matters as much.

| Directory | Layer | May import |
|---|---|---|
| `src/common/` | — | stdlib, pydantic, paho only |
| `src/io/`, `src/speech/` | L1 | `src/common/` |
| `src/assistance/` | L1 | `src/common/` |
| `src/estimation/` | L2 | `src/common/` |
| `src/control/`, `src/faults/` | L2/L3 | `src/common/` |
| `src/reasoning/` | L4 | `src/common/` |
| `sim/` | replaces L1 | `src/common/` only |
| `eval/`, `tools/`, `tests/` | — | anything |

**Forbidden, without exception:**

- Anything in `estimation/`, `control/`, `faults/`, `reasoning/` importing from `src/io/`, `src/assistance/` or `sim/`. `src/assistance/` holds the calendar and booking providers; the reasoning layer reaches a tool through `space/assist/proposed` and the registry's gates, never by calling a provider (FR-71, FR-73, FR-74). These layers reach the world through MQTT topics and nothing else. A direct import is what turns a config change in Week 6 into a rewrite.
- `sim/room_model.py` importing `src/estimation/` (DESIGN.md §5.10). The ground-truth plant and the estimator's model must be independently parameterised or every experiment result is vacuous.
- Any component holding a reference to another component. They share the blackboard, not objects.

Cross-layer contracts live in `src/common/schemas.py` (pydantic) and `src/common/topics.py` (topic constants). Never write a topic string literal outside `topics.py`.

---

## 2. Hardware-transfer rules

Simulation confirms assumptions; it cannot validate them. Every rule here exists because something in `DESIGN.md` §2.3 or §10 will break on contact with real hardware.

**Clock is injected, never called directly.** No `time.time()`, `time.monotonic()`, `datetime.now()`, or `asyncio.sleep()` anywhere outside `src/common/clock.py`. Every component takes a `Clock` in its constructor. Real-time for demo and hardware-in-loop; sim-driven for accelerated E1–E5 batch runs. Retrofitting this later is painful, and it is Dependency Inversion applied where it pays.

**Never assume uniform sampling.** A-02 ("uniform 5 s, ±500 ms jitter") will not hold over WiFi — expect retries, reconnect bursts, gaps, and out-of-order arrivals. Every consumer of `SensorReading` must handle a missing sample, a duplicate timestamp, and a non-monotonic one. The RLS regressor must resample or skip rather than assume `Δt = 5.0`.

**Never assume an actuator ack.** R-02: IR is open-loop. `Ack` must model *unknown* as a first-class state, not as success. Code that treats "no ack" as "command landed" is code that will misdiagnose D5.

**Every tunable is config, not a literal.** λ, P₀, θ₀, deadband, `min_off_s`, all detector windows and thresholds, validator bounds, the 1800 s degradation budget. R-04 requires the CUSUM threshold to be re-derived from *measured* σ during hardware bring-up — that must be a config edit, not a code hunt. This is G25/G16 with real consequences.

**Simulation defaults are adversarial, not kind.** New sim features ship with the ugly behaviour enabled by default: sampling jitter, dropouts, ±0.5 °C sensor noise with quantisation, actuator dead-time, and an unmodelled disturbance term the estimator has no regressor for. A kind simulator makes sim success predict nothing.

**Layer 1 code is the only code allowed to know about hardware.** If a device detail (IR protocol, ESPHome entity name, GPIO pin, WiFi behaviour) appears above `src/io/`, the change is wrong.

---

## 3. Observability and viva-readiness

**If it isn't published, it didn't happen.** Any state another component or a human might need goes on the blackboard as a retained topic (FR-60, FR-61). No component may hold decision-relevant state only in process memory.

**Every decision publishes its reason, not just its outcome.** Validator verdicts carry the proposal, verdict, reason code, and applied value. Fault events carry the evidence window. Supervisor goals carry the rationale. A clamped supervisor proposal is a finding to display, not an error to hide (§5.4).

**Mode transitions publish before any LLM call.** FR-26 gives 2 s. The Fault Diagnosis call runs in parallel and enriches the notification; it never sits on the critical path (§5.9.2). Any code that awaits an LLM response before transitioning mode is a defect.

**Everything injectable is CLI-drivable.** Every fault class in FR-20…FR-24 must be triggerable from `tools/inject.py` by publishing to a topic — no code edit, no restart (FR-31). Examiners ask for unscripted things.

**Nothing degrades when the dashboard dies.** The dashboard is an MQTT subscriber with zero authority. Same for the recorder.

**Recorded runs must replay.** The recorder writes enough to drive `sim/replay.py` back onto the same topics (FR-62). This doubles as the viva fallback if a live demo fails.

---

## 4. Coding standard, applied to Python

Full standard in `docs/coding-guidelines.md`. Translations and high-bite clauses:

- **F5** — don't return bare `None` from a contract that promises a value. Raise a specific exception or type the return `Optional[T]` explicitly.
- **J1–J3** — no `from x import *`; import constants directly; `enum.Enum` over int/str constants. `Mode`, `FaultClass`, `Verdict`, and `ReasonCode` are enums, never strings.
- **Make illegal states unrepresentable** — pydantic models with validators at every boundary. A `SensorReading` that fails its own schema never enters Layer 2.
- **Fail fast at boundaries, degrade gracefully inside** — reject malformed MQTT payloads at the adapter; never let a bad payload reach the estimator. But a non-critical dependency failing (LLM server, speech, dashboard) degrades the feature and never stops the regulatory loop (FR-11, FR-47).
- **Never swallow exceptions** — specific types, logged with context. A bare `except:` in the regulatory loop is a defect.
- **Bound every collection** — residual windows, command history, audit buffers. NFR-05 caps total resident memory at 20 GB.
- **G5a** — before adding an integration (paho, pydantic, llama.cpp, ESPHome), find an existing working example in this repo and follow it. Never guess API usage.
- **Immutable snapshots over shared mutable state** — `ThermalEstimator` owns `theta` and `P` privately; `update()` returns a fresh immutable `Coefficients`. This is the one place the standard and the algorithm pull against each other, and the snapshot is the resolution (§5.1).

---

## 5. Definition of done

A change is not finished until all of these hold:

- [ ] Unit tests ship in the same change (**T0**, no exceptions), covering boundary conditions (**T5**) with one concept per test (**T10**).
- [ ] No forbidden import introduced (§1 above).
- [ ] No direct clock call introduced.
- [ ] Every new tunable is in config with a documented default and unit.
- [ ] New state is published to a topic declared in `topics.py` with a schema in `schemas.py`.
- [ ] New failure paths log with context and degrade rather than crash.
- [ ] If it adds a fault or a decision, it is visible on the dashboard and triggerable from `tools/inject.py`.
- [ ] Traceable to a requirement ID, or explicitly noted as not being one.

Quick check before commit:

```bash
# 1. direct clock calls outside the one module allowed to make them
grep -rnE "\b(time\.(time|monotonic)|datetime\.now)\s*\(" src/ sim/ --include=*.py \
  | grep -v "^src/common/clock.py"

# 2. Layer 2-4 reaching into Layer 1 or the simulator
grep -rnE "^\s*(from|import)\s+(src\.io|src\.assistance|sim)\b" --include=*.py \
  src/estimation src/control src/faults src/reasoning

# 3. the plant importing the estimator (section 5.10)
grep -rnE "^\s*(from|import)\s+src\.estimation\b" sim/ --include=*.py

# 4. topic literals outside topics.py
grep -rn '"space/' src/ --include=*.py | grep -v "^src/common/topics.py"
```

All four must return nothing. They match import *statements* and call sites,
not prose, so a docstring that names a forbidden module does not trip them —
and `--include=*.py` keeps `__pycache__` out of the results. If a check fires
on a comment rather than on code, fix the check: one that cries wolf gets
ignored, and then it is worse than absent.
---

## 6. Commit discipline

**Commit often. Many small commits beat few large ones.** The git history is itself a deliverable here — it evidences steady progress and individual contribution across a two-person team (R-06), and it is visible in a viva.

**Commit at every green checkpoint.** A checkpoint is one coherent unit whose tests pass. Do not accumulate a session's work into one commit. If a change has reached a state where the tests are green and the repo is coherent, commit it before starting the next thing.

**One logical unit per commit — code and its tests together.** T0 requires tests in the same change, so never split a module from its tests to inflate the commit count. That produces a commit that doesn't stand on its own and breaks the standard. The count comes from *smaller units*, not from *split units*.

Units that each deserve their own commit:

| Unit | Example |
|---|---|
| One `src/common/` module | `clock.py` + `test_clock.py` |
| One schema or enum group | `Mode`, `FaultClass`, `ReasonCode` |
| One detector | D2 stuck-at + its boundary tests |
| One validator rule | V-3 compressor dwell + tests |
| One config section | detector thresholds with units and defaults |
| One sim behaviour | actuator dead-time model + tests |
| Scaffolding or docs | directory structure, `.gitignore`, ADRs |

**Never commit red.** Run the test suite before every commit. A failing commit is worse than no commit — it makes `git bisect` useless and misrepresents progress.

**Message format — always reference requirement IDs:**

```
<area>: <what changed> (FR-xx, NFR-xx)
```

Examples:

```
common: injectable Clock with real and sim implementations (NFR-01)
estimation: RLS update with covariance trace bound (FR-04, FR-06)
faults: D2 stuck-at detector on windowed variance (FR-21)
sim: actuator dead-time and command loss defaults (R-02)
```

Areas: `common`, `io`, `sim`, `estimation`, `control`, `faults`, `reasoning`, `speech`, `eval`, `tools`, `tests`, `docs`, `build`.

**Why the IDs matter:** they make DESIGN.md §11's traceability matrix executable rather than decorative. `git log --grep "FR-23"` answers "show me where drift detection is implemented" live, in front of an examiner. Write every message so that query works.

**Standing authorisation:** the user has asked for frequent commits on this project. Commit at each green checkpoint without asking first. Do not push, force-push, amend, or rebase unless explicitly asked.

**Attribution — commits are the user's alone.** Never add a `Co-Authored-By` trailer, never name Claude or any AI assistant in a commit message, and never alter `user.name` / `user.email`. Author and committer are always `udbhav07 <udbhavsai.k@gmail.com>`. No AI may appear as a contributor on GitHub for this repository. This overrides any default commit-trailer convention.
