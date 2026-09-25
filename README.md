# edge-smart-space

A self-calibrating, fault-tolerant autonomous personal smart space that runs
entirely on one edge node.

The room builds a model of its own thermal behaviour online, and keeps
controlling through sensor and actuator faults instead of acting on bad
readings. Those two things are the contribution. Voice interaction and
contextual assistance demonstrate the reasoning layer; they are not the point.

Design document: [docs/DESIGN.md](docs/DESIGN.md).

---

## How it fits together

Three cadences, deliberately separated:

| Layer | Period | Determinism | Role |
|---|---|---|---|
| Reasoning (LLM) | 300 s | non-deterministic | proposes setpoint goals, diagnoses faults |
| Supervision + safety | event-driven | rule-based | arbitrates goals, enforces hard limits |
| Regulatory | 5 s | fully deterministic | tracks the setpoint, detects faults |

The reasoning layer never touches an actuator. Everything it wants goes
through a proposed setpoint that the safety validator can clamp or refuse. The
regulatory loop keeps running when reasoning is slow, wrong, or entirely
absent — that independence is a requirement, and killing the reasoning process
mid-run is part of the demonstration.

Components share nothing but an MQTT blackboard, so any one of them can be
restarted alone and the whole system is inspectable with `mosquitto_sub`.

---

## Requirements

- **Python 3.11 or 3.12.** The core needs 3.11+ (numpy 2.3 dropped 3.10). The
  optional speech stack pins torch 2.5.1, which has no wheel for 3.13+, so use
  3.11 or 3.12 if you want voice. CI runs 3.11.
- **An MQTT broker** — mosquitto. Required: it is the only thing connecting
  components.
- **An inference server** — Ollama or `llama.cpp`, OpenAI-compatible on
  `localhost:11434`. Optional; the control loop runs without it.
- **NVIDIA GPU** — optional. The project is CUDA-first but falls back to CPU.

---

## Setup

```bash
git clone https://github.com/udbhav07/edge-smart-space.git
cd edge-smart-space

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # core + test tools
pip install -e ".[dev,speech]"     # add voice (needs Python 3.11/3.12)
```

`speech` pulls torch, faster-whisper, openWakeWord, Silero VAD and PyAudio.
It is a large install and is why it is optional — everything except the voice
pipeline works without it, and the speech tests skip themselves when it is
absent.

**Linux (Ubuntu 22.04)** needs PortAudio headers before PyAudio will build:

```bash
sudo apt update && sudo apt install -y portaudio19-dev
```

Wake-word models download once, at setup rather than at startup, because the
deployed node has no internet:

```bash
python setup_models.py
```

---

## External services

Neither is started by this project — both are checked for.

**MQTT broker**

```bash
sudo apt install mosquitto mosquitto-clients   # Ubuntu
sudo systemctl start mosquitto                 # Ubuntu / Jetson
brew services start mosquitto                  # macOS
net start mosquitto                            # Windows
```

**Inference server** (only needed for reasoning)

```bash
ollama serve
ollama pull qwen2.5:7b
```

On the Orin it is `llama-server` built with CUDA, installed and started by
`deploy/provision.sh` (see *Running on the Jetson* below). Either answers the
same OpenAI-compatible protocol on `localhost:11434`, and nothing in the code
knows which is running.

---

## Running

Check everything is in place first:

```bash
python start.py --check
```

```
ok       mqtt broker at localhost:1883
ok       inference server at localhost:11434
```

A missing broker stops the launch. A missing inference server is a warning
and the run continues — regulatory control does not depend on reasoning.

Then start everything:

```bash
python start.py
```

Or a subset:

```bash
python start.py --only simulator
python start.py --skip speech
```

Each service runs as its own process with its output prefixed. `Ctrl-C` stops
them all. `python start.py --help` lists what can currently be started.

### Running one component directly

```bash
python -m sim.run_sim --steps 100     # room plant, sensors, actuator
python -m src.estimation              # online RC identification
python -m src.control                 # regulatory loop and safety gate
python -m src.faults                  # fault detector bank (D1-D5)
python -m src.assistance              # calendar and the booking gate
python -m src.reasoning               # supervisor, Personal Context, diagnosis
python -m src.speech                  # wake word and transcription
```

Both take `--config` and both stop cleanly on `Ctrl-C`.

### Watching what happens

Every decision in the system is on the blackboard. Nothing is hidden in
process memory, so a debugger is never needed to see what it did:

```bash
mosquitto_sub -t 'space/#' -v
```

Useful subsets:

```bash
mosquitto_sub -t 'space/sensor/+/state' -v      # raw readings
mosquitto_sub -t 'space/audit/validation' -v    # what the safety gate decided
mosquitto_sub -t 'space/system/mode' -v         # degradation state
```

A clamped setpoint appearing in `space/audit/validation` is the gate working,
not a failure. Every verdict carries the proposal, the reason code, and the
value actually applied.

### Talking to it

The reasoning process answers anything said to the room, whether it came from
the microphone or a keyboard: speech publishes each transcript to
`space/context/utterance`, and so does this:

```bash
python -m tools.say "it is too warm in here"
python -m tools.say "put the design review in on Thursday at three"
python -m tools.say "what have I got on Thursday?"
python -m tools.say "book me a flight to Delhi on Friday"
```

A temperature request becomes a preference the gate still clamps — ask for
5 °C and you get the validator's answer, not 5 °C. A meeting goes into the
calendar at `state/calendar.json` and the reply says so. A flight comes back
as a question, because a booking commits you to someone outside the system
and only you can agree to that:

```bash
python -m tools.confirm        # describes each waiting booking, asks y/N
```

A yes reaches the mock endpoint, and every result it produces says it is a
mock. Nothing real is ever booked.

The Environmental Supervisor runs on its own every five minutes and whenever
occupancy, the tariff or the fault state changes. It reads the room through
four tools and proposes a setpoint through a fifth; the gate decides. Every
call any of the three reasoning call sites makes — inputs, what the model
said, which tools it used, the verdict, latency and tokens — is on
`space/audit/reasoning`:

```bash
mosquitto_sub -t 'space/audit/reasoning' -v   # every reasoning decision
mosquitto_sub -t 'space/diagnosis' -v         # what each fault means, in words
```

Kill `src.reasoning` mid-run and nothing but reasoning stops: the control loop
holds the last setpoint the gate admitted, and an occupant's spoken request
still reaches it.

### Running on the Jetson

One script provisions the Orin and makes the whole stack come up on boot —
MAXN power mode, the broker, `llama-server` built with CUDA and the model it
serves, a service user, every systemd unit:

```bash
sudo deploy/provision.sh --dry-run            # read every step first
sudo deploy/provision.sh                      # simulator as Layer 1
sudo deploy/provision.sh --source esphome     # the real sensors and AC
```

Then check the hardware is really there:

```bash
python -m tools.bringup                       # every sensor: rate, jitter, limits, sigma
python -m tools.bringup --actuate             # and does the AC run when asked?
```

`--actuate` never sends a command itself. It asks the gate for cooling as an
operator and watches the power meter for the compressor starting — an IR
blaster cannot acknowledge, a watt-meter can. The sigma it reports per sensor
is what Week 7 sets the detector thresholds from.

### Running on hardware instead of the simulator

Layer 1 is chosen by configuration, not by code. Everything above it
subscribes to the same topics either way, so the whole test suite applies to
both:

```yaml
io:
  source: simulated   # or: esphome
```

`python start.py` then launches the room simulator or the hardware bridge and
leaves the other four services untouched. The device topics each node
publishes on live in the same `io:` block, because they are decided when a
node is flashed rather than when this code was written.

**Nothing in `deploy/` has touched hardware yet**, and it says so where it
matters. `deploy/esphome/room-node.yaml` is marked UNVERIFIED: the pins, the
one-wire address and the IR protocol are placeholders until a board exists.
What is real is the shape — and a test asserts that every device topic the
config expects is one the node definition actually publishes, so a rename
cannot silently disconnect them.

`deploy/provision.sh` does the following and more; by hand it is:

```bash
docker compose -f deploy/docker-compose.yml up -d    # the broker
sudo cp deploy/systemd/* /etc/systemd/system/        # the components
sudo systemctl enable space-layer1@space.service     # or space-simulator@
sudo systemctl enable --now space.target
```

`space.target` deliberately does not name a Layer 1: exactly one of the
hardware bridge and the simulator is enabled into it, and each refuses to run
if `io.source` names the other.

Every unit restarts itself and none requires another, so killing any one
process and watching the rest carry on is a thing you can demonstrate.

### Seeing the whole thing work, without a broker

```bash
python -m tools.demo                      # a sensor breaks, the loop survives
python -m tools.demo --scenario actuator  # the air conditioner stops cooling
```

Runs all four services in one process against a simulated clock, so half an
hour of room time takes a couple of seconds, and narrates what happens. It is
the quickest way to see the system work, and the fallback if a live
demonstration fails.

### Measuring it

The experiments in `eval/experiments/` are the evidence behind the report.
They run against the simulator with no broker and print what they measured:

```bash
python -m eval.experiments.e1_convergence    # does the model converge?
python -m eval.experiments.e2_adaptation     # does adapting help tracking?
python -m eval.experiments.e3_detection      # every fault class, repeated
python -m eval.experiments.e4_degradation    # is the 1800 s budget right?
python -m eval.experiments.e5_baseline       # do we beat a thermostat?
python -m eval.experiments.e6_tool_selection # does the model pick the right tool?
```

E6 is the one that needs the inference server. It runs fifty hand-built
scenarios through the real call sites and reports schema validity, tool
selection and argument plausibility separately, because the first is close to
100% by construction and is not a result.

E5 is the one the project stands on. It runs the same faults against this
system and against a fixed-deadband thermostat that shares the same plant,
seed, sensors, control law and setpoint — the only differences are the
identified model and the fault layer, so any difference in the result is
attributable to them.

### Breaking it on purpose

Every fault the detector bank can find is triggerable from a terminal while
the system runs — no code edit, no restart:

```bash
python -m tools.inject --list                # what can be injected, and where
python -m tools.inject temp_01 stuck 27.0    # freeze the indoor sensor
python -m tools.inject temp_01 dropout       # make it go quiet
python -m tools.inject temp_01 range 999.0   # report something impossible
python -m tools.inject temp_01 drift 0.01    # 0.01 C per second, invisible per sample
python -m tools.inject temp_01 clear         # stop injecting
```

Then watch what the detectors make of it:

```bash
mosquitto_sub -t 'space/fault/#' -v             # faults, with their evidence
mosquitto_sub -t 'space/sensor/+/health' -v     # what is still trusted
mosquitto_sub -t 'space/system/mode' -v         # what the system is allowed to do
mosquitto_sub -t 'space/actuator/ac/command' -v # proof the loop is still closed
```

The interesting thing to watch is the last two together. When a sensor is
detected as broken the mode becomes `DEGRADED_SENSOR` — and the commands
**keep coming**, because the loop switches to the model's prediction instead of
the reading it no longer trusts. That substitution is the whole point of
identifying a model, and it is the thing a thermostat cannot do. It is bounded:
after 30 minutes the system stops rather than treat a stale prediction as a
measurement, and the mode becomes `SAFE_HOLD`.

`SAFE_HOLD` is the one state the system will not leave on its own:

```bash
python -m tools.reset --reason "replaced the sensor"
```

The fault is applied at the sensor, so nothing above Layer 1 can tell an
injected fault from a real one — which is the only way the detection means
anything. `space/inject/{subject}` is retained, so it always answers what is
being injected right now.

Detection is not instant, and the delays are honest ones: a dropout takes 15 s
(three missed samples), an implausible reading 10 s (two samples), a stuck
sensor about five minutes because that is how long the variance window is, and
an actuator fault about ten minutes because that is how slowly a room responds.
Drift is found by accumulating evidence, so how long it takes depends on how
fast the sensor is drifting — which is the point of using a cumulative test
rather than a threshold.

---

## Tests

```bash
pytest                                  # whole suite
pytest --cov --cov-report=term-missing  # with coverage
pytest tests/control -q                 # one area
```

Tests needing torch or PortAudio skip themselves when those are not installed,
so a core-only checkout still runs the full core suite rather than erroring at
collection. CI installs the speech extra, so nothing is skipped there.

---

## Configuration

Every tunable lives in [config/default.yaml](config/default.yaml) — no numeric
policy value is written in code. Detector thresholds especially: they get
re-derived from measured noise during hardware bring-up, and that has to be a
config edit rather than a code hunt.

Point any entry point at a different file with `--config`.

A few worth knowing:

| Setting | Default | Note |
|---|---|---|
| `speech.asr_device` | `auto` | Prefers CUDA, falls back to CPU. `cuda` demands it and fails loudly; `cpu` forces it |
| `validator.setpoint_bounds_c` | 18.0–30.0 | Hard limits nothing can propose past |
| `controller.min_off_s` | 180 | Compressor protection, enforced twice on purpose |
| `sim.*` | all imperfections **on** | Jitter, dropouts, noise, dead-time, command loss |

The simulator defaults are adversarial deliberately. A kind simulator makes
simulated success predict nothing about real hardware.

---

## Layout

```
src/common/      clock, config, schemas, topics, MQTT blackboard, tools, local time
src/control/     arbitration, safety gate, regulatory loop, tariff (python -m src.control)
src/estimation/  RC model, RLS, persistence                    (python -m src.estimation)
src/faults/      detector bank D1-D5, aggregator, mode manager (python -m src.faults)
src/reasoning/   supervisor, Personal Context, diagnosis       (python -m src.reasoning)
src/assistance/  tool executor, calendar, travel mock          (python -m src.assistance)
src/speech/      wake word, capture, ASR                       (python -m src.speech)
src/io/          ESPHome bridge and actuator driver            (python -m src.io)
sim/             ground-truth room model, sensors, actuator, power meter, runner
eval/            baseline thermostat, harness, experiments E1-E6
tools/           view, inject, reset, record, demo, say, confirm, bringup
deploy/          provisioning, systemd units, broker, ESPHome node
tests/           unit and system tests, mirroring the source layout
config/          default.yaml
docs/            DESIGN.md, ROADMAP.md, coding-guidelines.md
```

`sim/room_model.py` must never import from `src/estimation/`. The simulated
plant and the estimator's model have to be parameterised independently, or the
evaluation degenerates into the model predicting itself.

---

## Not built yet

Honest about the gaps, so nobody hunts for something that isn't there:

- **Anything that needs a person holding the board.** The ESP32 has not been
  flashed; `deploy/esphome/room-node.yaml` is marked UNVERIFIED and its pins,
  addresses and IR protocol are placeholders. `tools.bringup` is how the
  wiring gets checked once it exists.
- **E6 against a real model.** The experiment and its scoring are built and
  tested against scripted models; the numbers need the Orin's server.
- **The local console** (FR-56 to FR-58) — Week 8. `tools.confirm` stands in
  for its confirmation half until then.
- `src/speech/speaker_profile.py` (FR-52) and `docs/adr/` — specified in the
  design, not written.

`start.py` only lists services that exist, so its `--help` is the honest
inventory.

---

## Contributing

`.claude/skills/smart-space-code/SKILL.md` holds the rules this codebase is
written to: layer and import discipline, the injected clock, config over
constants, adversarial simulation defaults, and what has to be true before a
change is done. Read it before adding a component.

The short version: components talk only through the blackboard, nothing calls
the clock directly, every tunable is config, and tests ship in the same change
as the code.
