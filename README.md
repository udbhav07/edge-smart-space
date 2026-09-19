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

**Inference server** (only needed for voice and reasoning)

```bash
ollama serve
ollama pull qwen2.5:7b
```

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
python -m src.faults                  # fault detector bank (D1-D3)
python -m src.speech                  # wake word, transcription, reasoning
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
src/common/      clock, config, schemas, topics, MQTT blackboard, device
src/control/     safety validator, regulatory controller
src/estimation/  RC model, RLS, persistence          (python -m src.estimation)
src/faults/      detector bank, aggregator           (python -m src.faults)
src/speech/      wake word, capture, ASR, pipeline  (python -m src.speech)
src/reasoning/   single-shot LLM calls
src/io/          actuator driver contracts
sim/             ground-truth room model, sensors, actuator, runner
tests/           unit tests, mirroring the source layout
config/          default.yaml
docs/            DESIGN.md, coding-guidelines.md
```

`sim/room_model.py` must never import from `src/estimation/`. The simulated
plant and the estimator's model have to be parameterised independently, or the
evaluation degenerates into the model predicting itself.

---

## Not built yet

Honest about the gaps, so nobody hunts for something that isn't there:

- `src/faults/detectors/` — D4 drift and D5 actuator-response are not written
- `src/faults/mode_manager.py` — degraded modes and control on prediction
- `deploy/systemd/` — the unit files that supervise this on the Jetson
- `eval/` — the baseline thermostat and the experiment harness

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
