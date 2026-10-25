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

The wake-word models and the Whisper model download once, at setup rather
than at startup, because the deployed node has no internet. The speech
service then loads Whisper with network access refused, so skipping this step
fails with a message naming it rather than reaching for the network:

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

### The console: a dashboard in the browser

`python start.py` also starts the console. Open **http://127.0.0.1:8077**.
Each page has its own address, so any of them can be bookmarked:

| Page | What it shows |
|---|---|
| **Overview** `/` | Indoor temperature, humidity, outside, presence, AC power and state, tariff; the temperature against the model's prediction and the setpoint held; faults, services and the latest decisions |
| **Services** `/services` | Each process, judged by what it is publishing against its own cadence: live, late or silent, its rate, and its latest output per topic |
| **Thermal model** `/estimator` | The RLS coefficients and what they mean (time constant, cooling rate, occupant gain), their history, the covariance trace, the one-step prediction and its residual against ±2σ |
| **MQTT board** `/mqtt` | Every topic published since the console started: producer, rate, count, retained or not, the last payload verbatim; and the message stream, pausable and filterable by MQTT pattern |
| **Faults & control** `/faults` | Mode and why, faults with their diagnosis, sensor health, goal and setpoint held, operator fault injection and reset, and the full decision log |
| **Chat** `/chat` | Talk to Jarvis by typing, for when you would rather not speak aloud; each answer shows the steps taken to reach it |
| **Calendar** `/calendar` | The week ahead; add and remove entries |
| **Bookings** `/bookings` | Requests waiting for your yes, a booking form, and what has been booked |
| **Mail** `/mail` | An inbox of what the room tells you; sending goes to a mock outbox after your yes, because this node never connects to the internet |

`Ctrl-K` opens every action from anywhere. The operator's fault injection
and reset can be turned off with `console.operator_controls: false`.

It is pushed live over a WebSocket and falls back to polling if the socket
drops. It has no authority: typed words go through the same extraction as
speech, a comfort setting is a preference the validator gates, and nothing
it does can reach the actuator (FR-56, FR-57). Stop it, or never start it,
and the room runs exactly the same.

It listens on this machine only. To open it to phones on the same network,
set `console.host` to this machine's LAN address; it never reaches beyond the
local network, and its page loads nothing from the internet (NFR-06).

The page is built ahead of time into `src/console/static`, so running it
needs no Node. To change the frontend (`console/web`, React and Vite):

```bash
cd console/web
npm install
npm run dev     # http://localhost:5173, proxied to the running console
npm run build   # rebuilds src/console/static
```

### Running one component directly

```bash
python -m sim.run_sim --steps 100     # room plant, sensors, actuator
python -m src.estimation              # online RC identification
python -m src.control                 # regulatory loop and safety gate
python -m src.faults                  # fault detector bank (D1-D5) and mode
python -m src.speech                  # wake word, transcription, Personal Context
python -m src.reasoning               # supervisor, assistant, fault diagnosis
python -m src.assistance              # tool executor: local calendar, mock travel
python -m src.io                      # the real devices, when devices.source is esphome
```

Both take `--config` and both stop cleanly on `Ctrl-C`.

### Talking to it without a microphone

```bash
python -m tools.say "make it 22 degrees"
python -m tools.say "put a meeting with Ravi on Thursday at 3"
python -m tools.say "what's on my calendar on Thursday"
python -m tools.say "book me a flight to Delhi next Monday"   # asks first
python -m tools.say                                          # a conversation
```

A stand-in for a voice, not a change to one: it runs the same Personal
Context call the speech pipeline runs after Whisper and publishes the same
hint to the same topic, so nothing downstream can tell a typed sentence from
a spoken one. It prints what the system said back on `space/context/reply`,
and when a booking needs confirming it plays the console's part. Needs the
`assistance` and `reasoning` services and the inference server running.

### Who is speaking, and what the room remembers

The room can tell household members apart by voice and remember things each
of them asks it to — which only ever personalises its replies. Recognising a
voice unlocks nothing: a booking still waits for confirmation, every setpoint
still passes the gate, and the first reply of a conversation says so (FR-52).

```bash
python -m tools.enroll ravi --name Ravi one.wav two.wav three.wav   # or --record 3
python -m tools.say --as ravi "remember that I prefer meetings after ten"
python -m tools.say --as ravi "when should we put the review?"
python -m tools.profiles list
python -m tools.profiles show ravi
python -m tools.profiles delete ravi        # memory, conversation and voiceprint
```

Three things are kept per person, each in a file you can open:

| What | Where | Lasts |
|---|---|---|
| Long-term memory | `state/profiles/<id>.md` — plain Markdown; edit it by hand | until forgotten |
| Short-term context | `state/context/<id>.json` — the conversation so far | 30 minutes |
| Voiceprint | `state/voiceprints.json` — numbers only, never audio | until deleted |

A voice nobody enrolled is a *guest*: it gets the room's short-term context and
no long-term memory. The recogniser is deliberately simple (numpy only) and
has been measured on synthetic voices, not yet on real ones; DESIGN.md section
5.8.3 has the numbers.

### Always listening: what it overhears

The room listens continuously (`speech.listen_mode: continuous`, the default).
Anything said with "Hey Jarvis" is a request, as before. Anything else is
*overheard*: transcribed on this machine, never answered, and read for two
things only (FR-76 to FR-78):

- **A plan** — a meeting, an appointment, a trip — is pencilled straight into
  the calendar, hatched and marked **model tentative**, with the words heard.
  Keep it on the Calendar page to make it firm, or discard it. A trip is
  pencilled in, never booked.
- **A fact** about a recognised voice is offered on the Chat page and kept
  only if that person says Keep. Until then it sits under *Suggested
  (overheard)* in their profile file, apart from what is remembered.

```bash
python -m tools.overhear --as ravi "I've got the dentist on Thursday at four"
python -m tools.overhear "we're flying to Goa next Friday"
python -m tools.overhear --as ravi "I'm allergic to peanuts, so no satay for me"
```

`tools.overhear` publishes exactly what the microphone would, and prints what
came of it. Audio is never kept or published, overheard words are not retained
on the blackboard, and `listen_mode: wake_word` restores capture after the wake
word only.

### Watching what happens

Every decision in the system is on the blackboard. Nothing is hidden in
process memory, so a debugger is never needed to see what it did. The
project's own viewer works the same on every platform:

```bash
python -m tools.blackboard_view            # one screen: sensors, unit, mode, faults, diagnoses
python -m tools.blackboard_view --stream   # one line per message, flushed, safe to pipe
```

The summary lists every configured sensor, a silent one first; shows the
setpoint the room is actually being held to beside the one asked for, and why
they differ (peak tariff); and keeps the latest explanation of each active
fault, so it is useful even when opened after the fact. Its last line names
any declared topic nothing has published to, which answers "is that service
actually running?" without guesswork.

`mosquitto_sub` shows the raw messages:

```bash
mosquitto_sub -t 'space/#' -v
mosquitto_sub -t 'space/sensor/+/state' -v      # raw readings
mosquitto_sub -t 'space/audit/validation' -v    # what the safety gate decided
mosquitto_sub -t 'space/system/mode' -v         # degradation state
```

On Windows the installer leaves `mosquitto_sub` off the PATH, and cmd and
PowerShell do not treat single quotes as quoting. In PowerShell:

```powershell
& "C:\Program Files\mosquitto\mosquitto_sub.exe" -t "space/#" -v
python -m tools.blackboard_view --stream | Select-String fault
```

A clamped setpoint appearing in `space/audit/validation` is the gate working,
not a failure. Every verdict carries the proposal, the reason code, and the
value actually applied.

### Seeing the whole thing work, without a broker

```bash
python -m tools.demo                      # a sensor breaks, the loop survives
python -m tools.demo --scenario actuator  # the air conditioner stops cooling
```

Runs all four services in one process against a simulated clock, so half an
hour of room time takes a couple of seconds, and narrates what happens. It is
the quickest way to see the system work, and the fallback if a live
demonstration fails.

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
python -m tools.inject ac dead               # the unit accepts commands and cools nothing
python -m tools.inject ac clear
```

Then watch what the detectors make of it, with `python -m tools.blackboard_view`
or the topics themselves:

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
measurement, and the mode becomes `SAFE_HOLD`. Entering it switches the
compressor off — with nothing trusted to say when to stop cooling, off is the
one safe state — and the room drifts toward ambient until someone looks.

Only the indoor temperature sensor and the unit decide the mode. Break the
humidity sensor or the power meter (`python -m tools.inject hum_01 dropout`)
and the fault is published with its evidence as *advisory*: the mode stays
`NORMAL`, because temperature control does not depend on it.

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
sensor five to six minutes — a whole five-minute variance window of frozen
readings, then the next detector tick — and an actuator fault ten to twelve
minutes: ten minutes of sustained cooling that achieves nothing, counted from
when the unit should have started. Those are the times measured on a live run,
not the window lengths alone.
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

Point any entry point at a different file with `--config`. A file may
`extends:` another and override only what it names, which is how a
calibration and a node's own settings stay out of the reference
(DESIGN.md section 9.4).

A few worth knowing:

| Setting | Default | Note |
|---|---|---|
| `speech.asr_device` | `auto` | Prefers CUDA, falls back to CPU. `cuda` demands it and fails loudly; `cpu` forces it |
| `validator.setpoint_bounds_c` | 18.0–30.0 | Hard limits nothing can propose past |
| `controller.min_off_s` | 180 | Compressor protection, enforced twice on purpose |
| `sim.*` | all imperfections **on** | Jitter, dropouts, noise, dead-time, command loss |
| `site.utc_offset_h` | 5.5 | The room's local time. Never the machine's zone, so a run replays the same anywhere |
| `sim.room.cooling_power_w` | 3500 | A 1-ton split. At 2500 W a healthy system could not hold its own room |

The simulator defaults are adversarial deliberately. A kind simulator makes
simulated success predict nothing about real hardware.

---

## Layout

```
src/common/      clock, local time, config, schemas, topics, blackboard, tools
src/control/     safety validator, goal path, regulatory loop  (python -m src.control)
src/estimation/  RC model, RLS, persistence                    (python -m src.estimation)
src/faults/      detector bank, aggregator, mode manager       (python -m src.faults)
src/speech/      wake word, capture, ASR, pipeline             (python -m src.speech)
src/reasoning/   supervisor, assistant, fault diagnosis        (python -m src.reasoning)
src/assistance/  tool executor, local calendar, mock travel    (python -m src.assistance)
src/io/          the real devices: ESPHome bridge              (python -m src.io)
sim/             ground-truth room model, sensors, actuator, power meter, replay
eval/            baseline thermostat, harness, metrics, experiments E1-E6
tools/           inject, reset, say, confirm, record, bringup, calibrate, demo
deploy/          systemd units, provisioning, the ESPHome node definition
tests/           unit tests, mirroring the source layout
config/          default.yaml, the documented reference
docs/            DESIGN.md, ROADMAP.md, coding-guidelines.md
```

`sim/room_model.py` must never import from `src/estimation/`. The simulated
plant and the estimator's model have to be parameterised independently, or the
evaluation degenerates into the model predicting itself.

---

## Recording and replaying

```bash
python -m tools.record runs/demo.jsonl       # everything on space/#, verbatim
python -m sim.replay runs/demo.jsonl --speed 10
```

A replay publishes each message on its original topic, in order and in time,
taking retain and QoS from the topic table. Run it against a separate broker
or restrict it with `--only`: replayed onto a live system it feeds it the past.

## Experiments

```bash
python -m eval.experiments.e1_convergence    # does identification converge?
python -m eval.experiments.e2_adaptation     # adapting against a frozen model
python -m eval.experiments.e3_detection      # every fault class, found by its own detector
python -m eval.experiments.e4_degradation    # how long control on prediction stays viable
python -m eval.experiments.e5_baseline       # the same faults, injected into a thermostat
python -m eval.experiments.e6_tool_selection # the supervisor against the served model
```

All but E6 run against the simulator in seconds to minutes, and every one
takes `--config`. The results as of DESIGN.md v1.11 are in its section 8.5;
the one to know is E5: with the indoor sensor dropped, this system keeps the
room inside the comfort band for the whole half hour and the thermostat is
outside it 69 % of the time.

## Bringing up the hardware, and calibrating it

```bash
python -m tools.bringup                      # is every sensor publishing, at its rate?
python -m tools.bringup --actuate            # does the unit draw power when asked to cool?
python -m tools.record runs/healthy.jsonl --seconds 7200
python -m tools.calibrate runs/healthy.jsonl --out config/calibration.yaml
```

`tools.calibrate` re-derives the D1, D2 and D4 thresholds from a recorded
healthy run (R-04) and writes them as an overlay that states, beside every
value, what it was measured from. On the Orin the overlay is the middle of
three configuration layers; `deploy/README.md` has the commands.

Until the local console exists, `python -m tools.confirm` answers the room's
booking questions from a terminal.

---

## Not built yet

Honest about the gaps, so nobody hunts for something that isn't there:

- hardware: nothing has run on the Jetson or the ESPHome nodes yet. The
  bridge (`src/io/devices.py`) and the systemd units (`deploy/`) exist and are
  tested against simulated device messages; the node topics in config are
  placeholders until the nodes are flashed
- standing prompts in the console (FR-57): the comfort setting is built,
  saved instructions to the supervisor are not
- speaker recognition on real voices: it is built and measured on synthetic
  ones only (FR-52)
- experiment E7 (latency and memory on the Jetson) — it needs the board
- drift slower than about 0.3 C/min is not separable from a weak unit with
  one indoor sensor, and at 0.6 C/min D5 names the unit first in 2 of 10
  trials; DESIGN.md sections 5.5 and 8.5 have the measurements
- scheduled excitation (R-01) — under ordinary closed-loop control the model
  is poor for its first hour, which limits how long control on prediction
  stays accurate early in a run

`start.py --help` lists the services that exist.

---

## Contributing

`.claude/skills/smart-space-code/SKILL.md` holds the rules this codebase is
written to: layer and import discipline, the injected clock, config over
constants, adversarial simulation defaults, and what has to be true before a
change is done. Read it before adding a component.

The short version: components talk only through the blackboard, nothing calls
the clock directly, every tunable is config, and tests ship in the same change
as the code.
