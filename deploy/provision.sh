#!/usr/bin/env bash
# Provision a Jetson AGX Orin (JetPack 6.x) to run the whole stack on boot.
#
#   sudo deploy/provision.sh --dry-run                 # print every step, change nothing
#   sudo deploy/provision.sh                           # simulator as Layer 1
#   sudo deploy/provision.sh --source esphome          # the real sensors and AC
#   sudo deploy/provision.sh --model /path/to/qwen2.5-7b-instruct-q4_k_m.gguf
#
# Idempotent: every step checks before it changes anything, so running it
# again after a JetPack update or a git pull converges rather than duplicates.
#
# What it does, in order (DESIGN.md sections 4.5, 9.2, 9.3):
#   1. checks this is a Jetson and says so if it is not
#   2. MAXN power mode and jetson_clocks on every boot (section 9.2: E7's
#      numbers are not reproducible otherwise)
#   3. the room's time zone, so the machine's clock agrees with site.utc_offset_h
#   4. a 'space' service user, in the audio and dialout groups
#   5. Python 3.11+ and a virtualenv at /opt/edge-smart-space/.venv
#   6. the broker, as the container deploy/docker-compose.yml describes
#   7. llama.cpp built with CUDA, and the model it serves
#   8. every systemd unit, with space.target and llama-server enabled at boot
#   9. Layer 1 selected in the deployed config: simulated or esphome
#  10. the preflight: python start.py --check
#
# What it cannot do, and says so at the end: flash an ESP32, wire a sensor, or
# decide the IR protocol. Those are bring-up steps with a person holding the
# board, checked afterwards with python -m tools.bringup.
#
# Network: provisioning is a setup-time step and fetches packages and the
# model. The running system makes no outbound connection (NFR-06); nothing
# installed here is started with a reason to.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="/opt/edge-smart-space"
LLAMA_DIR="/opt/llama.cpp"
MODEL_DIR="/opt/models"
ENV_DIR="/etc/edge-smart-space"
SERVICE_USER="space"
TIMEZONE="Asia/Kolkata"          # site.utc_offset_h: 5.5 in config/default.yaml
LLM_PORT="11434"                 # reasoning.base_url in config/default.yaml
LLAMA_CPP_REF="b4600"            # pinned: a build is only reproducible from a tag
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"

SOURCE="simulated"
MODEL_PATH=""
DRY_RUN=0
WITH_SPEECH=0

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --with-speech) WITH_SPEECH=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

case "$SOURCE" in
    simulated|esphome) ;;
    *) echo "--source must be simulated or esphome, got $SOURCE" >&2; exit 2 ;;
esac

step() { printf '\n==> %s\n' "$*"; }
run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '    [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

if [[ "$DRY_RUN" -eq 0 && "$(id -u)" -ne 0 ]]; then
    echo "run as root (sudo), or with --dry-run to see what it would do" >&2
    exit 1
fi

# --- 1. platform -------------------------------------------------------------
step "Checking the platform"
if [[ -f /etc/nv_tegra_release ]]; then
    echo "    Jetson: $(head -n 1 /etc/nv_tegra_release)"
    IS_JETSON=1
else
    echo "    not a Jetson (no /etc/nv_tegra_release): skipping power mode and"
    echo "    building llama.cpp for whatever CUDA is here, if any"
    IS_JETSON=0
fi

# --- 2. power mode (section 9.2) ---------------------------------------------
if [[ "$IS_JETSON" -eq 1 ]]; then
    step "MAXN power mode, and jetson_clocks on every boot"
    run nvpmodel -m 0
    if [[ ! -f /etc/systemd/system/jetson-clocks.service ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "    [dry-run] write /etc/systemd/system/jetson-clocks.service"
        else
            cat > /etc/systemd/system/jetson-clocks.service <<'UNIT'
[Unit]
Description=Pin Jetson clocks for reproducible latency (DESIGN.md section 9.2)
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
        fi
    fi
    run systemctl enable jetson-clocks.service
fi

# --- 3. time zone ------------------------------------------------------------
step "Time zone $TIMEZONE, to agree with site.utc_offset_h"
run timedatectl set-timezone "$TIMEZONE"

# --- 4. service user ---------------------------------------------------------
step "Service user '$SERVICE_USER'"
if id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "    exists"
else
    run useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi
run usermod -aG audio,dialout "$SERVICE_USER"

# --- 5. python and the code --------------------------------------------------
step "Python 3.11+ and the code at $INSTALL_DIR"
PYTHON=""
for candidate in python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done
if [[ -z "$PYTHON" ]]; then
    # JetPack 6 ships 3.10, and numpy 2.3 needs 3.11 (pyproject.toml).
    run apt-get update
    run apt-get install -y software-properties-common
    run add-apt-repository -y ppa:deadsnakes/ppa
    run apt-get update
    run apt-get install -y python3.11 python3.11-venv python3.11-dev
    PYTHON="python3.11"
fi
run apt-get install -y git cmake build-essential mosquitto-clients portaudio19-dev
run mkdir -p "$INSTALL_DIR"
run rsync -a --delete --exclude .git --exclude .venv --exclude state \
    --exclude __pycache__ "$REPO_DIR/" "$INSTALL_DIR/"
run mkdir -p "$INSTALL_DIR/state"
if [[ ! -x "$INSTALL_DIR/.venv/bin/python" ]]; then
    run "$PYTHON" -m venv "$INSTALL_DIR/.venv"
fi
run "$INSTALL_DIR/.venv/bin/pip" install --upgrade pip
if [[ "$WITH_SPEECH" -eq 1 ]]; then
    # The speech extra pins torch. On a Jetson, PyPI's torch is CPU-only;
    # replace it with NVIDIA's JetPack wheel afterwards or ASR runs on the CPU
    # (section 5.8.2 reports which one it got).
    run "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR[speech]"
    run sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/setup_models.py"
else
    run "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR"
fi
run chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# --- 6. the broker -----------------------------------------------------------
step "The broker (deploy/docker-compose.yml)"
run docker compose -f "$INSTALL_DIR/deploy/docker-compose.yml" up -d

# --- 7. the model server -----------------------------------------------------
step "llama.cpp with CUDA, and the model"
if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        run git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi
    run git -C "$LLAMA_DIR" fetch --tags
    run git -C "$LLAMA_DIR" checkout "$LLAMA_CPP_REF"
    run cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    run cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-server
else
    echo "    llama-server already built"
fi
run mkdir -p "$MODEL_DIR" "$ENV_DIR"
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$MODEL_DIR/$(basename "$MODEL_URL")"
    if [[ ! -f "$MODEL_PATH" ]]; then
        run curl -L --fail -o "$MODEL_PATH" "$MODEL_URL"
    fi
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    [dry-run] write $ENV_DIR/llm.env: LLM_MODEL_PATH=$MODEL_PATH LLM_PORT=$LLM_PORT"
else
    printf 'LLM_MODEL_PATH=%s\nLLM_PORT=%s\n' "$MODEL_PATH" "$LLM_PORT" > "$ENV_DIR/llm.env"
fi

# --- 8. systemd --------------------------------------------------------------
step "systemd units, enabled at boot"
run cp "$INSTALL_DIR"/deploy/systemd/*.service "$INSTALL_DIR"/deploy/systemd/*.target \
    /etc/systemd/system/
run systemctl daemon-reload
run systemctl enable llama-server.service
run systemctl enable space.target

# --- 9. Layer 1 --------------------------------------------------------------
step "Layer 1: $SOURCE"
CONFIG="$INSTALL_DIR/config/default.yaml"
run sed -i "s/^  source: \(simulated\|esphome\)/  source: $SOURCE/" "$CONFIG"
if [[ "$SOURCE" == "simulated" ]]; then
    # space.target wants the hardware bridge; on a board with no sensors yet,
    # the simulator stands in for it on the same topics (section 9.1).
    run systemctl disable space-layer1@space.service || true
    run systemctl enable space-simulator@space.service
else
    run systemctl disable space-simulator@space.service || true
    run systemctl enable space-layer1@space.service
fi

# --- 10. start and check -----------------------------------------------------
step "Starting, then the preflight"
run systemctl restart llama-server.service
run systemctl restart space.target
run sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/start.py" \
    --check --config "$CONFIG"

cat <<DONE

Provisioned. Everything above comes back by itself after a reboot.

  Watch it:            $INSTALL_DIR/.venv/bin/python -m tools.blackboard_view
  GPU offload:         journalctl -u llama-server | grep -i offloaded
  Talk to it:          $INSTALL_DIR/.venv/bin/python -m tools.say "it is too warm in here"
  Hardware bring-up:   $INSTALL_DIR/.venv/bin/python -m tools.bringup --actuate

Not done here, because it needs a person holding the board:
  flash deploy/esphome/room-node.yaml to the ESP32, settle its PLACEHOLDER pins,
  one-wire address, meter address and IR protocol, then run tools.bringup.
DONE
