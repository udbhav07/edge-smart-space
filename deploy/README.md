# Deploying on the Jetson

One systemd unit per service, `Restart=always` (NFR-08, section 9.3), grouped
under `smart-space.target`. The units assume the repository at
`/opt/edge-smart-space`, a virtual environment in `.venv` inside it, and a
`smartspace` user.

```bash
sudo cp deploy/systemd/*.service deploy/systemd/*.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now smart-space.target
```

Layer 1 is the simulator or the real devices, chosen by `devices.source` in
the configuration. On hardware, set `devices.source: esphome`, fill in the
nodes' topics, and swap the simulator for the bridge:

```bash
sudo systemctl disable --now smart-space-simulator.service
sudo systemctl enable --now smart-space-devices.service
```

Watch everything with `journalctl -fu 'smart-space-*'` and the blackboard with
`mosquitto_sub -t 'space/#' -v`. Killing any one service -- `sudo systemctl
kill smart-space-reasoning` -- leaves the others running and systemd brings it
back within seconds; the regulatory loop never notices.
