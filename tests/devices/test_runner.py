"""Running a device service as a process (FR-87): announce once connected,
again after the link comes back, and say goodbye on the way out."""

import pytest

from src.common.devices import DeviceAvailability, DeviceKind, DeviceManifest
from src.devices import runner
from src.devices.runner import DeviceParts


class _Transport:
    """A broker link whose state the test decides, tick by tick."""

    def __init__(self, holder, link_states):
        self.holder = holder
        self.link_states = list(link_states)
        self.published = []
        self.will = None

    def connect(self, host, port, keepalive):
        pass

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, qos):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload))

    def advance(self):
        """Move the link to its next state, as paho's callbacks would."""
        if not self.link_states:
            raise KeyboardInterrupt
        up = self.link_states.pop(0)
        board = self.holder[0]
        if up and not board.connected:
            board.on_connected()
        elif not up and board.connected:
            board.on_disconnected()


class _Clock:
    def __init__(self, transport_ref):
        self.transport_ref = transport_ref

    def now(self):
        return 1000.0

    def sleep(self, _seconds):
        self.transport_ref[0].advance()


@pytest.fixture(name="harness")
def _harness(monkeypatch):
    made = []

    def install(link_states, tick=None):
        def build_transport(_config, _client_id, holder, will=None):
            transport = _Transport(holder, link_states)
            transport.will = will
            made.append(transport)
            return transport

        monkeypatch.setattr(runner, "build_transport", build_transport)
        monkeypatch.setattr(runner, "RealClock", lambda: _Clock(made))
        closed = []
        manifest = DeviceManifest(ts=1.0, device_id="widget", name="Widget", kind=DeviceKind.OTHER, simulated=True)
        parts = DeviceParts(manifest=manifest, handlers={}, state=lambda: {}, tick=tick, close=lambda: closed.append(True))
        code = runner.run([], "test", "widget", lambda _config, _clock: parts)
        return code, made[0], closed

    return install


def _availability(transport):
    return [DeviceAvailability.model_validate_json(payload).online for topic, payload in transport.published if topic.endswith("/availability")]


class TestRunner:
    def test_nothing_is_announced_before_the_link_is_up(self, harness):
        _, transport, _ = harness([False, False])
        assert transport.published == []

    def test_announced_once_while_the_link_stays_up(self, harness):
        _, transport, _ = harness([True, True, True])
        assert _availability(transport) == [True, False]  # online, then the goodbye

    def test_announced_again_after_a_reconnect(self, harness):
        _, transport, _ = harness([True, False, True])
        assert _availability(transport) == [True, True, False]

    def test_the_last_will_names_the_device(self, harness):
        _, transport, _ = harness([True])
        assert transport.will[0] == "space/device/widget/availability"

    def test_the_device_is_closed_on_the_way_out(self, harness):
        code, _, closed = harness([True])
        assert closed == [True] and code == 0

    def test_tick_runs_only_while_announced(self, harness):
        ticks = []
        harness([False, True, True], tick=lambda service: ticks.append(service.device_id))
        assert ticks == ["widget", "widget"]


class TestSeveralDevices:
    def _run(self, monkeypatch, device_ids):
        made = []

        def build_transport(_config, client_id, holder, will=None):
            transport = _Transport(holder, [True])
            transport.will, transport.client_id = will, client_id
            made.append(transport)
            return transport

        class _Ticks:
            def now(self):
                return 1000.0

            def sleep(self, _seconds):
                for transport in made:
                    transport.advance()

        monkeypatch.setattr(runner, "build_transport", build_transport)
        monkeypatch.setattr(runner, "RealClock", _Ticks)
        parts = [
            DeviceParts(manifest=DeviceManifest(ts=1.0, device_id=device_id, name=device_id, kind=DeviceKind.OTHER, simulated=True), handlers={}, state=lambda: {})
            for device_id in device_ids
        ]
        return runner.run([], "test", "bridge", lambda _config, _clock: parts), made

    def test_each_device_has_its_own_connection_and_last_will(self, monkeypatch):
        _, made = self._run(monkeypatch, ["lamp", "kettle"])
        assert [t.will[0] for t in made] == ["space/device/lamp/availability", "space/device/kettle/availability"]
        assert [t.client_id for t in made] == ["bridge-lamp", "bridge-kettle"]

    def test_nothing_configured_runs_nothing(self, monkeypatch):
        code, made = self._run(monkeypatch, [])
        assert code == 0 and made == []
