"""Short device commands, understood without a model (FR-97).

"Stop", "next song", "carry on", "turn it up", "turn off the fan": said to a
room with a 7B model, each took two model calls -- reading the request, then
choosing the tool -- 15 to 20 s for something a remote does at once. These
are recognised here from the words alone, and the call they name rides on
the hint; the assistant runs it through the executor exactly as a remote's
tap is run, with the same argument checks and gates, and says the device's
own answer.

Only short utterances whose whole meaning is the command are recognised:
the device named or, unnamed, the one that is playing; the verb from a fixed
list; and nothing else left over. Anything more -- "stop the music in ten
minutes", "play something by Queen" -- goes to the model as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.common.devices import DeviceKind
from src.reasoning.devices import KnownDevices

#: Longest utterance, in words, still taken as a bare command.
_MAX_WORDS = 8
#: Said around a command without changing it.
_AROUND = re.compile(r"^(hey |ok |okay )?jarvis\b|\b(please|now|for me|thanks|thank you)\b|^(can|could|would|will) you\b")
#: Words that name nothing but the thing already meant.
_IDLE = frozenset("the it this that one music song songs track a bit little playing".split())
#: How much "louder" or "quieter" moves the volume, in percent.
VOLUME_STEP = 15

#: What is left once the device and idle words are gone, and the capability it means.
_VERBS: dict[str, tuple[str, int]] = {
    **{phrase: ("pause", 0) for phrase in ("pause", "hold", "hold on")},
    **{phrase: ("stop", 0) for phrase in ("stop", "stop playing")},
    **{phrase: ("resume", 0) for phrase in ("resume", "continue", "carry on", "keep going", "keep playing", "unpause", "play again")},
    **{phrase: ("next", 0) for phrase in ("next", "skip", "play next", "next one", "skip ahead")},
    **{phrase: ("set_volume", VOLUME_STEP) for phrase in ("louder", "turn up", "volume up", "turn volume up", "increase volume", "raise volume")},
    **{phrase: ("set_volume", -VOLUME_STEP) for phrase in ("quieter", "softer", "turn down", "volume down", "turn volume down", "lower volume", "decrease volume", "reduce volume")},
    **{phrase: ("turn_on", 0) for phrase in ("turn on", "switch on", "on")},
    **{phrase: ("turn_off", 0) for phrase in ("turn off", "switch off", "off")},
}
#: "Volume 30", "set the volume to 30", "turn it up to 80": a level, with a word that makes it one.
_VOLUME_TO = re.compile(r"^(?=.*\b(volume|turn|set)\b)(set |turn |put )?(volume )?(up |down )?(to |at )?(\d{1,3})( percent| %)?$")


@dataclass(frozen=True)
class QuickCommand:
    device_id: str
    capability: str
    arguments: dict


def quick_command(transcript: str, devices: KnownDevices) -> QuickCommand | None:
    """The device call a short utterance plainly asks for, or None to ask the model."""
    text = re.sub(r"[^a-z0-9% ]+", " ", transcript.lower())
    text = " ".join(_AROUND.sub(" ", " ".join(text.split())).split())
    if not text or len(text.split()) > _MAX_WORDS:
        return None
    device_id, text = _named(text, devices)
    rest = " ".join(word for word in text.split() if word not in _IDLE)
    volume_to = _VOLUME_TO.match(rest)
    if volume_to:
        capability, step, level = "set_volume", 0, min(int(volume_to.group(6)), 100)
    elif rest in _VERBS:
        (capability, step), level = _VERBS[rest], None
    else:
        return None
    device_id = device_id or _the_one_meant(devices, capability)
    if device_id is None or not devices.can(device_id, capability):
        return None
    if capability == "set_volume":
        if level is None:
            current = devices.state_of(device_id).get("volume")
            if not isinstance(current, (int, float)):
                return None  # "louder" needs to know how loud it is now
            level = max(0, min(100, int(current) + step))
        return QuickCommand(device_id, capability, {"volume": level})
    return QuickCommand(device_id, capability, {})


def _named(text: str, devices: KnownDevices) -> tuple[str | None, str]:
    """The device the words name, and the words with its name taken out."""
    best: tuple[int, str, str] | None = None
    for manifest in devices.operable():
        for name in (manifest.device_id.replace("_", " "), manifest.name, *manifest.aliases):
            name = name.lower().strip()
            if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text):
                if best is None or len(name) > best[0]:
                    best = (len(name), manifest.device_id, name)
    if best is None:
        return None, text
    return best[1], " ".join(re.sub(rf"(?<![a-z0-9]){re.escape(best[2])}(?![a-z0-9])", " ", text).split())


def _the_one_meant(devices: KnownDevices, capability: str) -> str | None:
    """Unnamed, a command is for the one media device playing or paused, if there is one.

    Otherwise it is the model's to work out: "stop" with no music on may be
    about the booking just offered, not the speaker.
    """
    if capability in ("turn_on", "turn_off"):
        return None
    active = [
        manifest.device_id
        for manifest in devices.operable()
        if manifest.kind is DeviceKind.MEDIA
        and manifest.capability(capability) is not None
        and (devices.state_of(manifest.device_id).get("playing") or devices.state_of(manifest.device_id).get("paused"))
    ]
    return active[0] if len(active) == 1 else None
