"""Sound from a stream rather than a file: decoded ahead, played as it comes (FR-96).

A song found online arrives as an HTTPS stream (Opus or AAC). PyAV decodes it
on a thread of its own into a bounded buffer of PCM, and the sound card reads
from that buffer. The sound card's callback therefore never waits on the
network: if the buffer runs short it plays a moment of silence instead of
stalling, and the track ends only when the stream has ended and the buffer
is empty.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterable

LOGGER = logging.getLogger(__name__)


class StreamUnavailable(ValueError):
    """A stream could not be opened, in words for the occupant."""


class PcmBuffer:
    """Interleaved 16-bit PCM between a decoder thread and the sound card."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._data = bytearray()
        self._done = False
        self._closed = False
        self._condition = threading.Condition()

    def put(self, pcm: bytes) -> bool:
        """Add decoded sound, waiting while the buffer is full. False once closed."""
        with self._condition:
            while len(self._data) >= self._max_bytes and not self._closed:
                self._condition.wait(timeout=0.5)
            if self._closed:
                return False
            self._data.extend(pcm)
            return True

    def finish(self) -> None:
        """The stream has ended; what is buffered still plays."""
        with self._condition:
            self._done = True
            self._condition.notify_all()

    def close(self) -> None:
        """Stop: the decoder gives up and nothing more plays."""
        with self._condition:
            self._closed = True
            self._data.clear()
            self._condition.notify_all()

    def take(self, size: int) -> bytes | None:
        """Exactly ``size`` bytes, padded with silence if the decoder is behind.

        :returns: None once the stream has ended and everything has played.
        """
        with self._condition:
            if self._closed or (self._done and not self._data):
                return None
            chunk = bytes(self._data[:size])
            del self._data[:size]
            self._condition.notify_all()
        return chunk + bytes(size - len(chunk))


def pcm_source(buffer: PcmBuffer, channels: int) -> Generator[bytes, int, None]:
    """A miniaudio-style source over a buffer: sent a frame count, yields that many frames."""
    frames = yield b""
    while True:
        chunk = buffer.take(frames * channels * 2)
        if chunk is None:
            return
        frames = yield chunk


class StreamDecoder:
    """Opens a stream and decodes it into a :class:`PcmBuffer` on its own thread."""

    def __init__(
        self,
        url: str,
        headers: Iterable[tuple[str, str]] = (),
        sample_rate: int = 44100,
        channels: int = 2,
        ahead_s: float = 10.0,
        timeout_s: float = 10.0,
    ) -> None:
        """
        :raises StreamUnavailable: the stream could not be opened or holds no sound.
        """
        import av

        self._av = av
        try:
            options = {"headers": "".join(f"{name}: {value}\r\n" for name, value in headers)} if headers else {}
            self._container = av.open(url, options=options, timeout=timeout_s)
            self._stream = self._container.streams.audio[0]
        except (av.FFmpegError, OSError, IndexError) as exc:
            LOGGER.info("could not open a stream: %s", exc)
            raise StreamUnavailable("That song couldn't be opened; try another.") from exc
        self.buffer = PcmBuffer(int(ahead_s * sample_rate * channels * 2))
        self._resampler = av.AudioResampler(format="s16", layout="stereo" if channels == 2 else "mono", rate=sample_rate)
        self._thread = threading.Thread(target=self._decode, name="music-stream", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.buffer.close()

    def wait(self, timeout_s: float) -> bool:
        """Whether the whole stream has been decoded within ``timeout_s``."""
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _decode(self) -> None:
        try:
            for frame in self._container.decode(self._stream):
                for out in self._resampler.resample(frame):
                    if not self.buffer.put(bytes(out.planes[0])[: out.samples * len(out.layout.channels) * 2]):
                        return
            for out in self._resampler.resample(None):
                self.buffer.put(bytes(out.planes[0])[: out.samples * len(out.layout.channels) * 2])
        except (self._av.FFmpegError, OSError) as exc:
            # A dropped connection ends the song early; it never stops the service.
            LOGGER.warning("the stream broke off: %s", exc)
        finally:
            self.buffer.finish()
            self._container.close()
