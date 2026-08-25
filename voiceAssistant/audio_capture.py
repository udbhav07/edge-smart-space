import pyaudio
import numpy as np
import threading
import queue
import config

class AudioCapture:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.mic = self.p.open(
            format=config.FORMAT,
            channels=config.CHANNELS,
            rate=config.RATE,
            input=True,
            input_device_index=config.DEVICE_INDEX,
            frames_per_buffer=config.CHUNK,
        )
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.thread = threading.Thread(target=self._record_loop, daemon=True)

    def start(self):
        self.is_running = True
        self.thread.start()

    def _record_loop(self):
        while self.is_running:
            try:
                # exception_on_overflow=False prevents crashes if the queue backs up
                data = self.mic.read(config.CHUNK, exception_on_overflow=False)
                frame = np.frombuffer(data, dtype=np.int16)
                self.audio_queue.put(frame)
            except Exception as e:
                print(f"Audio read error: {e}")

    def get_queue(self):
        return self.audio_queue