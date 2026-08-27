import pyaudio
import numpy as np
import threading
import queue
import time
from . import config

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
        
        # FIX 1: Bounded queue (e.g., max 5 seconds of audio chunks)
        # Prevents unbounded RAM growth if Whisper stalls.
        max_queue_size = int((config.RATE / config.CHUNK) * 5)
        self.audio_queue = queue.Queue(maxsize=max_queue_size)
        
        self.is_running = False
        self.thread = threading.Thread(target=self._record_loop, daemon=True)

    def start(self):
        self.is_running = True
        self.thread.start()

    def stop(self):
        """FIX 2: Real stop method to clean up resources and release the mic."""
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        try:
            self.mic.stop_stream()
            self.mic.close()
        except Exception:
            pass
        
        try:
            self.p.terminate()
        except Exception:
            pass
        print("[AudioCapture] Microphone stream and PyAudio terminated safely.")

    def _record_loop(self):
        consecutive_errors = 0
        max_consecutive_errors = 10  # FIX 3: Bailout threshold

        while self.is_running:
            try:
                # exception_on_overflow=False prevents crashes if read lags slightly
                data = self.mic.read(config.CHUNK, exception_on_overflow=False)
                frame = np.frombuffer(data, dtype=np.int16)
                
                # FIX 1: Backpressure / drop oldest frame if queue is full
                if self.audio_queue.full():
                    try:
                        self.audio_queue.get_nowait()  # Drop oldest frame
                    except queue.Empty:
                        pass
                
                self.audio_queue.put(frame)
                consecutive_errors = 0  # Reset error count on success
                
            except (IOError, OSError) as e:
                # FIX 3: Catch specific PyAudio errors (e.g., mic unplugged or overflow)
                consecutive_errors += 1
                print(f"[AudioCapture Error] Stream read failure ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("[AudioCapture Fatal] Too many consecutive mic failures. Stopping capture loop.")
                    self.is_running = False
                    break
                
                # Backoff briefly to avoid spinning at 100% CPU on hardware failure
                time.sleep(0.1)
            except Exception as e:
                # Catch unexpected bugs (like AttributeErrors) instead of hiding them
                print(f"[AudioCapture Bug] Unexpected critical error: {e}")
                self.is_running = False
                raise e

    def get_queue(self):
        return self.audio_queue