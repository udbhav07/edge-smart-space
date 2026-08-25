import pyaudio

# Audio Settings
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 512 # Required size for Silero VAD
DEVICE_INDEX = None # Update if needed

# Agent Settings
WAKE_WORD_THRESHOLD = 0.1
VAD_SILENCE_TIMEOUT_MS = 1000  # Wait 1 second of silence before ending command
COMMAND_TIMEOUT_SEC = 15       # Max duration user can speak
NO_SPEECH_TIMEOUT_SEC = 3      # Timeout if wake word heard but no one speaks