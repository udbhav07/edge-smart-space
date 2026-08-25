from audio_capture import AudioCapture
from agent_core import SmartAgent

if __name__ == "__main__":
    # 1. Spin up the background microphone thread
    mic_stream = AudioCapture()
    mic_stream.start()

    # 2. Start the main processing engine
    agent = SmartAgent(mic_stream.get_queue())
    agent.run()