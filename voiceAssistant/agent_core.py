import numpy as np
import torch
import openwakeword
from openwakeword.model import Model
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel
from . import config
from .llm_engine import SmartAgentLLM

class SmartAgent:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.state = "WAKE_WORD"
        self.command_buffer = []
        self.chunks_since_wake = 0
        self.has_spoken = False

        print("Loading OpenWakeWord...")
        # self.oww_model = Model(wakeword_models=["hey_jarvis"])
        self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

        print("Loading Silero VAD...")
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.vad_model,
            sampling_rate=config.RATE,
            min_silence_duration_ms=config.VAD_SILENCE_TIMEOUT_MS
        )

        print(f"Loading Faster-Whisper ({config.STT_DEVICE})...")
        self.stt_model = WhisperModel(
            "base.en",
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )

        print("Connecting to LLM Engine...")
        self.llm = SmartAgentLLM()
        
        print("\n[System Ready] Agent is online.")
        print("Listening for Hey Jarvis...\n")

    def run(self):
        while True:
            # Block until the background thread gives us an audio chunk
            frame_int16 = self.audio_queue.get()
            
            if self.state == "WAKE_WORD":
                score = self.oww_model.predict(frame_int16).get("hey_jarvis", 0)
                if score > config.WAKE_WORD_THRESHOLD:
                    print("\n[!] Wake word detected! Listening for command...")
                    self.state = "LISTENING"
                    self.oww_model.reset()
                    self.vad_iterator.reset_states()
                    self.command_buffer = []
                    self.chunks_since_wake = 0
                    self.has_spoken = False
                    
            elif self.state == "LISTENING":
                self.command_buffer.append(frame_int16)
                self.chunks_since_wake += 1

                # Silero VAD strictly requires float32 tensors between -1 and 1
                frame_float = frame_int16.astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_float)

                speech_dict = self.vad_iterator(frame_tensor, return_seconds=False)

                if speech_dict:
                    if 'start' in speech_dict:
                        self.has_spoken = True
                        print(" -> User started speaking...")
                    elif 'end' in speech_dict:
                        print(" -> User finished speaking! Processing...")
                        self.transcribe_command()
                        self.reset_state()

                # Safety Timeouts
                timeout_chunks = int(config.NO_SPEECH_TIMEOUT_SEC * config.RATE / config.CHUNK)
                max_chunks = int(config.COMMAND_TIMEOUT_SEC * config.RATE / config.CHUNK)

                if not self.has_spoken and self.chunks_since_wake > timeout_chunks:
                    print(" -> [Timeout] No speech detected.")
                    self.reset_state()
                elif self.has_spoken and self.chunks_since_wake > max_chunks:
                    print(" -> [Timeout] Max command length reached.")
                    self.transcribe_command()
                    self.reset_state()

    def transcribe_command(self):
        if not self.command_buffer:
            return

        audio_np = np.concatenate(self.command_buffer).astype(np.float32) / 32768.0
        segments, info = self.stt_model.transcribe(audio_np, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        print(f"\n[JARVIS HEARD]: {text}")
        
        # Send text to the LLM and get the AI's response
        if text:
            ai_response = self.llm.chat(text)
            print(f"[JARVIS SAYS]: {ai_response}\n")

    def reset_state(self):
        # Flush the queue to discard ambient noise collected while STT was running
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.state = "WAKE_WORD"
        print("Listening for Hey Jarvis...")