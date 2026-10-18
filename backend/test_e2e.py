import asyncio
import os
import time
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import TranscriptionFrame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, StartFrame, EndFrame, SystemFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport

from voice.bot import run_bot
from conversation_manager import ConversationManager

class MockInput(FrameProcessor):
    def __init__(self, user_text):
        super().__init__()
        self.user_text = user_text
        
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (StartFrame, EndFrame, SystemFrame)):
            await self.push_frame(frame, direction)
            
        if isinstance(frame, StartFrame):
            # When pipeline starts, simulate a user turn
            asyncio.create_task(self._push_user_turn())

    async def _push_user_turn(self):
        await asyncio.sleep(2) # let pipeline init
        logger.info(f"--- MOCK TRANSPORT INJECTING USER TRANSCRIPT: '{self.user_text}' ---")
        self.t0 = time.time()
        # Pipecat aggregators expect TranscriptionFrames from STT
        # We push it downstream.
        await self.push_frame(
            TranscriptionFrame(text=self.user_text, user_id="test", timestamp="now"),
            FrameDirection.DOWNSTREAM
        )
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

class MockOutput(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.t_first_audio = None
        self.audio_frames = 0
        self.text_seen = ""
        
    async def process_frame(self, frame, direction):
        fname = type(frame).__name__
        if fname == "TextFrame":
            self.text_seen += frame.text
            
        if fname in ("TTSAudioFrame", "AudioRawFrame") and self.t_first_audio is None:
            self.t_first_audio = time.time()
            logger.info(f"--- MOCK TRANSPORT RECEIVED FIRST AUDIO (TTS) ---")
            
        if fname == "LLMFullResponseEndFrame":
            logger.info(f"--- MOCK TRANSPORT RECEIVED LLM END. TEXT: {self.text_seen} ---")
            
        await super().process_frame(frame, direction)

class MockTransport(BaseTransport):
    def __init__(self, user_text):
        super().__init__()
        self._input = MockInput(user_text)
        self._output = MockOutput()

    def input(self):
        return self._input

    def output(self):
        return self._output

async def test_e2e(scenario_name, user_text):
    logger.info(f"\n\n{'='*20} SCENARIO: {scenario_name} {'='*20}")
    transport = MockTransport(user_text)
    
    # Run the bot as an async task
    bot_task = asyncio.create_task(
        run_bot(transport=transport, phone_number="+1234567890", call_id=scenario_name)
    )
    
    # We must explicitly trigger the on_client_connected event that run_bot registers
    await asyncio.sleep(0.5)
    for h in transport._event_handlers.get("on_client_connected", []):
        await h(transport, None)
        
    # Wait for the turn to complete
    await asyncio.sleep(12)
    
    t_start = getattr(transport._input, "t0", None)
    t_audio = transport._output.t_first_audio
    if t_start and t_audio:
        logger.info(f"\n[E2E RESULT] Time to first audio: {(t_audio - t_start)*1000:.0f}ms")
    
    bot_task.cancel()

async def main():
    await test_e2e("Deterministic", "Yes.")
    await test_e2e("Fallback", "I guess so, but I'm a little busy right now.")

if __name__ == "__main__":
    asyncio.run(main())
