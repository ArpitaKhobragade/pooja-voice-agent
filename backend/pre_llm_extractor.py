import time
from loguru import logger
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, LLMContextFrame

from conversation_manager import ConversationManager
from perf_logger import CallPerformanceLogger
from extractors import extract_slot
from prompts import build_dynamic_block

class PreLLMStateExtractor(FrameProcessor):
    """
    Intercepts LLMContextFrame right before it hits the LLM.
    Reads the latest user utterance and attempts to deterministically extract the CURRENT expected slot.
    If successful, updates the ConversationManager and context.messages[1] in-flight,
    allowing the LLM to completely bypass the tool-calling Gen #1.
    """
    def __init__(
        self,
        cm: ConversationManager,
        perf: CallPerformanceLogger,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cm = cm
        self.perf = perf
        self._original_tools = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        frame_name = type(frame).__name__
        logger.debug(f"[PreLLM-Trace] IN  {direction.name:10} {frame_name}")
        
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._handle_llm_context(frame)
            
        logger.debug(f"[PreLLM-Trace] OUT {direction.name:10} {frame_name}")
        await self.push_frame(frame, direction)

    async def _handle_llm_context(self, frame: LLMContextFrame):
        # 1. PRUNE CONVERSATION HISTORY
        if frame.context and frame.context.messages:
            system_msgs = [m for m in frame.context.messages if m.get("role") == "system"]
            
            # Filter out tool calls, tool responses, and assistant messages that contain tool_calls
            convo_msgs = []
            for m in frame.context.messages:
                if m.get("role") == "system":
                    continue
                if m.get("role") == "tool":
                    continue
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    continue
                convo_msgs.append(m)
                
            # Keep only the last 6 conversational messages (e.g., 3 user, 3 assistant)
            MAX_CONVO_MESSAGES = 6
            convo_msgs = convo_msgs[-MAX_CONVO_MESSAGES:]
            
            frame.context.set_messages(system_msgs + convo_msgs)
            logger.debug(f"[PreLLM] Pruned context to {len(system_msgs)} system msgs and {len(convo_msgs)} convo msgs.")

        # Only process deterministic extraction if there are messages and the last message is from the user
        if not frame.context.messages or frame.context.messages[-1].get("role") != "user":
            return
            
        user_message = frame.context.messages[-1]["content"]
        if not isinstance(user_message, str):
            return

        # Save the original tools on the first pass
        if self._original_tools is None:
            self._original_tools = frame.context.tools

        # Determine the CURRENT goal
        current_objective = self.cm.get_next_objective()
        
        # Only attempt extraction for collect_slot
        if current_objective.kind != "collect_slot" or not current_objective.slot:
            # Fallback path: Ensure tools are enabled
            frame.context.set_tools(self._original_tools)
            self.perf.llm_fallbacks += 1
            logger.info(f"[PreLLM] No active slot to extract. Falling back to LLM.")
            return
            
        target_slot = current_objective.slot
        
        # Run deterministic extraction ONLY for the expected slot
        t0 = time.time()
        extracted_value, confidence = extract_slot(target_slot, user_message)
        latency_ms = (time.time() - t0) * 1000
        
        if extracted_value is not None and confidence >= 0.90:
            logger.info(f"[PreLLM] SUCCESS: Extracted {target_slot}='{extracted_value}' (conf={confidence:.2f}) from '{user_message}'")
            self.perf.deterministic_extractions_successful += 1
            self.perf.latency_deterministic_path.append(latency_ms)
            
            # 1. Update the state
            self.cm.state.update_slot(target_slot, extracted_value)
            
            # 2. Recalculate objective based on NEW state
            new_objective = self.cm.get_next_objective()
            
            # 3. Overwrite the dynamic context block in the in-flight frame!
            # messages[0] = static persona, messages[1] = dynamic context
            if len(frame.context.messages) >= 2 and "Known:" in frame.context.messages[1].get("content", ""):
                frame.context.messages[1]["content"] = build_dynamic_block(self.cm.get_state(), new_objective)
                logger.info(f"[PreLLM] Updated system context. New objective: {new_objective.slot}")
            else:
                logger.warning("[PreLLM] Expected messages[1] to be the dynamic context block, but it wasn't found.")
                
            # 4. Disable tools architecturally for this generation
            frame.context.set_tools([])
            logger.info("[PreLLM] Disabled tools architecturally for this turn.")
        else:
            # Silently defer to the LLM for complex or low-confidence extractions
            logger.debug(f"[PreLLM] Silently deferring {target_slot} extraction to LLM (confidence={confidence:.2f})")
            
            # Re-enable tools for the fallback path
            frame.context.set_tools(self._original_tools)
            
            self.perf.llm_fallbacks += 1
            self.perf.latency_llm_path.append(latency_ms)
