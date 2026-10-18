import asyncio
import os
import time
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import LLMContextFrame, TextFrame
from pipecat.services.groq.llm import GroqLLMService
from pipecat.processors.aggregators.llm_context import LLMContext

from conversation_manager import ConversationManager
from perf_logger import CallPerformanceLogger
from pre_llm_extractor import PreLLMStateExtractor
from objective import determine_next_objective
from prompts import BASE_PERSONA_PROMPT, build_dynamic_block
from tools import SIA_TOOLS

async def run_scenario(scenario_name: str, user_text: str):
    logger.info(f"\n\n================ RUNNING SCENARIO: {scenario_name} ================")
    logger.info(f"User: {user_text}")
    
    cm = ConversationManager("test_phone")
    perf = CallPerformanceLogger("test_call")
    
    pre_llm = PreLLMStateExtractor(cm=cm, perf=perf)
    
    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
            max_tokens=80,
        ),
    )
    
    # We will hook into the push_frame of llm to catch output
    output_frames = []
    
    async def mock_push_frame(frame, direction=None):
        output_frames.append(frame)
        if type(frame).__name__ == "TextFrame":
            if not getattr(llm, "_first_token_time", None):
                llm._first_token_time = time.time()
                logger.info(f"<== First TextFrame received")
        if type(frame).__name__ == "LLMFullResponseEndFrame":
            llm._end_time = time.time()
            logger.info(f"<== LLMFullResponseEndFrame received")
            
    llm.push_frame = mock_push_frame
    
    # We also hook PreLLM's push frame to pass it to LLM
    async def pre_llm_push_frame(frame, direction):
        await llm.process_frame(frame, direction)
        
    pre_llm.push_frame = pre_llm_push_frame

    # Set up dummy tool handler
    def make_handler(name):
        async def handler(pipeline, pipeline_task, params):
            logger.info(f"TOOL EXECUTED: {name} (args: {params.args})")
            perf.mark_tool_execution(name, 0)
        return handler
        
    for t in ["update_state", "book_meeting", "schedule_callback", "mark_not_interested", "end_call"]:
        llm.register_function(t, make_handler(t))

    initial_objective = determine_next_objective(cm.get_state())
    
    context = LLMContext(
        messages=[
            {"role": "system", "content": BASE_PERSONA_PROMPT},
            {"role": "system", "content": build_dynamic_block(cm.get_state(), initial_objective)},
            {"role": "user", "content": user_text}
        ],
        tools=SIA_TOOLS,
    )
    
    logger.info(f"==> Injecting LLMContextFrame")
    t0 = time.time()
    await pre_llm.process_frame(LLMContextFrame(context), None)
    
    # wait for tasks to finish (Pipecat LLM service spawns async tasks)
    await asyncio.sleep(4)
    
    # Metrics
    logger.info(f"\n--- METRICS FOR {scenario_name} ---")
    logger.info(f"Tool executions: {len(perf.tool_executions)}")
    logger.info(f"Deterministic successes: {perf.deterministic_extractions_successful}")
    logger.info(f"LLM fallbacks: {perf.llm_fallbacks}")
    
    text_frames = [f for f in output_frames if type(f).__name__ == "TextFrame"]
    output_text = "".join([f.text for f in text_frames])
    logger.info(f"LLM Output Text: {output_text}")
    
    t_first = getattr(llm, "_first_token_time", None)
    t_end = getattr(llm, "_end_time", None)
    
    if t_first:
        logger.info(f"Time to first token (TTFB): {(t_first - t0)*1000:.0f}ms")
    if t_end:
        logger.info(f"Total turnaround time: {(t_end - t0)*1000:.0f}ms")

async def main():
    await run_scenario("Scenario 1 (Deterministic)", "Yes.")
    await run_scenario("Scenario 2 (Fallback)", "I guess so, but I'm a little busy right now.")

if __name__ == "__main__":
    asyncio.run(main())
