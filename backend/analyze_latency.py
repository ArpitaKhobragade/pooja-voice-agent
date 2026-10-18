import json
import glob
from pathlib import Path
from collections import defaultdict
import statistics

LOGS_DIR = Path(__file__).parent / "logs"

def analyze():
    files = glob.glob(str(LOGS_DIR / "latency_*.json"))
    if not files:
        print("No latency logs found. Please run the calls first.")
        return

    all_turns = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                turns = json.load(fp)
                all_turns.extend(turns)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_turns:
        print("No turn records found in logs.")
        return

    deltas = defaultdict(list)
    total_perceived_latencies = []

    print(f"Loaded {len(all_turns)} turns across {len(files)} calls.\n")

    for turn in all_turns:
        ts = turn.get("timestamps", {})
        
        # We need all these timestamps to compute a full breakdown
        required_keys = [
            "turn_stop",
            "stt_final",
            "llm_run",
            "llm_first_token",
            "first_text_to_tts",
            "first_tts_audio",
            "bot_started"
        ]
        
        if not all(k in ts for k in required_keys):
            continue

        # Stages
        d1 = ts["stt_final"] - ts["turn_stop"]
        d2 = ts["llm_run"] - ts["stt_final"]
        d3 = ts["llm_first_token"] - ts["llm_run"]
        d4 = ts["first_text_to_tts"] - ts["llm_first_token"]
        d5 = ts["first_tts_audio"] - ts["first_text_to_tts"]
        d6 = ts["bot_started"] - ts["first_tts_audio"]

        total_latency = ts["bot_started"] - ts["turn_stop"]
        
        if total_latency <= 0:
            continue

        deltas["1. STT Finalization (Turn Stop -> STT Final)"].append(d1 * 1000)
        deltas["2. Pre-LLM Processing (STT Final -> LLM Run)"].append(d2 * 1000)
        deltas["3. LLM TTFB (LLM Run -> LLM First Token)"].append(d3 * 1000)
        deltas["4. LLM Chunking (LLM First Token -> Text to TTS)"].append(d4 * 1000)
        deltas["5. TTS TTFB (Text to TTS -> First TTS Audio)"].append(d5 * 1000)
        deltas["6. Transport/Buffer (First TTS Audio -> Bot Started)"].append(d6 * 1000)
        
        total_perceived_latencies.append(total_latency * 1000)

    if not total_perceived_latencies:
        print("No fully complete turn records found. Check if the pipeline dropped frames.")
        return

    avg_total = statistics.mean(total_perceived_latencies)
    print("=" * 60)
    print(f"  PERCEIVED RESPONSE LATENCY: {avg_total:.0f} ms")
    print(f"  (Min: {min(total_perceived_latencies):.0f} ms | Max: {max(total_perceived_latencies):.0f} ms)")
    print("=" * 60)
    print(f"{'STAGE':<55} | {'AVG (ms)':<8} | {'MIN':<6} | {'MAX':<6} | {'% OF TOTAL':<5}")
    print("-" * 90)

    biggest_stage_name = None
    biggest_stage_avg = 0

    for stage_name, values in sorted(deltas.items()):
        avg_val = statistics.mean(values)
        min_val = min(values)
        max_val = max(values)
        pct = (avg_val / avg_total) * 100
        
        if avg_val > biggest_stage_avg:
            biggest_stage_avg = avg_val
            biggest_stage_name = stage_name

        print(f"{stage_name:<55} | {avg_val:<8.0f} | {min_val:<6.0f} | {max_val:<6.0f} | {pct:>5.1f}%")

    print("=" * 90)
    print(f"\nBOTTLENECK SUMMARY:")
    print(f"The single largest contributor to perceived silence is:\n  **{biggest_stage_name}** ({biggest_stage_avg:.0f} ms, {(biggest_stage_avg/avg_total)*100:.1f}%)")
    
    print("\nRECOMMENDED OPTIMIZATION TARGET:")
    if "LLM TTFB" in biggest_stage_name:
        print("Focus on reducing LLM context size, prompting complexity, or switching to a faster model tier.")
    elif "TTS TTFB" in biggest_stage_name:
        print("Focus on streaming TTS earlier, using a faster TTS provider, or reducing the chunk size required before TTS generation.")
    elif "Pre-LLM" in biggest_stage_name:
        print("Focus on optimizing the `PreLLMStateExtractor` and database queries that run before the LLM generates.")
    elif "STT Finalization" in biggest_stage_name:
        print("Focus on VAD sensitivity, Smart Turn configurations, or using a faster STT provider.")
    elif "Transport" in biggest_stage_name:
        print("Focus on optimizing jitter buffers and transport network layers.")
    elif "LLM Chunking" in biggest_stage_name:
        print("Focus on adjusting the LLM output token aggregators before sending to TTS.")

if __name__ == "__main__":
    analyze()
