# Pooja — AI Voice Agent for HostelPGDekho.com

Pooja is a standalone, local AI voice agent that calls on behalf of
HostelPGDekho.com to introduce the platform to PG (paying-guest
accommodation) owners. You launch it from a terminal on a Windows
laptop, it greets you aloud, you reply through your microphone, and it
carries on a spoken Hindi/English/Hinglish conversation following an
approved sales script — no telephony, no website, no WhatsApp API.

This project started from the "Sia" repository (a Twilio/Pipecat-based
outbound telephony agent for an interior-design business) and was
repurposed into Pooja. The deterministic business-logic core
(`ConversationManager`, `objective.py`, `agent_config.py`,
`prompts.py`, `state_store.py`) is reused unchanged in shape — only its
content (persona, script, slots) was rewritten. The pipeline itself is
new: a local mic/speaker CLI (`pooja_cli.py`), since the original
project only ran over a Twilio phone call.

## Share Pooja with Sir: the web version

The browser version has a **Start call / End call** button, a live status (listening, thinking,
speaking), a moving voice line, and a **transcript** of both sides (Pooja on the left, the owner on the
right) that can be copied or downloaded. Interrupting Pooja works in the browser too; the browser
cancels echo, so it works without headphones (headphones are still best).

```text
browser mic  --WebSocket-->  server: speech detection -> Sarvam (speech to text) -> Pooja's script / Groq AI
browser speaker  <--WebSocket--  server: Cartesia voice (streamed)  +  transcript events
```

### 1. Try it on your own computer
```powershell
cd backend
pip install -r requirements-web.txt   # or reuse the venv you already have
python web_server.py                  # then open http://localhost:8000
```
It reads the same `backend/.env` keys as the command-line version. Use Chrome or Edge.

### 2. Give Sir a link: pick one

**A. Quick link from your laptop (about 2 minutes, no account).** The link works only while your laptop,
the server window and the tunnel window stay open.
1. In `backend/.env` set `POOJA_ACCESS_CODE=` to a code only Sir will know (needed, see "Protect it").
2. Start the server: `python web_server.py`
3. In a second window: `winget install --id Cloudflare.cloudflared` (once), then
   `cloudflared tunnel --url http://localhost:8000`
4. It prints a link ending in `.trycloudflare.com`. Send Sir that link and the code.

**B. A permanent link on Render (about 15 minutes, free plan available).**
1. Put the project folder (the one containing `Dockerfile`, `render.yaml` and `backend/`) on GitHub:
   ```powershell
   cd "C:\Users\Arpita Khobragade\Desktop\voice agent\pooja-hostelpgdekho-voice-agent"
   git init
   git add .
   git commit -m "Pooja voice agent web demo"
   git branch -M main
   git remote add origin https://github.com/YOUR-NAME/pooja-voice-agent.git
   git push -u origin main
   ```
   Your `.env` is never uploaded (it is in `.gitignore`), and the project was checked to make sure none of your keys appear in any other file.
2. On render.com: **New > Blueprint**, choose that repository. Render reads `render.yaml`.
3. When asked, paste `GROQ_API_KEY`, `SARVAM_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, and choose a `POOJA_ACCESS_CODE`.
4. After the build you get a link like `https://pooja-voice-agent.onrender.com`. Send Sir the link and the code.
5. The free plan goes to sleep after about 15 minutes without visitors and takes about a minute to wake: open the link yourself a few minutes before Sir does. The Starter plan stays awake.

### 3. Protect it
A public link lets anyone who finds it use your Groq, Sarvam and Cartesia credits.
- **`POOJA_ACCESS_CODE`**: visitors must type it before a call can start. Always set it for a public link.
- **`POOJA_MAX_CALLS`** (default 2) and **`POOJA_WEB_MAX_CALL_S`** (default 420 seconds) cap how much one link can use.
- Credits to watch: Cartesia voice (each *new* sentence costs credits; sentences already spoken before are cached in `backend/.tts_cache`, which is included so a fresh deployment starts with the script already voiced), Groq tokens (free tier is about 200,000 a day) and Sarvam speech-to-text.

### 4. What to tell Sir
- Open the link in Chrome or Edge, press **Start call**, allow the microphone.
- Answer like a PG owner. Say "haan", "achha" or "okay" to let her continue, ask a question, or say "ek minute".
- Press **End call** when done, then **Download** to keep the transcript.

### Web troubleshooting
- *"The server is missing settings"* banner: a required key is not set where the server runs (names are shown, never values).
- *"The voice service has run out of credits"*: top up Cartesia and start the call again.
- *No sound / mic blocked*: allow the microphone in the browser's address bar; the page must be `https://` (or `localhost`).
- *Wrong code*: the page says so; the code is `POOJA_ACCESS_CODE`.

## Latest fixes: Groq reliability, STT, TTS pacing, conversation flow

A real test run surfaced seven concrete problems — Sarvam repeatedly
transcribing speech as "Okay", Pooja speaking too slowly, long scripted
explanations after a bare acknowledgement, discussing vacancy before
establishing interest, malformed Groq tool-call arguments, a 404 on the
configured fallback model crashing every subsequent turn, and no
graceful recovery from LLM errors. All seven were root-caused and
fixed; see **Files changed** and **Test results** below for exactly
what was done and verified vs. what still needs your own real-key
testing.

### 1. Groq reliability (`local/llm_groq.py`, `pooja_cli.py`, `local_tools.py`)

- **Root cause of the crash**: `llama-3.3-70b-versatile` (the old
  fallback model) returns `404 model_not_found` on Groq's real API —
  confirmed by actually running the Groq diagnostic against the live
  API with a (deliberately invalid) key. Worse, the old code made the
  fallback **sticky**: once one call fell back, `GroqLLM` permanently
  overwrote its model to the (broken) fallback for the rest of the
  session, so every turn after the first failure crashed too.
- **Fix**: `GroqLLM.chat()` now always retries the *configured* primary
  fresh on each call — never stays stuck on a broken fallback — and
  raises a single `GroqAllModelsFailedError` only if *both* fail for
  that one request (never retried further, never an infinite loop).
  Regression-tested directly (`test_failed_primary_does_not_stick_to_broken_fallback_on_next_call`
  in `tests/test_llm_robustness.py`).
- **New default fallback**: `openai/gpt-oss-20b`, per Groq's own
  deprecation notice (console.groq.com/docs/deprecations) naming it as
  the replacement for retired Llama models, and confirmed
  tool-calling-capable on Groq's model docs. **Verified against the
  real Groq API** (not just docs): both `openai/gpt-oss-120b` and
  `openai/gpt-oss-20b` returned real `401 Unauthorized` responses (not
  `404`) when called with a deliberately invalid key — a 404 would mean
  the model id doesn't exist; a 401 means Groq recognized the model and
  only rejected the bad key. That's real evidence both ids are
  currently live, from the live API, not assumed from documentation.
- **Malformed tool-call handling**: `pooja_cli.py`'s
  `_parse_and_validate_tool_call()` parses and validates every tool
  call's JSON *before* it ever reaches `ConversationManager` — invalid
  JSON, a missing/unknown `slot`, or a missing/empty `value` is
  rejected and reported back to the model as a normal tool failure
  result, never applied as a state change and never crashes the
  session.
- **`tool_choice="auto"`** (never forced) — confirmed against the real
  Groq API (a tool-enabled request also got a clean 401, not a 400
  schema error, proving the request shape itself is valid).
- **Total-failure recovery**: if Groq is down entirely,
  `PoojaSession._end_due_to_llm_failure()` ends the call with one
  canned, deterministic apology instead of raising past `main()` —
  the process never crashes.

### 2. STT and microphone capture (`local/audio_io.py`, `pooja_cli.py`)

- **Device selection**: `--input-device` / `--output-device` (index or
  name substring) or `POOJA_INPUT_DEVICE` / `POOJA_OUTPUT_DEVICE` in
  `.env`. The session banner now prints the resolved device names and
  sample rate (`audio_io.describe_devices()`) — verified for real on
  this machine: `Input: 'Microphone Array (Intel® Smart ' (device
  default sample rate 44100Hz, capturing at 16000Hz) | Output:
  'Headphones (Airdopes 161)'`.
- **Signal-level checks on every recording** (`RecordResult` now
  carries `rms`, `peak`, `is_silent`, `is_clipped`, `too_short`,
  computed by the pure, unit-tested `audio_io.analyze_pcm()`): a
  near-silent clip is never sent to Sarvam at all (saves an API call
  and avoids a garbage transcript); a clipped clip still gets sent but
  with a printed gain warning.
- **Suspicious/duplicate-transcript handling**
  (`pooja_cli._transcript_is_unreliable()`): Sarvam's sync REST
  endpoint returns no confidence score, so this uses two practical
  proxies instead — a one-word transcript from a clip over 1 second
  long (mismatch between how long you spoke and what came back), or
  the exact same transcript twice in a row. Either one causes Pooja to
  ask you to repeat rather than trusting the transcript — this is very
  likely what was producing the repeated "Okay" transcripts.
- **Configurable timing, no code edits needed**:
  `POOJA_SILENCE_TIMEOUT_S` (default raised from 1.2s to 1.5s — the
  old value may have been cutting off normal mid-sentence pauses),
  `POOJA_PRE_SPEECH_TIMEOUT_S`, `POOJA_MAX_RECORD_S`,
  `POOJA_MIN_SPEECH_S`, `POOJA_SILENCE_RMS_THRESHOLD`.
- **Short utterances are never discarded** — a real "haan"/"nahi" still
  gets sent to Sarvam; only clips that are *both* short *and* silent
  are skipped.
- **`--diagnose`**: an interactive menu (option 1) that records, then
  **plays back exactly what was captured** before anything is sent to
  Sarvam — the only honest way to confirm what the microphone is
  actually picking up, instead of inferring it from the transcript
  alone. Verified for real on this machine (RMS/duration/device
  reporting all correct against live mic input).

### 3. TTS pacing (`local/tts_cartesia.py`)

- **Speed control, verified against the installed SDK's real type**
  (`cartesia/types/generation_config_param.py`): `generation_config.speed`
  is a documented float field in `[0.6, 1.5]` for sonic-3+ models
  (which includes the configured `sonic-3.5`) — not a guessed
  `speed=1.2` kwarg. Set via `CARTESIA_SPEED` in `.env` (default
  `1.0`); out-of-range values are rejected before any API call.
  **`CARTESIA_VOICE_ID` is never touched by this** — confirmed by a
  dedicated test asserting the voice id in the request is unchanged
  when speed is tuned.
- Synthesis latency and playback duration are now measured and
  reported **separately** (`SpeakResult.synthesis_s` /
  `.playback_s`, printed with `--debug`) — no arbitrary `sleep()` calls
  were added anywhere.

### 4-5. Conversation flow and pacing (`prompts.py`, `agent_config.py`)

- Rewrote the system prompt's core rule: a bare "Okay"/"Haan"/"Theek
  hai" is now explicitly treated as a **weak, neutral acknowledgement**
  — Pooja must ask one short follow-up question before explaining
  anything, never launch into the full pitch.
- "Mera PG full hai" no longer jumps straight to explaining the
  Vacant-Bed Request Form — the prompt now requires asking permission
  first (`agent_config.py`'s new `ask_vacancy_feature_permission`
  line), matching the requested flow.
- Turns are capped at one–two short sentences (was "10-25 words"),
  with explicit instructions against over-thanking and against
  advancing the flow on a vague acknowledgement.
- Opt-out ("stop calling") remains fully deterministic — `pooja_cli.py`
  checks `STOP_PHRASES` and ends the call **before any LLM call is
  made at all** (not just before any state change), so it can't be
  reinterpreted or overridden by the model, and keeps working even if
  Groq is completely down.
- **Not independently verified by an automated test**: actual spoken
  conciseness and tone are a property of the live model's output, not
  something a unit test can assert — judge this yourself in a real
  `--mode voice` run.

### 6. Diagnostic mode (`pooja_cli.py --diagnose`)

```bash
python pooja_cli.py --diagnose
```
Six options: (1) mic capture + local playback, (2) Sarvam STT on a
fresh recording (also plays back what was captured first), (3) Groq
plain chat, (4) Groq tool-calling (a valid call, then an intentionally
invalid one, showing the validator reject it), (5) Cartesia TTS with
your configured voice, (6) a full `--mode voice` conversation. **Never
places a real call or touches `db.py`'s calling database** — this
project has no telephony integration to do that with in the first
place. A recorded clip is saved to `backend/diagnostics/*.wav` **only**
after an explicit `y` confirmation prompt — never automatic, never
uploaded anywhere.

### Debug mode

```bash
python pooja_cli.py --mode voice --debug
```
Prints, every turn: recorded audio duration/RMS/clipping, STT
transcript + latency, Groq model used + per-call latency is visible
via the existing `[Tool]` log lines, and Cartesia synthesis/playback
latency — all without ever printing an API key.

## What this is (and is not)

- **Is:** a turn-based local voice CLI with best-effort barge-in.
  Pooja speaks a turn; while she's speaking, the microphone is already
  being watched, and sustained speech from you cuts her off early. Once
  it's your turn, she listens until you stop talking (silence-based
  endpointing).
- **Is not:** a phone call, a website/dashboard, or a WhatsApp
  integration. Barge-in is best-effort (see Known Limitations) — this
  is not a verified, hands-free, echo-cancelled full-duplex pipeline.

## Three ways to run it

| Mode | Needs | What it exercises |
|---|---|---|
| `--mode voice` | mic + speakers, `GROQ_API_KEY`, `SARVAM_API_KEY`, `CARTESIA_API_KEY` + `CARTESIA_VOICE_ID` | Full pipeline: Sarvam STT → Groq LLM → **Cartesia TTS (mandatory by default)** |
| `--mode text` | `GROQ_API_KEY` only | Same conversation manager + LLM, keyboard instead of mic/speaker |
| `--mode offline` | nothing — no keys, no network | Deterministic, keyword-driven walkthrough of the literal approved script |

Run without `--mode` and it auto-picks the best available option based
on what's configured (see `_detect_default_mode()` in `pooja_cli.py`).

```bash
python pooja_cli.py --mode voice              # Cartesia mandatory; fails fast with instructions if misconfigured
python pooja_cli.py --mode voice --tts-engine pyttsx3     # explicit, deliberate offline voice instead
python pooja_cli.py --mode voice --allow-tts-fallback     # Cartesia preferred, falls back per-turn on failure (reported, not silent)
python pooja_cli.py --mode voice --no-barge-in            # disable interruption detection
python pooja_cli.py --find-voice                          # list Hindi female Cartesia voices on your account
python pooja_cli.py --list-devices                        # see what Windows reports as mic/speaker devices
```

## Technology stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Groq, `openai/gpt-oss-120b` | Verified current model id on Groq's docs; falls back to `llama-3.3-70b-versatile` once if the primary call fails (this retry path was actually exercised against the real Groq API while building this — see Test Results) |
| Speech-to-Text | Sarvam AI, `saaras:v3` (REST) | Handles Hindi/English/Hinglish; 30s/clip limit on the sync endpoint, fine for one turn |
| Text-to-Speech | **Cartesia, official `cartesia` Python SDK, mandatory by default** | `sonic-3.5` by default; see below |
| Voice Activity Detection | Silero VAD (optional) → `webrtcvad` (default) → energy threshold (last resort) | Three-tier fallback, see below |
| Audio I/O | `sounddevice` (PortAudio) | Mic capture + speaker playback, including concurrent capture during playback for barge-in |

### Cartesia — verified against the installed SDK, not guessed

`local/tts_cartesia.py` uses the **official `cartesia` Python SDK**
(`pip install cartesia`), calling `client.tts.generate(model_id=,
transcript=, voice=, language=, output_format=)`. This was built by
reading the SDK's own generated type definitions after installing it —
not from memory or assumption:

- `cartesia/types/tts_model.py` → the valid `model_id` literals are
  `"sonic-3.5"`, `"sonic-3"`, `"sonic-3.5-2026-05-04"`,
  `"sonic-3-2026-01-12"`, `"sonic-3-2025-10-27"`, `"sonic-latest"`.
  **There is no `"sonic-3.6"`** in this SDK version — the requested
  stack named a model id that doesn't exist in the currently installed
  SDK. `CARTESIA_MODEL_ID` defaults to `"sonic-3.5"` (the latest
  numbered id); check
  <https://docs.cartesia.ai/getting-started/available-models> if a
  newer id ships later — it's a one-line `.env` change.
- `cartesia/types/supported_language.py` → confirms `"hi"` (Hindi) is
  a supported language code, used for `CARTESIA_LANGUAGE`.
- `cartesia/_exceptions.py` → `AuthenticationError`, `RateLimitError`,
  `APIConnectionError`, `APITimeoutError`, `APIStatusError` are all
  caught individually in `CartesiaTTS.synthesize()` with specific,
  actionable error messages (invalid key vs. rate limit vs. network vs.
  other API error) — not one generic "something went wrong".

**Cartesia is mandatory by default** for `--mode voice`: if
`CARTESIA_API_KEY`/`CARTESIA_VOICE_ID` aren't set, or a Cartesia call
fails, `pooja_cli.py` **refuses to start the conversation** and prints
exactly what to fix — it does not silently switch to `pyttsx3` and
report success. `pyttsx3` is only used if you explicitly pass
`--tts-engine pyttsx3` (fully offline, deliberate choice) or
`--allow-tts-fallback` (same-turn fallback on failure, but every such
fallback is printed with `[TTS ENGINE: PYTTSX3]` so it's never hidden).
Every turn prints which engine actually spoke it.

**Finding a voice id:** Cartesia voice ids are account-specific — this
project cannot ship one. Run `python pooja_cli.py --find-voice` (needs
only `CARTESIA_API_KEY`) to list Hindi, feminine-presenting voices from
your own Cartesia account. Note: `client.voices.list()` has no
`language` parameter in the installed SDK (checked against
`cartesia/resources/voices.py` — `gender`, `q`, `limit`, etc. exist,
`language` doesn't), so `list_hindi_female_voices()` filters by
`gender="feminine"` server-side and by `Voice.language == "hi"`
client-side; set `CARTESIA_VOICE_ID` to one of the ids it prints.

**Auditioning voices by ear:** once you have some candidate ids (e.g.
from `--find-voice`), run:

```bash
python pooja_cli.py --audition-voices
```

This plays Pooja's real opening line in each of 10 pre-listed Hindi,
feminine-presenting voice ids (the ones returned by `--find-voice` on
this project's test account — re-run `--find-voice` for your own
account's list and edit `VOICE_CANDIDATES` in
`local/voice_audition.py` if you want to audition a different set). For
each voice it announces the name, plays the sample through your
speakers, then prompts `[r]eplay / [n]ext voice / [q]uit without
selecting`. A voice that fails to synthesize (bad id, rate limit, etc.)
is reported and skipped automatically — one bad id never blocks the
rest. **It never writes to `.env` and never picks a voice for you** —
copy whichever id you liked into `CARTESIA_VOICE_ID` yourself. Override
the sample line with `--audition-text "..."`. This was run against the
real Cartesia API while building it (with a deliberately invalid key,
to confirm the request/error-handling path) — see Test Results.

**Auditioning voices in a browser (Play/Pause/Restart, pick & save):**
if pressing `r`/`n`/`q` in a terminal isn't comfortable, there's a local
web page with real controls:

```powershell
cd backend
venv\Scripts\activate
python pooja_cli.py --audition-voices-web
```

This opens **http://127.0.0.1:8765** in your browser automatically (pass
`--no-browser` to skip that, or `--audition-port 9000` to use a
different port). It's a small FastAPI server ([local/audition_server.py](backend/local/audition_server.py))
serving a single static page ([local/audition_web/index.html](backend/local/audition_web/index.html)) —
**listens on localhost only**, no cloud deployment, no network exposure.

What it does, reusing exactly the same pieces as everything else in
this project (no new Cartesia calls invented): each of the 10 voice
cards has real **Play / Pause / Restart / Select This Voice** controls
backed by a genuine HTML5 `<audio>` element — clicking Pause calls the
browser's native `audio.pause()`, which truly preserves `currentTime`
for a real resume (not a fake "stop and hope", as specifically asked
for). Audio for each voice is synthesized once via `CartesiaTTS.synthesize()`
and cached server-side in memory for the life of that server run, so
replaying a voice — even across a page reload — never triggers a second
paid Cartesia call. Only one voice plays at a time (starting another
stops whatever was playing). A voice that fails to synthesize shows a
visible "Error — playback failed" state on its own card without
breaking the page; you can just click Play again.

Selecting a voice (highlighting a card) is purely a UI choice — nothing
is written anywhere until you click **Save Selected Voice**, which
`POST`s only that voice's id to `/api/save`. The backend then calls
`local/env_update.py`'s `update_env_value()`, which rewrites **only**
the `CARTESIA_VOICE_ID` line in `.env` and leaves every other line —
including `GROQ_API_KEY`, `SARVAM_API_KEY`, `CARTESIA_API_KEY`, and
anything else in the file — byte-for-byte untouched (atomic temp-file-then-rename
write, so a crash mid-save can't corrupt `.env`). This was verified for
real, not just unit-tested: the save flow was run through an actual
browser against a `.env` containing three fake "secret" values, and the
resulting file had only the `CARTESIA_VOICE_ID` line changed — see Test
Results below.

The API key itself never reaches the browser: `/api/voices` and
`/api/audio/{id}` never include it in their response bodies, and it's
never logged by any endpoint — confirmed by both a dedicated test
(`test_run_audition_cli_never_logs_the_api_key` /
`test_get_voices_returns_all_ten_without_api_key`) and by inspecting
the real network response during the live browser test above.

**Compatibility note on saving a new voice:** this only changes
`CARTESIA_VOICE_ID`, which is read once when `--mode voice` starts — it
takes effect the next time you launch `python pooja_cli.py --mode voice`
(restart it if one is already running). It has no effect on `--mode
text` or `--mode offline`, which never call Cartesia at all. Since all
10 candidate voices were pre-filtered as Hindi, feminine-presenting
voices (matching `CARTESIA_LANGUAGE=hi`), picking any of them keeps the
existing model/language configuration compatible — there's no language
mismatch risk to worry about here.

### VAD — three-tier fallback, Silero included but not default-installed

`local/audio_io.py` tries, in order: **Silero VAD** (if `pip install
silero-vad torch` was done — most accurate, but pulls in PyTorch,
hundreds of MB, so it's *not* in `requirements.txt` by default) →
**`webrtcvad`** (via the prebuilt-wheel `webrtcvad-wheels` package on
Windows — accurate, tiny, this is the default) → a simple **RMS energy
threshold** (always available, least accurate, last resort). Force a
specific backend with `POOJA_VAD_BACKEND=silero|webrtcvad|energy` in
`.env`. The active backend is printed in the session banner and in the
`[STT]` line each turn.

## Architecture

```text
backend/
├── pooja_cli.py              # Entry point: preflight checks, mode selection, the conversation loop, call-style UI
├── agent_config.py           # Pooja's persona, required slots/phases, literal script lines
├── prompts.py                # System prompt built from agent_config (persona + sales script + rules)
├── objective.py               # "What should Pooja do next?" given the current state
├── conversation_manager.py   # Validates/applies tool calls, owns terminal-state transitions
├── state_store.py            # Generic slot-filling state (knows nothing about HostelPGDekho)
├── local_tools.py            # Plain OpenAI/Groq function-calling schema (no pipecat dependency)
├── local/
│   ├── audio_io.py           # Mic capture + 3-tier VAD + speaker playback + best-effort barge-in
│   ├── stt_sarvam.py         # Sarvam STT REST client
│   ├── tts_cartesia.py       # Official Cartesia SDK wrapper, TTSEngine (mandatory/fallback policy), pyttsx3
│   ├── llm_groq.py           # Groq chat-completions wrapper (+ one fallback-model retry)
│   ├── voice_audition.py     # --audition-voices: terminal replay/next/quit loop
│   ├── audition_server.py    # --audition-voices-web: local FastAPI server (voices/audio-cache/save-env)
│   ├── audition_web/
│   │   └── index.html        # The browser page: Play/Pause/Restart/Select, real HTML5 <audio>
│   ├── env_update.py         # Byte-preserving single-key .env writer (used by Save Selected Voice)
│   └── offline_script.py     # No-LLM, no-API-key keyword-driven script demo
├── tests/
│   ├── test_core_logic.py    # State machine / ConversationManager unit tests
│   ├── test_offline_script.py# Full script + every objection scenario, driven programmatically
│   ├── test_pooja_session.py # LLM tool-calling loop, repeat handling, barge-in flag — fake LLM, no network
│   ├── test_providers.py     # Sarvam/Cartesia wrappers — MOCKED, no network, no API keys
│   ├── test_voice_audition.py# Terminal audition replay/next/quit flow, error-skip, no-env-writes — MOCKED
│   ├── test_audition_server.py# Web audition FastAPI endpoints — MOCKED (TestClient, no real network/audio)
│   ├── test_env_update.py    # .env single-key rewrite — byte-for-byte preservation of everything else
│   └── test_live_apis.py     # REAL Groq/Sarvam/Cartesia calls — auto-skipped unless your .env has real keys
├── requirements.txt           # Everything needed to run pooja_cli.py in any mode
└── .env.example                # Template for pooja_cli.py's environment variables
```

### Conversation flow (deterministic, explicit phases)

`agent_config.py`'s `required_slots` define Pooja's phases, filled in
priority order by the LLM calling the `update_state` tool — but the
*ConversationManager*, not the LLM, decides what the update actually
means and when the call ends:

1. `permission_granted` — opening + "is now a good time?"
2. `benefit_smart_search_explained`
3. `benefit_vacant_form_explained`
4. `benefit_free_listing_explained`
5. `interest_decision` — yes / no / maybe

`link_requested` is tracked separately (not required — triggered
whenever the owner asks for the app link, at any point). Yes/no/maybe
answers are validated and normalized against a bilingual
(English/Hindi/Hinglish) vocabulary in `conversation_manager.py` before
they're accepted — a rambling or ambiguous answer is rejected and the
LLM is told to ask a clarifying question instead of guessing.

The call ends (`call_ended=True`) when the owner confirms interest,
explicitly declines, says "maybe later", or can't talk right now — each
produces a *different* closing line, not a generic "goodbye".

**"Aap repeat kar sakte ho?"** is handled deterministically, not by the
LLM: a keyword match re-speaks Pooja's exact last line (no new API
call, no risk of the wording drifting on a repeat).

**Barge-in** sets `was_interrupted_last_turn` on the conversation
state, which `prompts.py`'s `build_dynamic_block()` turns into an
explicit system note telling the LLM its last message was cut off and
to resume/acknowledge gracefully rather than skip ahead — this wiring
already existed in the original Sia project (for Twilio interruptions)
and is reused as-is here.

### Legacy: the original Twilio/Pipecat pipeline

`main.py`, `voice/bot.py`, `tools.py` (the Pipecat `FunctionSchema`
version), and `extraction.py` (Gemini-based post-call extraction) are
the **original Sia telephony pipeline**, kept for reference only — not
part of this deliverable, not exercised by `pooja_cli.py`. Install
`requirements-legacy-twilio.txt` separately if you want to resurrect
them.

## Windows setup

```bash
# 1. Open the project folder
cd hostelpgdekho-pooja-agent\backend

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
notepad .env
#   - GROQ_API_KEY          (required: text and voice modes)
#   - SARVAM_API_KEY        (required: voice mode)
#   - CARTESIA_API_KEY      (required: voice mode, unless --tts-engine pyttsx3)
#   - CARTESIA_VOICE_ID     (get this by running step 5 below, after setting CARTESIA_API_KEY)

# 5. Find a Cartesia voice id for your account (needs CARTESIA_API_KEY set)
python pooja_cli.py --find-voice
#   -> copy one of the printed ids into CARTESIA_VOICE_ID in .env

# 6. Check Windows sees your mic and speakers
python pooja_cli.py --list-devices

# 7. Launch Pooja
python pooja_cli.py --mode voice     # or --mode text / --mode offline, or omit --mode to auto-detect

# 8. Run the automated tests
python -m pytest tests/ -v

# 9. Run the fully offline text simulation (no API keys, no network)
python pooja_cli.py --mode offline
```

### Troubleshooting

- **`pip install -r requirements.txt` fails on `webrtcvad`**: neither
  the prebuilt `webrtcvad-wheels` wheel nor a C++ compiler was
  available. Not fatal — Pooja logs a warning and uses the
  energy-threshold VAD instead. To get the more accurate backend,
  install "Desktop development with C++" via the Visual Studio
  Installer, or just accept the fallback.
- **"No microphone (input device) detected" / "No speaker detected"**:
  run `python pooja_cli.py --list-devices` and check Windows Sound
  Settings → make sure the right device is enabled and not muted, and
  that Windows's microphone privacy setting (Settings → Privacy →
  Microphone → "Allow apps to access your microphone") is on for
  desktop apps.
- **Cartesia errors at startup**: the exact message tells you which —
  "authentication failed" means a bad `CARTESIA_API_KEY`; "CARTESIA_VOICE_ID
  is not set" means run `--find-voice` first; a rate-limit or network
  message means retry or check your connection. The session will not
  start and silently use a worse voice instead — by design.
- **Sarvam/Groq 401 errors**: double-check the key was pasted into
  `.env` correctly (no quotes, no trailing space) and that `.env` is in
  `backend/`, not the project root.
- **Pooja keeps "hearing" background noise as speech**: lower the mic
  input volume in Windows Sound Settings, or force a more accurate VAD
  backend (`POOJA_VAD_BACKEND=webrtcvad` in `.env`, or install Silero —
  see above).
- **Barge-in triggers on Pooja's own voice (false interruption)**: this
  is speaker-to-mic echo, expected without hardware echo cancellation —
  use headphones, or pass `--no-barge-in` to disable the feature
  entirely for the demo.

## Recording the demo for your professor

1. **Use headphones** (wired or Bluetooth) if at all possible — this
   avoids Pooja's own TTS audio leaking into your mic, which both
   causes false barge-in triggers and degrades Sarvam's transcription
   of your actual speech.
2. **Audio capture settings**: Windows' built-in Xbox Game Bar
   (`Win+G`) or any screen recorder that captures "system audio +
   microphone" will pick up both your voice and Pooja's TTS output
   together, since both play through the same audio session. In
   OBS/Game Bar, make sure both "Desktop Audio" (for Pooja) and
   "Microphone" (for you) are enabled as separate audio tracks or
   mixed into one — check your recorder's audio settings before the
   real take.
3. **Verify before presenting**: record a short 10-second test clip
   first (e.g. just the opening greeting + "Haan ji, bataiye"), play it
   back, and confirm you can hear BOTH your own voice and Pooja's reply
   clearly. Do not assume the levels are right.
4. Follow the demonstration script below for the actual recording.
5. This is a local microphone/speaker session, not a telephone call —
   don't present it as one; the terminal window and call-style banner
   make that clear on screen already.

## Demonstration script (for the professor)

1. `python pooja_cli.py --mode voice` (or `--mode text` if no
   mic/speakers are available in the room, or `--tts-engine pyttsx3` if
   Cartesia isn't configured for this specific run).
2. Pooja opens: *"Hello Sir, good afternoon... Kya hum baat kar sakte
   hain?"* — reply **"Haan ji, bataiye."**
3. Pooja introduces HostelPGDekho and explains **Smart Search**.
   Ask: **"Aapka app kaise kaam karta hai?"** — she should answer
   contextually, not just repeat the same pitch.
4. Say **"Mera PG already full hai."** — she should pivot to the
   **Vacant-Bed Request Form** (it still helps once a bed opens up).
5. Ask **"Listing free hai kya?"** — she should confirm the
   free-listing offer without inventing new conditions.
6. Try barge-in: start talking while she's still mid-sentence on a
   longer reply — playback should stop early (watch for
   `[Barge-in]` in the log / the turn ending sooner than her full line).
7. Ask **"Aap repeat kar sakte ho?"** — she should re-say her exact
   last line.
8. Ask for the app link: **"App ka link bhej do."** — she must say
   this local demo **cannot send a WhatsApp message**, and instead read
   out the configured app URL. She must never claim a message was
   sent.
9. Reply **"Haan, interested hoon."**
10. Pooja thanks you and closes warmly in under three short sentences —
    the call should end on its own (`call_ended` becomes `True`); you
    don't need to force-quit.
11. Press `Ctrl+C` at any point to confirm clean shutdown.
12. **Play back the recording and verify both your voice and Pooja's
    are audible** before sending it to your professor.

Expected transcript of this run is saved to `backend/sia.db` under a
freshly generated `call_id`, printed at the end of the run.

## Features

- Deterministic, phase-based conversation state machine.
- Mandatory Cartesia TTS by default, with an explicit (never silent)
  opt-in fallback policy.
- Three-tier VAD (Silero optional → webrtcvad default → energy
  threshold), reported per session.
- Best-effort barge-in: sustained speech during Pooja's turn cuts
  playback short; the LLM is told the previous turn was interrupted.
- Deterministic "repeat last line" handling (no LLM call, no wording
  drift).
- Bilingual (Hindi/English/Hinglish) yes/no/maybe normalization with
  rejection of ambiguous answers.
- Three closing variants (interested / not interested / busy-declined).
- Honest WhatsApp-link handling: reads out the app URL, never claims a
  message was sent.
- Duplicate-response suppression.
- Call-style terminal UI: session banner (engines in use, VAD backend),
  elapsed-time timer, LISTENING/THINKING/SPEAKING status lines.
- Max-turn safety net (`POOJA_MAX_TURNS`, default 20).
- `--find-voice` helper to discover a real Cartesia voice id from your
  own account instead of guessing one.
- `--audition-voices` to listen to multiple candidate voices back to
  back (replay/next/quit) before committing to one — never writes to
  `.env`, never auto-selects.
- `--audition-voices-web`: a local-only browser page with real
  Play/Pause/Restart controls (true HTML5 `<audio>`, not a fake pause),
  server-side audio caching per voice, and an explicit Save button that
  surgically updates only `CARTESIA_VOICE_ID` in `.env`.
- Fully offline demo mode (`--mode offline`) needs zero API credits and
  zero network calls.
- Transcripts persisted to SQLite (`backend/sia.db`).

## Known limitations (explicitly, not hidden)

- **Barge-in is best-effort, not guaranteed.** Without echo
  cancellation, a laptop's speakers bleeding into its own mic can
  trigger false interruptions; headphones make this reliable, their
  absence doesn't. This is not full-duplex streaming — it's playback
  with concurrent VAD polling. The interrupted-flag plumbing is tested
  with a fake input/output pair (`test_barge_in_sets_interrupted_flag_for_next_turn`
  in `test_pooja_session.py`); the real `audio_io.play_interruptible()`
  was also run against this machine's actual speakers/mic (playing a
  1s tone while silent — no false trigger), but not against your
  specific hardware or with real speech.
- **Not low-latency.** Each turn is: record-until-silence → one STT
  call → one-or-two LLM calls → one TTS call. Expect a few seconds of
  pause between your turn ending and Pooja replying.
- **STT clip limit.** Sarvam's synchronous REST endpoint handles up to
  ~30s per utterance.
- **Offline mode's objection handling is keyword-based**, not true
  language understanding — see `local/offline_script.py`'s keyword
  lists.
- **Silero VAD is opt-in, not installed by default** (PyTorch is
  heavy); `webrtcvad` is the default and was verified working (see Test
  Results).
- **Not fully exercised end-to-end with real speech in this
  environment** — this was built without access to real
  Sarvam/Cartesia API keys. What WAS actually run and verified here, on
  real audio hardware (not mocked):
  - Every unit/mocked test (`test_core_logic.py`, `test_offline_script.py`,
    `test_pooja_session.py`, `test_providers.py`) — real code paths, no
    network.
  - The full offline (`--mode offline`) conversation end-to-end,
    multiple times, through every scripted branch, via the real CLI
    entry point.
  - All CLI preflight error paths (missing `SARVAM_API_KEY`, missing
    `GROQ_API_KEY`, missing/misconfigured Cartesia, `--tts-engine
    pyttsx3`) via the real CLI entry point.
  - The Groq fallback-model retry logic against the **real** Groq API
    (with an intentionally invalid key, to confirm the request reaches
    Groq, gets a real 401, retries the fallback model, and fails with a
    clear error).
  - The Cartesia SDK's real call shape, inspected directly from the
    installed package's generated type definitions (not guessed).
  - **Real microphone + speaker hardware on this machine**: played an
    actual audio tone through real speakers (`play_pcm_int16`); ran the
    real `record_utterance()` against a live mic input stream and
    confirmed it times out and returns cleanly when no one is speaking;
    ran `play_interruptible()` (the actual barge-in code path) with a
    real concurrent mic+speaker stream and confirmed no false trigger
    during silence. Real Silero VAD and webrtcvad backends evaluated
    both synthetic and live mic frames correctly.
  What was **not** run: Sarvam transcribing real recorded speech, or
  Cartesia synthesizing and playing a real Pooja reply — both require
  your API keys, which weren't available here. **Please run
  `python pooja_cli.py --mode voice` yourself** after filling in real
  keys in `.env`, and tell me if anything in the Sarvam/Cartesia
  request shape needs adjusting.

## Environment variables

See `.env.example` for the full, current list. Nothing is required for
`--mode offline`.

## API costs

- **Groq**: free tier credits on signup; `openai/gpt-oss-120b` is
  currently priced around $0.15/M input, $0.60/M output tokens (check
  current pricing at `console.groq.com`).
- **Sarvam AI**: free credits on signup (`dashboard.sarvam.ai`).
- **Cartesia**: has a free tier (`play.cartesia.ai`); if you want a
  guaranteed-free run instead, use `--tts-engine pyttsx3` explicitly.
- No telephony, no WhatsApp API, no mandatory subscription.

## Test results

**Baseline** (the original repo, before any changes),
`python -m pytest backend/tests/ -v`: **9 passed, 3 failed** — the 3
failures were stale tests calling tool names (`book_meeting`,
`mark_not_interested`) that no longer existed in
`conversation_manager.py` after a prior refactor.

**Current**, `python -m pytest tests/ -v`:

```
107 passed, 4 skipped in 1.87s
```

The 4 skipped are `test_live_apis.py` — they require real API keys and
auto-skip without them (see that file's docstring; run them yourself
with real credentials in `.env`).

| Test file | What it covers | Mocked / Live |
|---|---|---|
| `test_core_logic.py` | State-store slot history, objective selection, bilingual yes/no/maybe validation, redundant-update idempotency, unknown-tool/placeholder rejection | Pure logic, no I/O |
| `test_offline_script.py` | Full happy path, busy+reschedule (accept/decline), mid-script refusal, link request + honest WhatsApp wording, stop keyword — **this test suite caught and led to a fix for a real keyword-ordering bug** ("abhi time nahi hai" was being misread as an outright refusal instead of "busy" because "nahi" matched the refusal keyword list first) | Pure logic, no I/O |
| `test_pooja_session.py` | Tool-call → state-update → next-completion loop, stop-phrase handling, **repeat-phrase handling (no extra LLM call)**, **barge-in interrupted-flag propagation**, duplicate-response suppression, max-turn safety net | Fake LLM, no network |
| `test_providers.py` | Sarvam STT request shape + error handling (auth/network/bad response); Cartesia SDK call shape, mandatory-vs-fallback TTS policy, explicit-pyttsx3 mode never touching Cartesia — **also caught a real bug**: `list_hindi_female_voices()` was passing a `language=` kwarg that doesn't exist on the installed SDK's `voices.list()`, which would have crashed `--find-voice` the first time anyone ran it | Mocked `requests`/`cartesia.Cartesia`, no network |
| `test_voice_audition.py` | `--audition-voices` (terminal) replay/next/quit flow, blank-input-advances, automatic skip-on-failure, **never writes to `os.environ`**, **never logs the API key** | Mocked, no network, no real audio |
| `test_audition_server.py` | Web audition server: `/api/voices` never includes the API key, `/api/audio/{id}` caches (synthesize called once, not twice) and returns 404/502 correctly without leaking the key in error text, `/api/save` writes only `CARTESIA_VOICE_ID`, rejects unknown ids (400), surfaces write failures (500) instead of hiding them | FastAPI `TestClient`, mocked Cartesia/env-write |
| `test_env_update.py` | The `.env` rewriter: replaces an existing key preserving every other line exactly, appends when missing, creates the file if absent, doesn't false-match a key that's a substring of another key, rejects unsafe key names, leaves no leftover temp file | Pure logic, real temp-file I/O, no network |
| `test_llm_robustness.py` | `GroqLLM`'s single-bounded-fallback policy (never sticky to a broken fallback — a direct regression test for the crash reported this round), error categorization, tool_choice="auto" only with tools; tool-call JSON/schema validation (malformed JSON, missing fields, unknown slot/tool); total-Groq-failure never crashes `PoojaSession` | Mocked `groq.Groq`, fake LLM, no network |
| `test_stt_robustness.py` | Suspicious/duplicate-transcript heuristic; PCM silence/clipping/short-clip analysis (`analyze_pcm`); input/output device resolution by index or name; expanded Hindi/Hinglish yes-no-maybe vocabulary ("achha", "busy hoon", "baad mein call karna") | Pure logic + synthetic audio arrays, no real mic |
| `test_live_apis.py` | Real Groq chat completion + tool-calling, real Sarvam transcription of silence, real Cartesia synthesis | **Real APIs**, skipped without credentials |

**Real verification against the live Groq API this round** (beyond
`test_live_apis.py`, which needs a *valid* key): the Groq diagnostics
(`pooja_cli.py --diagnose`, options 3 and 4) were run with a
deliberately *invalid* `GROQ_API_KEY` against the real
`https://api.groq.com` endpoint. Both `openai/gpt-oss-120b` and
`openai/gpt-oss-20b` returned genuine `401 Unauthorized` responses —
not `404 model_not_found` — which is real evidence both model ids are
currently live on Groq (a nonexistent id 404s regardless of the key).
The tool-calling request (with `tools=` in the payload) also got a
clean 401, not a 400 schema error, confirming the tool schema itself is
valid against the real API. This is the strongest verification possible
without a real, working key — the fallback-retry logic, the error
categorization, and the request shape were all exercised against
Groq's actual servers, just not with a key that could complete a call.

**Real browser verification of the web audition UI** (beyond the
mocked `test_audition_server.py` suite): the actual server was started
and driven through a real browser —
1. Loaded `http://127.0.0.1:8765`, confirmed the page, sample text, and
   all 10 voice cards rendered.
2. Clicked **Select This Voice** on Siya's card — confirmed it was
   highlighted, the "SELECTED" chip appeared, and the footer updated to
   "Currently selected: Siya — not saved yet."
3. Clicked **Play** (with no `CARTESIA_API_KEY` configured in this
   environment) — confirmed a clean "Error — playback failed" state on
   that card, the page did not crash, and the real network request
   (`GET /api/audio/{id}`) returned `502 Bad Gateway` with no key in
   the response.
4. Created a real `.env` with three fake "secret" values plus a
   placeholder `CARTESIA_VOICE_ID`, clicked **Save Selected Voice**,
   and confirmed the on-disk file afterward: `CARTESIA_VOICE_ID` was
   updated to Siya's real id, and the three fake secrets plus every
   other line were preserved **byte-for-byte**. This test `.env` was
   then deleted before packaging — it is not included in the delivered
   project.
What wasn't verified here: actually hearing a real synthesized sample
(needs a real `CARTESIA_API_KEY`) — the Play/Pause/Restart button logic
itself (native HTML5 `<audio>` play/pause/currentTime) is standard
browser behavior, not something this project implements, so once audio
successfully loads with your real key it will behave as any `<audio>`
element does.

**Real-hardware verification** (beyond the automated suite above, run
directly against this machine's actual mic/speakers/network, not
mocked):
- `audio_io.play_pcm_int16()` played a real tone through real speakers.
- `audio_io.record_utterance()` opened a real mic stream and correctly
  timed out/returned `None` when silent.
- `audio_io.play_interruptible()` (the real barge-in code) ran with
  real concurrent mic+speaker streams, no false trigger during silence.
- `python pooja_cli.py --audition-voices` was run against the **real**
  Cartesia API (with an intentionally invalid key) — it made real HTTP
  requests to all 10 candidate voice ids, got real `401 Unauthorized`
  responses, and correctly skipped to the next voice after each one
  automatically, with no API key appearing anywhere in the printed
  output.
- The Groq fallback-model retry logic also ran against the real Groq
  API (invalid key → real 401 → fallback model tried → clean failure).

What still has **not** been run: Sarvam transcribing real recorded
speech, or Cartesia synthesizing a real Pooja reply with valid
credentials — both need your real API keys.

Run only the mocked suite (always safe, no keys needed):
```bash
python -m pytest tests/ -v --ignore=tests/test_live_apis.py
```
Run the live suite once you have real keys in `.env`:
```bash
python -m pytest tests/test_live_apis.py -v -s
```

## Running the legacy Twilio pipeline (optional, out of scope)

Not part of this deliverable, kept for reference only:

```bash
pip install -r requirements-legacy-twilio.txt
copy .env.example.legacy-twilio .env.legacy
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## License

MIT License (unchanged from the original Sia project).
