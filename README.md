# Pooja: AI Voice Agent for HostelPGDekho.com

Pooja is an AI voice agent that introduces HostelPGDekho.com to PG (paying-guest accommodation) owners. She holds a spoken conversation in **Hindi, English and Hinglish**, follows an approved sales script, answers questions, and ends the call on her own once the owner has decided.

It runs in two ways: from a terminal on your own computer, or in a browser with a Start call button and a live transcript.

**[Live demo](https://pooja-voice-agent.onrender.com/)** | **[Source code](https://github.com/ArpitaKhobragade/pooja-voice-agent)**

> The live demo asks for an access code, which protects the API credits behind it. Contact the author for the code. On a free hosting plan the demo can also take about a minute to wake up. See [Protect a public link](#protect-a-public-link).

## Highlights

- **Spoken, bilingual conversation.** Hindi, English and Hinglish, with yes / no / maybe answers understood in all three.
- **Deterministic call flow.** A state machine decides what happens next and when the call ends. The language model only proposes updates through validated tool calls, so one misheard word cannot change the outcome.
- **Interruption handling.** You can talk over Pooja and she stops and resumes gracefully.
- **Two interfaces.** A command-line version and a browser version with live status and a downloadable transcript.
- **Safe for public links.** Access code, call limits and per-call time caps protect your API credits.
- **Runs without API keys.** An offline mode walks through the script with no network.
- **Tested.** 107 automated tests pass without any network access.

## How it works

```text
microphone -> voice activity detection -> Sarvam speech-to-text -> script and state machine / Groq LLM -> Cartesia text-to-speech -> speaker
```

The browser version streams the same pipeline over a WebSocket:

```text
browser mic      --WebSocket-->  server: speech detection -> Sarvam STT -> Pooja's script / Groq LLM
browser speaker  <--WebSocket--  server: Cartesia voice (streamed) + transcript events
```

### Technology

| Layer | Technology | Notes |
|---|---|---|
| Language model | Groq, `openai/gpt-oss-120b` | Falls back once to `openai/gpt-oss-20b` if the primary call fails |
| Speech-to-text | Sarvam AI, `saaras:v3` (REST) | Hindi, English and Hinglish; up to about 30 seconds per utterance |
| Text-to-speech | Cartesia (official Python SDK), `sonic-3.5` | Required by default; `pyttsx3` only if you choose it explicitly |
| Voice activity detection | Silero (optional), then `webrtcvad` (default), then an energy threshold | Three-tier fallback; choose with `POOJA_VAD_BACKEND` |
| Audio input and output | `sounddevice` (PortAudio) | Microphone capture and playback, including capture during playback for interruptions |
| Web server | Python, FastAPI | Browser version and the voice audition page |
| Storage | SQLite | Call transcripts |
| Deployment | Docker, Render | `Dockerfile` and `render.yaml` included |

## Conversation flow

The conversation follows five phases. The conversation manager, not the language model, decides what each update means.

1. **Permission.** Greeting, then "is now a good time?"
2. **Smart Search.** The first of three benefits Pooja explains.
3. **Vacant-Bed Request Form.** Explained only after Pooja asks permission to discuss vacancies.
4. **Free listing.** Confirms the free-listing offer without adding conditions.
5. **Interest decision.** Yes, no or maybe, validated against a bilingual vocabulary. An unclear answer triggers a clarifying question instead of a guess.

Other behaviour:

- **Weak acknowledgements.** A bare "okay" or "haan" is treated as neutral. Pooja asks one short follow-up instead of continuing the pitch.
- **Repeat requests.** "Aap repeat kar sakte ho?" is handled without the model, so the exact last line is spoken again.
- **Opt-out.** Phrases like "stop calling" end the call before any model call is made, and keep working even if the model service is down.
- **App link.** Pooja reads out the app link and never claims that a message was sent.
- **Different endings.** Interested, not interested and busy each get their own closing line.
- **Failure handling.** If the language model fails completely, the call ends with a short apology instead of crashing. Malformed tool calls are rejected before they can change any state.

## Quick start

You need Python and, for the command-line version, a Windows computer with a microphone and speakers.

```bash
git clone https://github.com/ArpitaKhobragade/pooja-voice-agent.git
cd pooja-voice-agent\backend

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
notepad .env
```

Fill in these values in `.env`:

| Variable | Needed for |
|---|---|
| `GROQ_API_KEY` | Text and voice modes |
| `SARVAM_API_KEY` | Voice mode |
| `CARTESIA_API_KEY` | Voice mode (unless you use `--tts-engine pyttsx3`) |
| `CARTESIA_VOICE_ID` | Voice mode; find one with `--find-voice` |

Then:

```bash
python pooja_cli.py --find-voice       # list Hindi, feminine voices on your Cartesia account
python pooja_cli.py --list-devices     # check which microphone and speaker Windows reports
python pooja_cli.py --mode voice       # start talking to Pooja
```

### Run modes

| Mode | Needs | What it does |
|---|---|---|
| `--mode voice` | Microphone, speakers, all four keys above | Full pipeline: Sarvam, then Groq, then Cartesia |
| `--mode text` | `GROQ_API_KEY` only | Same conversation logic, typing instead of speaking |
| `--mode offline` | Nothing | Keyword-driven walkthrough of the script, no network |

With no `--mode`, Pooja picks the best option your configuration allows.

```bash
python pooja_cli.py --mode voice --tts-engine pyttsx3     # offline system voice instead of Cartesia
python pooja_cli.py --mode voice --allow-tts-fallback     # fall back per turn if Cartesia fails (always reported)
python pooja_cli.py --mode voice --no-barge-in            # turn off interruption detection
python pooja_cli.py --mode voice --debug                  # per-turn audio levels, transcripts and latencies
python pooja_cli.py --diagnose                            # step-by-step checks of mic, STT, LLM and TTS
```

Use headphones if you can. They stop Pooja's own voice from leaking into the microphone.

## Browser version

The browser version has **Start call** and **End call** buttons, a live status (listening, thinking, speaking), and a transcript of both sides that can be copied or downloaded. Interrupting Pooja works in the browser too, and the browser cancels echo, so headphones are optional.

### Run it locally

```bash
cd backend
pip install -r requirements-web.txt
python web_server.py          # then open http://localhost:8000 in Chrome or Edge
```

It reads the same `backend/.env` keys as the command-line version.

### Share a link

**Quick link (a few minutes, no account).** The link works while your computer, the server and the tunnel stay running.

1. Set `POOJA_ACCESS_CODE` in `backend/.env`.
2. Start the server with `python web_server.py`.
3. In a second window, install the tunnel tool once with `winget install --id Cloudflare.cloudflared`, then run `cloudflared tunnel --url http://localhost:8000`.
4. Share the `.trycloudflare.com` link it prints, together with the access code.

**Permanent link on Render.**

1. Push this repository to GitHub. Your `.env` is listed in `.gitignore`, so keys are never uploaded.
2. On [render.com](https://render.com) choose **New > Blueprint** and select the repository. Render reads `render.yaml`.
3. When asked, enter `GROQ_API_KEY`, `SARVAM_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` and a `POOJA_ACCESS_CODE`.
4. After the build you get a link such as `https://pooja-voice-agent.onrender.com`.

On the free plan the service sleeps after about 15 minutes without visitors and takes about a minute to wake up. Open the link a few minutes before a demo. A paid plan stays awake.

### Protect a public link

A public link lets anyone who finds it use your Groq, Sarvam and Cartesia credits.

| Setting | Default | Purpose |
|---|---|---|
| `POOJA_ACCESS_CODE` | none | Visitors must type it before a call can start. Always set it for a public link. |
| `POOJA_MAX_CALLS` | 2 | Calls allowed per link |
| `POOJA_WEB_MAX_CALL_S` | 420 | Maximum length of one web call, in seconds |

Credits to watch: Cartesia (each new sentence costs credits; sentences spoken before are cached in `backend/.tts_cache`), Groq (the free tier allows about 200,000 tokens a day) and Sarvam speech-to-text.

### Browser troubleshooting

- **"The server is missing settings"**: a required key is not set where the server runs. The banner shows names, never values.
- **"The voice service has run out of credits"**: top up Cartesia and start the call again.
- **No sound or blocked microphone**: allow the microphone in the browser's address bar. The page must be served over `https://` or `localhost`.
- **"Wrong code"**: enter the value of `POOJA_ACCESS_CODE`.

## Choosing a voice

Cartesia voice ids belong to your own account, so none ships with the project.

```bash
python pooja_cli.py --find-voice           # lists Hindi, feminine voices on your account
python pooja_cli.py --audition-voices      # plays Pooja's opening line in 10 voices, in the terminal
python pooja_cli.py --audition-voices-web  # the same in a browser at http://127.0.0.1:8765
```

The browser audition page has Play, Pause, Restart and Select buttons for each voice. Audio is generated once per voice and cached, so replaying never costs extra. **Save Selected Voice** changes only the `CARTESIA_VOICE_ID` line of `.env` and leaves every other line untouched. The page listens on localhost only, and your API key is never sent to the browser.

A new voice takes effect the next time you start `--mode voice`.

## Configuration

See `backend/.env.example` for the full list. Nothing is required for `--mode offline`.

| Variable | Default | Meaning |
|---|---|---|
| `CARTESIA_MODEL_ID` | `sonic-3.5` | Cartesia model |
| `CARTESIA_LANGUAGE` | `hi` | Voice language |
| `CARTESIA_SPEED` | `1.0` | Speaking speed, from 0.6 to 1.5 |
| `POOJA_MAX_TURNS` | 20 | Safety limit on the number of turns |
| `POOJA_SILENCE_TIMEOUT_S` | 1.5 | Silence that ends the owner's turn |
| `POOJA_PRE_SPEECH_TIMEOUT_S` | | Wait for the owner to start speaking |
| `POOJA_MAX_RECORD_S` | | Longest recording for one turn |
| `POOJA_MIN_SPEECH_S` | | Shortest clip treated as speech |
| `POOJA_SILENCE_RMS_THRESHOLD` | | Level below which a clip counts as silence |
| `POOJA_VAD_BACKEND` | auto | `silero`, `webrtcvad` or `energy` |
| `POOJA_INPUT_DEVICE`, `POOJA_OUTPUT_DEVICE` | system default | Microphone and speaker, by index or name |

Silero VAD is optional because it pulls in PyTorch. Install it with `pip install silero-vad torch`.

## Project structure

```text
backend/
├── pooja_cli.py            # Command-line entry point: checks, mode selection, conversation loop
├── web_server.py           # Browser version (FastAPI + WebSocket)
├── agent_config.py         # Persona, required slots and the script lines
├── prompts.py              # System prompt built from the configuration
├── objective.py            # Decides what Pooja should do next
├── conversation_manager.py # Validates tool calls and owns the end of the call
├── state_store.py          # Generic slot-filling state
├── local_tools.py          # Function-calling schema
├── local/
│   ├── audio_io.py         # Microphone, speaker, voice detection, interruptions
│   ├── stt_sarvam.py       # Sarvam speech-to-text client
│   ├── tts_cartesia.py     # Cartesia text-to-speech and the engine policy
│   ├── llm_groq.py         # Groq client with one bounded fallback
│   ├── voice_audition.py   # Terminal voice audition
│   ├── audition_server.py  # Browser voice audition server
│   ├── env_update.py       # Writes only CARTESIA_VOICE_ID to .env
│   └── offline_script.py   # Keyword-driven script with no network
├── tests/                  # Unit and integration tests
├── requirements.txt
└── .env.example
Dockerfile
render.yaml
```

Transcripts are saved to `backend/sia.db`, each under its own call id.

## Testing

```bash
cd backend
python -m pytest tests/ -v --ignore=tests/test_live_apis.py    # no keys, no network
python -m pytest tests/test_live_apis.py -v -s                  # real APIs, needs keys in .env
```

Current result: **107 passed, 4 skipped**. The 4 skipped tests call the live Groq, Sarvam and Cartesia APIs and are skipped when no keys are present.

| Test file | Covers |
|---|---|
| `test_core_logic.py` | State history, next-step selection, bilingual yes / no / maybe validation, rejection of bad tool calls |
| `test_offline_script.py` | The full script and every objection path |
| `test_pooja_session.py` | Tool-calling loop, repeat handling, interruptions, duplicate-response suppression, turn limit |
| `test_providers.py` | Sarvam and Cartesia request shapes, error handling, the text-to-speech policy |
| `test_llm_robustness.py` | Bounded model fallback, malformed tool-call rejection, graceful end on total failure |
| `test_stt_robustness.py` | Unreliable-transcript detection, silence and clipping analysis, device selection |
| `test_voice_audition.py`, `test_audition_server.py` | Voice audition flows; the API key never leaks |
| `test_env_update.py` | `.env` rewriting leaves every other line unchanged |
| `test_live_apis.py` | Real Groq, Sarvam and Cartesia calls |

## Demo walkthrough

A short script to try Pooja out, in `--mode voice` or in the browser:

1. Start the call. Pooja greets you and asks if it is a good time. Say **"Haan ji, bataiye."**
2. She introduces HostelPGDekho and explains Smart Search. Ask **"Aapka app kaise kaam karta hai?"** She should answer the question, not repeat the pitch.
3. Say **"Mera PG already full hai."** She should ask permission before explaining the Vacant-Bed Request Form.
4. Ask **"Listing free hai kya?"** She should confirm the free listing without adding conditions.
5. Interrupt her in the middle of a long sentence. She should stop and continue sensibly.
6. Ask **"Aap repeat kar sakte ho?"** She repeats her last line exactly.
7. Say **"App ka link bhej do."** She reads out the link and explains that she cannot send a message.
8. Say **"Haan, interested hoon."** She thanks you, closes the call in a few short sentences and ends it herself.

If you record the demo, use headphones and make sure your recorder captures both system audio and your microphone. Record a short test clip first.

## Known limitations

- **Interruptions are best effort.** Without echo cancellation, a laptop speaker can leak into its own microphone and cause false interruptions. Headphones make this reliable, or use `--no-barge-in`.
- **Not instant.** Each turn records until silence, transcribes, calls the model once or twice, then synthesizes speech. Expect a few seconds between your turn ending and Pooja's reply.
- **Clip length.** Sarvam's synchronous endpoint handles about 30 seconds per utterance.
- **Offline mode is keyword-based.** It is a script walkthrough, not language understanding.
- **Not a phone call.** Pooja talks through your microphone and speakers or the browser. There is no telephony or WhatsApp integration.

## Costs

- **Groq:** free credits on signup. `openai/gpt-oss-120b` costs about $0.15 per million input tokens and $0.60 per million output tokens; check the current price on the Groq console.
- **Sarvam AI:** free credits on signup.
- **Cartesia:** has a free tier. For a run that costs nothing, use `--tts-engine pyttsx3`.
- No subscription is required.

## Origin

Pooja grew out of **Sia**, a Twilio and Pipecat voice agent for an interior-design business. The deterministic conversation core (`ConversationManager`, `objective.py`, `agent_config.py`, `prompts.py`, `state_store.py`) kept its structure while the persona, script and slots were rewritten for HostelPGDekho.com. The local microphone and speaker pipeline and the browser version are new.

The original Twilio pipeline (`main.py`, `voice/bot.py`, `tools.py`, `extraction.py`) is kept for reference only and is not used by Pooja. To revive it, install `requirements-legacy-twilio.txt` and copy `.env.example.legacy-twilio` to `.env.legacy`.

## License

Released under the MIT License.
