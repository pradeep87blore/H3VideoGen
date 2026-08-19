# H3 Video Gen

Local pipeline that uses:

- **Google Gemini** — film **Director** (shot list / prompts) and harsh **YouTube Critic** (vision review of frames)
- **MiniMax H3** via **ComfyUI** — clip generation
- **ElevenLabs** — reserved for narration (disabled for now)
- **FFmpeg** — frame extract + master assemble

## Quick start

**Windows (recommended):** double-click `launch.bat`  
It creates `.venv`, installs deps if needed, ensures `.env` exists, starts the server, and opens **http://127.0.0.1:7860**.

Or manually:

1. Copy `.env.example` → `.env` and set `GEMINI_API_KEY`.
2. Ensure ComfyUI is available with H3 models (default `http://127.0.0.1:8188`; Generate/Resume can auto-start it).
3. Create a venv and install deps:

```powershell
cd E:\Programming\H3VideoGen
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python run.py serve
```

4. Open **http://127.0.0.1:7860**

## UI

- **Story prompt** — what the video is about
- **Visual style** — art direction (kept separate from story)
- **Director plan only** — Gemini planning without GPU
- **Generate full video** — plan → H3 each shot → Gemini critic (+ retakes) → FFmpeg master
- **Resume** on a project card — continue from unfinished shots (keeps character board + passed clips). Auto-starts **ComfyUI** / **Ollama** if down.

## CLI / automation

Same fields as the UI (`prompt`, `style`, `target_duration_sec`, `max_shots`, `max_retakes`, …). Use **CLI flags**, a **JSON job file**, or the web UI — flags override JSON when both are set. `POST /api/generate` accepts the same JSON body.

```powershell
.\.venv\Scripts\python run.py plan --prompt "A fox and a crow" --style "watercolor animation"
.\.venv\Scripts\python run.py generate --prompt "..." --style "..." --duration 60 --shots 10
.\.venv\Scripts\python run.py generate --json jobs\tiny_bunny_origami.json
.\.venv\Scripts\python run.py generate -j jobs\my_job.json --retakes 1
.\.venv\Scripts\python run.py resume 20260804_181821_3d341b75
```

**ComfyUI note:** Generate needs a Comfy install with native MiniMax H3 nodes (`MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`). If something else is listening on port 8188 without those nodes, set `COMFYUI_ROOT` (default `E:/AI/ComfyUI`) and leave `COMFY_REPLACE_NON_H3=true` so the app can free the port and start the correct install.

## Outputs

`E:\Programming\H3VideoGen\outputs\<project_id>\`

- `production.json` — director plan  
- `character_board/` — multi-view identity sheet + `sheet_manifest.json`  
- `takes/` — raw takes + review frames  
- `reviews/` — critic JSON per take  
- `shots/` — accepted clips  
- `master/` — YouTube-oriented concat  
- `state.json` — full run state (including log array)  
- `run.log` — plain-text live log (same content as the UI log window)  

## Notes

- Critic threshold default **7.5/10** (see `.env` `CRITIC_PASS_THRESHOLD`). Below that → RETAKE.
- LLM fallback: **Gemini** (model cascade) → **local OpenAI-compatible** (Ollama/LM Studio) → **offline** template director + frame heuristic critic. Configure via `LLM_FALLBACK_ORDER`, `LOCAL_LLM_*`, `GEMINI_MODEL_FALLBACKS`.
  - Local setup: `ollama pull llama3.2` (director text) and `ollama pull qwen2.5vl` (critic vision; better fallback than LLaVA). Set `LOCAL_LLM_MODEL` + `LOCAL_LLM_VISION_MODEL` — text-only models cannot review frames; vision route uses Ollama `/api/chat` with `format=json` + images.
- **Narrative modes:** `character` (cast + R2V sheets), `documentary` (events/history, T2V, no cast sheets), `explainer` (concepts/metaphors, T2V). CLI: `--narrative-mode` or JSON `narrative_mode`. UI: dropdown on Create.
- **Narration:** when `ENABLE_VOICE=true` and ElevenLabs is configured, a documentary-style VO (`ELEVENLABS_VOICE_ID`) is mixed under the master for all modes.
- **Character consistency (R2V):** default `H3_MODE=r2v` uses MiniMax H3 reference-to-video (`H3_UNET_R2V`). For each main cast member the pipeline builds a **multi-view character sheet** (front, 3/4, side, face close-up, action — up to 5 for lead, 2 for support; configurable). Stills come from Gemini image models, optional H3 studio turnaround, manual drop-ins (`character_board/C01_front_full.png`, …), or bootstrap/enrich from accepted takes. Per shot the director/camera picks a relevant subset of views (H3 hard cap: **≤9** images). Continuity between shots uses the previous accepted **clip as `<Video 1>`** (motion/camera) and, on T2V, **FL2VA keyframes** (previous last frame → this shot’s preclip still). Set `H3_MODE=t2v` for text-only. Native stereo SFX/ambience is prompted in-graph (`H3_PROMPT_NATIVE_AUDIO`); ElevenLabs remains the narrator bed when `ENABLE_VOICE=true`.
- **Resume:** unfinished projects (cancelled / failed / partial) show **Resume** in the UI. Passed shots are skipped; character sheets and plan are reused. CLI: `run.py resume <project_id>`.
- **Auto-start services:** on Generate/Resume, if ComfyUI is down and `AUTO_START_COMFY=true`, the app launches `COMFYUI_ROOT` (`E:/AI/ComfyUI` by default) and waits up to `ESSENTIALS_WAIT_SEC` / `COMFY_START_TIMEOUT_SEC` (default **5 minutes**). If still down, generation **stops with a clear error** (no multi-hour hang). The UI shows a red banner + browser alert when ComfyUI, FFmpeg, or both Gemini/Ollama are missing. Ollama starts when `AUTO_START_OLLAMA=true`. Manual: `POST /api/services/ensure` or **Start services** on the banner.
- **Fail-fast shots:** if a shot fails after all retakes (or hard gen error), the **job stops** — remaining shots are skipped and no master is assembled.
- **Job queue:** Generate/Resume enqueue; default **1 parallel** (`MAX_PARALLEL_JOBS` / UI “Parallel jobs”). Raise to run multiple at once (share Comfy/GPU carefully). UI **Jobs** tabs switch log/cast per job; Stop targets the selected tab.
- Voice: `ENABLE_VOICE=false` by default; `app/agents/voice.py` is a stub for ElevenLabs later.
- Logs restore after browser refresh (from `state.json` + active job poll).
