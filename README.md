# H3 Video Gen

Local pipeline that uses:

- **Google Gemini** — film **Director** (shot list / prompts) and harsh **YouTube Critic** (vision review of frames)
- **MiniMax H3** via **ComfyUI** — clip generation
- **ElevenLabs** — reserved for narration (disabled for now)
- **FFmpeg** — frame extract + master assemble

## Quick start

1. Copy `.env.example` → `.env` and set `GEMINI_API_KEY`.
2. Ensure ComfyUI is running with H3 models (default `http://127.0.0.1:8188`).
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

## CLI

```powershell
.\.venv\Scripts\python run.py plan --prompt "A fox and a crow" --style "watercolor animation"
.\.venv\Scripts\python run.py generate --prompt "..." --style "..." --duration 60 --shots 10
.\.venv\Scripts\python run.py resume 20260804_181821_3d341b75
```

## Outputs

`E:\Programming\H3VideoGen\outputs\<project_id>\`

- `production.json` — director plan  
- `character_board/` — multi-view identity sheet + `sheet_manifest.json`  
- `takes/` — raw takes + review frames  
- `reviews/` — critic JSON per take  
- `shots/` — accepted clips  
- `master/` — YouTube-oriented concat  
- `state.json` — full run state + logs  

## Notes

- Critic threshold default **7.5/10** (see `.env` `CRITIC_PASS_THRESHOLD`). Below that → RETAKE.
- LLM fallback: **Gemini** (model cascade) → **local OpenAI-compatible** (Ollama/LM Studio) → **offline** template director + frame heuristic critic. Configure via `LLM_FALLBACK_ORDER`, `LOCAL_LLM_*`, `GEMINI_MODEL_FALLBACKS`.
- **Character consistency (R2V):** default `H3_MODE=r2v` uses MiniMax H3 reference-to-video (`H3_UNET_R2V`). For each main cast member the pipeline builds a **multi-view character sheet** (front, 3/4, side, face close-up, action — up to 5 for lead, 2 for support; configurable). Stills come from Gemini image models, optional H3 studio turnaround, manual drop-ins (`character_board/C01_front_full.png`, …), or bootstrap/enrich from accepted takes. Per shot the director/camera picks a relevant subset of views (H3 hard cap: **≤9** images; one slot often held for previous-shot continuity). Set `H3_MODE=t2v` for text-only.
- **Resume:** unfinished projects (cancelled / failed / partial) show **Resume** in the UI. Passed shots are skipped; character sheets and plan are reused. CLI: `run.py resume <project_id>`.
- **Auto-start services:** on Generate/Resume, if ComfyUI is down and `AUTO_START_COMFY=true`, the app launches `COMFYUI_ROOT` (`E:/AI/ComfyUI` by default) and waits up to `COMFY_START_TIMEOUT_SEC`. Ollama is started similarly when `AUTO_START_OLLAMA=true` and `LOCAL_LLM_*` points at port 11434. Manual: `POST /api/services/ensure`.
- Voice: `ENABLE_VOICE=false` by default; `app/agents/voice.py` is a stub for ElevenLabs later.
- One generation job at a time (VRAM safety).
- Logs restore after browser refresh (from `state.json` + active job poll).
