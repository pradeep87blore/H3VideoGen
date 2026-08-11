const $ = (id) => document.getElementById(id);
const logEl = $("log");
const planEl = $("plan");
const resultEl = $("result");
const projectsEl = $("projects");
const healthEl = $("health");

const STORAGE_KEY = "h3vg_job";
const HEARTBEAT_MS = 10000;
let essentialsDismissedKey = "";
let essentialsAlertedKey = "";
let lastHeartbeatAt = null;
let heartbeatTimer = null;

function appendLog(line) {
  if (logEl.textContent === "Idle. Plan or generate to begin.") logEl.textContent = "";
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function setLogLines(lines) {
  if (!lines || !lines.length) return;
  logEl.textContent = lines.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

function setBusy(busy, { keepGenerate = false } = {}) {
  // Generate stays available so more jobs can be queued while others run.
  $("btn-run").disabled = keepGenerate ? false : busy;
  $("btn-plan").disabled = busy && !keepGenerate;
  const stop = $("btn-stop");
  if (stop) stop.disabled = !busy;
}

function isLiveStatus(status) {
  return ["running", "planning", "assembling", "generating", "reviewing", "cancelling", "queued"].includes(
    String(status || "").toLowerCase()
  );
}

function isWorkerLiveStatus(status) {
  return ["running", "planning", "assembling", "generating", "reviewing", "cancelling"].includes(
    String(status || "").toLowerCase()
  );
}

function isTerminalStatus(status) {
  return ["completed", "completed_no_assemble", "failed", "error", "cancelled"].includes(
    String(status || "").toLowerCase()
  );
}

function saveSession(partial) {
  try {
    const prev = loadSession() || {};
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ ...prev, ...partial }));
  } catch (_) {
    /* private mode etc. */
  }
}

function loadSession() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
}

function clearSession() {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch (_) {
    /* ignore */
  }
}

function boardFileUrl(projectId, relOrAbs) {
  if (!projectId || !relOrAbs || String(projectId).startsWith("pending_")) return null;
  const norm = String(relOrAbs).replace(/\\/g, "/");
  let rel = norm;
  const marker = "character_board/";
  const idx = norm.toLowerCase().indexOf(marker);
  if (idx >= 0) {
    rel = norm.slice(idx);
  } else if (!norm.includes("/")) {
    rel = "character_board/" + norm;
  } else {
    const base = norm.split("/").pop();
    if (!base) return null;
    rel = "character_board/" + base;
  }
  return `/api/projects/${encodeURIComponent(projectId)}/file?path=${encodeURIComponent(rel)}`;
}

function formatDuration(sec) {
  if (sec == null || sec === "" || Number.isNaN(Number(sec))) return "—";
  const s = Math.max(0, Number(sec));
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s - m * 60);
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function formatClock(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return String(iso).slice(11, 19) || iso;
  try {
    return new Date(t).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch (_) {
    return iso;
  }
}

function stagePriority(key) {
  const k = String(key || "");
  if (k === "total") return 0;
  if (k === "essentials") return 10;
  if (k === "director") return 20;
  if (k === "character_sheets") return 30;
  if (k.startsWith("char:")) return 31;
  if (k === "shots") return 40;
  if (k.startsWith("shot:")) return 41;
  if (k === "assemble") return 50;
  return 60;
}

function renderJobStatus(job) {
  const body = $("job-status-body");
  const meta = $("job-meta");
  if (!body) return;

  if (!job) {
    if (meta) meta.textContent = "No active job.";
    body.innerHTML =
      '<tr class="empty-row"><td colspan="5">Stage timings appear when a job runs.</td></tr>';
    return;
  }

  const pid = job.project_id || "";
  const live = isLiveStatus(job.status);
  const start = job.job_started_at || job.created_at;
  if (meta) {
    const title = job.title ? `“${job.title}” · ` : "";
    meta.textContent = live
      ? `${title}${pid || "job"} · started ${formatClock(start)} · ${String(job.status || "running")}`
      : `${title}${pid || "job"} · ${String(job.status || "idle")}` +
        (job.job_finished_at ? ` · finished ${formatClock(job.job_finished_at)}` : "");
  }

  let stages = Array.isArray(job.stage_timings) ? [...job.stage_timings] : [];
  // Synthesize char rows from cast timing if stage list lacks them
  if (Array.isArray(job.characters)) {
    for (const c of job.characters) {
      const key = `char:${c.id}`;
      if (stages.some((s) => s.key === key)) continue;
      if (c.sheet_duration_sec == null && !c.sheet_started_at) continue;
      stages.push({
        key,
        label: `Sheet · ${c.name || c.id}`,
        started_at: c.sheet_started_at,
        ended_at: c.sheet_finished_at,
        duration_sec: c.sheet_duration_sec,
        status: c.sheet_status === "ready" ? "done" : c.sheet_status || "pending",
        detail: `${c.sheet_count || 0} view(s)${c.sheet_source ? " · " + c.sheet_source : ""}`,
      });
    }
  }
  stages.sort((a, b) => {
    const pa = stagePriority(a.key);
    const pb = stagePriority(b.key);
    if (pa !== pb) return pa - pb;
    return String(a.key).localeCompare(String(b.key));
  });

  if (!stages.length) {
    body.innerHTML =
      '<tr class="empty-row"><td colspan="5">Waiting for pipeline stages…</td></tr>';
    return;
  }

  body.innerHTML = stages
    .map((row) => {
      const st = String(row.status || "pending").toLowerCase();
      let dur = row.duration_sec;
      if (st === "running" && row.started_at && (dur == null || live)) {
        const t0 = Date.parse(row.started_at);
        if (!Number.isNaN(t0)) dur = (Date.now() - t0) / 1000;
      }
      const indent = String(row.key || "").includes(":") ? " stage-indent" : "";
      return `<tr class="stage-${escapeHtml(st)}${indent}">
        <td class="stage-label">${escapeHtml(row.label || row.key)}</td>
        <td class="mono">${escapeHtml(formatClock(row.started_at))}</td>
        <td class="mono">${escapeHtml(formatDuration(dur))}${
          st === "running" ? " …" : ""
        }</td>
        <td><span class="stage-badge stage-badge-${escapeHtml(st)}">${escapeHtml(st)}</span></td>
        <td class="stage-detail">${escapeHtml(row.detail || "")}</td>
      </tr>`;
    })
    .join("");
}

function renderCast(characters, projectId) {
  const el = $("cast-list");
  if (!el) return;
  const list = Array.isArray(characters) ? characters : [];
  if (!list.length) {
    el.innerHTML =
      '<p class="hint cast-empty">Characters appear here after the director plans the cast.</p>';
    return;
  }
  el.innerHTML = list
    .map((c, i) => {
      const id = c.id || `C${String(i + 1).padStart(2, "0")}`;
      const name = c.name || id;
      const look = c.look || "";
      const sheets = c.sheet_count != null ? c.sheet_count : null;
      const status = c.sheet_status || "";
      const finalized =
        status === "ready" || !!(c.image_path || c.thumb_path || (c.sheet_thumbs && c.sheet_thumbs.length));
      const initial = escapeHtml((name || "?").slice(0, 1).toUpperCase());
      let avatar = `<div class="cast-avatar" aria-hidden="true">${initial}</div>`;
      const thumbSrc =
        boardFileUrl(projectId, c.thumb_path) ||
        boardFileUrl(projectId, c.image_path) ||
        (c.sheet_thumbs && c.sheet_thumbs[0]
          ? boardFileUrl(projectId, c.sheet_thumbs[0].path)
          : null);
      if (finalized && thumbSrc) {
        avatar = `<div class="cast-avatar has-img"><img src="${thumbSrc}" alt="" loading="lazy" /></div>`;
      }
      let mini = "";
      if (finalized && Array.isArray(c.sheet_thumbs) && c.sheet_thumbs.length > 1 && projectId) {
        mini = `<div class="cast-thumbs">${c.sheet_thumbs
          .slice(0, 4)
          .map((t) => {
            const src = boardFileUrl(projectId, t.path);
            if (!src) return "";
            return `<img src="${src}" alt="${escapeHtml(t.pose_id || "")}" title="${escapeHtml(
              t.label || t.pose_id || ""
            )}" loading="lazy" />`;
          })
          .join("")}</div>`;
      }
      const timing =
        c.sheet_duration_sec != null
          ? `<p class="cast-time">Sheet ${formatDuration(c.sheet_duration_sec)}${
              c.sheet_source ? " · " + escapeHtml(c.sheet_source) : ""
            }</p>`
          : status === "building"
            ? `<p class="cast-time">Building sheet…</p>`
            : "";
      return `<article class="cast-card${finalized ? " ready" : ""}">
        ${avatar}
        <div class="cast-meta">
          <div class="cast-id">${escapeHtml(id)}${
            status ? ` · ${escapeHtml(status)}` : ""
          }</div>
          <p class="cast-name">${escapeHtml(name)}</p>
          ${look ? `<p class="cast-look">${escapeHtml(look)}</p>` : ""}
          ${
            sheets != null
              ? `<p class="cast-sheet">${sheets} sheet view${sheets === 1 ? "" : "s"}</p>`
              : ""
          }
          ${timing}
          ${mini}
        </div>
      </article>`;
    })
    .join("");
}

function charactersFromPlan(plan) {
  if (!plan || !Array.isArray(plan.characters)) return [];
  return plan.characters.map((c) => {
    const sheet = Array.isArray(c.sheet) ? c.sheet : [];
    return {
      id: c.id,
      name: c.name,
      look: c.look || "",
      image_path: c.image_path,
      thumb_path: c.image_path,
      sheet_count: sheet.filter((p) => p && p.image_path).length,
      sheet_status: c.sheet_status,
      sheet_duration_sec: c.sheet_duration_sec,
      sheet_source: c.sheet_source,
      sheet_started_at: c.sheet_started_at,
      sheet_finished_at: c.sheet_finished_at,
      sheet_thumbs: sheet
        .filter((p) => p && p.image_path)
        .map((p) => ({
          pose_id: p.pose_id,
          label: p.label || p.pose_id,
          path: p.image_path,
        })),
    };
  });
}

function essentialsSignature(ess) {
  if (!ess) return "";
  return [...(ess.blocking || []), ...(ess.warnings || [])].join("|");
}

function showEssentialsBanner(ess, { force = false } = {}) {
  const banner = $("essentials-banner");
  const body = $("essentials-body");
  const title = $("essentials-title");
  if (!banner || !body || !title) return;

  const blocking = ess?.blocking || [];
  const warnings = ess?.warnings || [];
  if (!blocking.length && !warnings.length) {
    banner.classList.add("hidden");
    essentialsDismissedKey = "";
    return;
  }

  const sig = essentialsSignature(ess);
  if (!force && sig && sig === essentialsDismissedKey) {
    banner.classList.add("hidden");
    return;
  }

  const waitMin = Math.max(1, Math.round((ess.wait_limit_sec || 300) / 60));
  title.textContent = blocking.length
    ? `Prerequisites missing — self-healing (auto-start, ≤~${waitMin} min)`
    : "Prerequisites: soft warnings";
  body.textContent =
    ess.prompt ||
    [...blocking.map((b) => "• " + b), ...warnings.map((w) => "• " + w)].join("\n");
  banner.classList.toggle("warn", !blocking.length);
  banner.classList.remove("hidden");

  // Silent self-heal: auto-start missing services instead of blocking on alert()
  if (blocking.length && sig !== essentialsAlertedKey) {
    essentialsAlertedKey = sig;
    appendLog(
      "Prerequisites missing — self-healing services in the background (no user action required)…"
    );
    startEssentialServices().catch(() => {});
  }
}

function formatAgo(checkedAt) {
  if (!checkedAt) return "—";
  const t = Date.parse(checkedAt);
  if (Number.isNaN(t)) return "just now";
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (sec < 2) return "just now";
  if (sec < 60) return `${sec}s ago`;
  return `${Math.floor(sec / 60)}m ago`;
}

function shortToolName(name) {
  const n = String(name || "");
  if (/comfy/i.test(n)) return "ComfyUI";
  if (/ffmpeg/i.test(n)) return "FFmpeg";
  if (/gemini/i.test(n)) return "Gemini";
  if (/ollama|local llm/i.test(n)) return "Ollama";
  return n.split("(")[0].trim() || n;
}

function renderHeartbeat(payload) {
  const toolsEl = $("heartbeat-tools");
  const pulseEl = $("heartbeat-pulse");
  const agoEl = $("heartbeat-ago");
  if (!toolsEl) return;

  const hb = payload?.heartbeat || {};
  const tools = Array.isArray(hb.tools) && hb.tools.length
    ? hb.tools
    : payload?.essentials?.services || [];
  lastHeartbeatAt = hb.checked_at || new Date().toISOString();
  if (agoEl) agoEl.textContent = `♥ ${formatAgo(lastHeartbeatAt)} · 10s`;

  if (!tools.length) {
    toolsEl.innerHTML = '<span class="tool-chip muted">No tool status</span>';
    if (pulseEl) pulseEl.className = "heartbeat-pulse down";
    return;
  }

  const anyRequiredDown = tools.some((t) => t.required && !t.ok);
  const anyWarn = tools.some((t) => !t.required && !t.ok);
  if (pulseEl) {
    pulseEl.className =
      "heartbeat-pulse " + (anyRequiredDown ? "down" : anyWarn ? "warn" : "live");
  }

  toolsEl.innerHTML = tools
    .map((t) => {
      const req = !!t.required;
      const ok = !!t.ok;
      const cls = ok ? "ok" : req ? "down" : "warn";
      const status = ok ? "up" : req ? "down" : "warn";
      const detail = t.detail || t.fix || "";
      const title = [
        t.name,
        status,
        req ? "required" : "optional",
        detail,
        t.fix && !ok ? t.fix : "",
      ]
        .filter(Boolean)
        .join(" · ");
      return `<span class="tool-chip ${cls}" title="${escapeHtml(title)}">
        <span class="dot"></span>
        <span class="tool-name">${escapeHtml(shortToolName(t.name))}</span>
        <span class="tool-detail">${escapeHtml(status)}${detail ? " · " + escapeHtml(String(detail).slice(0, 40)) : ""}</span>
      </span>`;
    })
    .join("");
}

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    const bits = [];
    bits.push(j.gemini_key_set ? "Gemini key ✓" : "Gemini key missing");
    bits.push(j.comfy_ok ? "ComfyUI ✓" : "ComfyUI ✗");
    bits.push(j.ollama_ok ? "Ollama ✓" : "Ollama ·");
    bits.push(j.voice_enabled ? "Voice on" : "Voice off");
    if (j.essentials?.ready_for_generate === false) bits.push("Not ready");
    else if (j.essentials?.ready_for_generate) bits.push("Ready");
    healthEl.textContent = bits.join(" · ");
    if (!j.comfy_ok || (j.essentials && !j.essentials.ok)) {
      healthEl.className = "health bad";
    } else if (j.essentials?.warnings?.length) {
      healthEl.className = "health warn";
    } else {
      healthEl.className = "health ok";
    }
    renderHeartbeat(j);
    if (j.essentials) showEssentialsBanner(j.essentials);
  } catch (e) {
    healthEl.textContent = "API unreachable";
    healthEl.className = "health bad";
    const toolsEl = $("heartbeat-tools");
    const pulseEl = $("heartbeat-pulse");
    if (toolsEl) {
      toolsEl.innerHTML =
        '<span class="tool-chip down"><span class="dot"></span><span class="tool-name">API</span><span class="tool-detail">down</span></span>';
    }
    if (pulseEl) pulseEl.className = "heartbeat-pulse down";
    if ($("heartbeat-ago")) $("heartbeat-ago").textContent = "error · 10s";
  }
}

function startHeartbeat() {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  refreshHealth();
  heartbeatTimer = setInterval(refreshHealth, HEARTBEAT_MS);
  // keep "Ns ago" fresh without re-probing tools
  setInterval(() => {
    const agoEl = $("heartbeat-ago");
    if (agoEl && lastHeartbeatAt) {
      agoEl.textContent = `♥ ${formatAgo(lastHeartbeatAt)} · 10s`;
    }
  }, 1000);
}

async function startEssentialServices() {
  appendLog("Starting ComfyUI / Ollama if needed (wait up to ~5 min)…");
  try {
    const r = await fetch("/api/services/ensure", { method: "POST" });
    const j = await r.json();
    (j.log || []).forEach((line) => appendLog(line));
    if (j.essentials?.prompt) appendLog(j.essentials.prompt.replace(/\n/g, " | "));
    if (j.ok && j.comfy_ok) {
      appendLog("Essentials look ready.");
      essentialsDismissedKey = "";
      essentialsAlertedKey = "";
    } else {
      const wait = j.essentials?.wait_limit_sec || 300;
      appendLog(
        `Still missing essentials after auto-start attempt (limit ~${Math.round(wait / 60)} min). ` +
          (j.essentials?.blocking || []).join(" · ")
      );
      if (j.essentials) showEssentialsBanner(j.essentials, { force: true });
    }
    await refreshHealth();
  } catch (e) {
    appendLog("Start services failed: " + e.message);
  }
}

async function refreshProjects() {
  try {
    const r = await fetch("/api/projects");
    const list = await r.json();
    if (!list.length) {
      projectsEl.innerHTML = "<p class='hint'>No projects yet.</p>";
      return;
    }
    projectsEl.innerHTML = list
      .map((p) => {
        const title = p.title || p.project_id;
        const master = p.master_path
          ? `<a href="/api/projects/${encodeURIComponent(p.project_id)}/master" target="_blank">Download master</a>`
          : "";
        const progress =
          p.shots_total != null
            ? `${p.shots_passed || 0}/${p.shots_total} shots`
            : "";
        const resumeBtn = p.resumable
          ? `<button class="btn primary sm" data-resume="${escapeHtml(p.project_id)}" title="Continue unfinished shots">Resume</button>`
          : "";
        return `<article class="card">
          <div>
            <h3>${escapeHtml(title)}</h3>
            <p>${escapeHtml(p.prompt || "")}</p>
            <p style="margin-top:0.4rem">${progress ? escapeHtml(progress) + " · " : ""}${master}
            ${master && resumeBtn ? " · " : ""}${resumeBtn || ""}
             · <button class="btn ghost sm" data-id="${escapeHtml(p.project_id)}">Open details</button></p>
          </div>
          <span class="badge ${escapeHtml(p.status || "")}">${escapeHtml(p.status || "?")}</span>
        </article>`;
      })
      .join("");
    projectsEl.querySelectorAll("button[data-id]").forEach((btn) => {
      btn.addEventListener("click", () => openProject(btn.dataset.id));
    });
    projectsEl.querySelectorAll("button[data-resume]").forEach((btn) => {
      btn.addEventListener("click", () => resumeProject(btn.dataset.resume));
    });
  } catch (e) {
    projectsEl.textContent = "Failed to load projects: " + e;
  }
}

async function resumeProject(projectId) {
  if (!projectId) return;
  resultEl.classList.add("hidden");
  appendLog(`Queuing resume for ${projectId}…`);
  try {
    await syncParallelJobs();
    const r = await fetch(`/api/projects/${encodeURIComponent(projectId)}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_retakes: Number($("max_retakes").value || 2),
        auto_assemble: true,
        seed_base: 42,
        redo_failed: true,
      }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    appendLog(j.message || "Resume queued");
    watchingId = j.job_ref || projectId;
    selectedTabKey = j.job_ref || projectId;
    saveSession({ job_ref: watchingId, watching: true, last_project_id: projectId });
    startPolling(watchingId);
  } catch (e) {
    appendLog("Resume failed: " + e.message);
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function showMaster(projectId, masterPath) {
  if (!masterPath) return;
  resultEl.classList.remove("hidden");
  resultEl.innerHTML = `<strong>Master:</strong> <a href="/api/projects/${encodeURIComponent(projectId)}/master" target="_blank">${escapeHtml(masterPath)}</a>`;
}

async function openProject(id) {
  const r = await fetch(`/api/projects/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error("Project not found");
  const j = await r.json();
  lastUiProjectId = id;
  planEl.textContent = JSON.stringify(j.plan || j, null, 2);
  renderCast(charactersFromPlan(j.plan) || j.characters || [], id);
  setLogLines(j.log || []);
  if (!j.log || !j.log.length) {
    logEl.textContent = "No logs stored for this project.";
  }
  // Project detail returns full state — map to job status shape
  renderJobStatus({
    project_id: j.project_id || id,
    status: j.status,
    title: j.plan?.title,
    stage_timings: j.stage_timings || [],
    characters: charactersFromPlan(j.plan),
    job_started_at: j.job_started_at || j.created_at,
    job_finished_at: j.job_finished_at,
    created_at: j.created_at,
  });
  if (j.master_path) {
    showMaster(id, j.master_path);
  } else {
    resultEl.classList.add("hidden");
  }
  saveSession({ last_project_id: id, job_ref: id });
  return j;
}

function formPayload() {
  const slug = ($("style_slug") && $("style_slug").value.trim()) || "";
  const payload = {
    prompt: $("prompt").value.trim(),
    style: $("style").value.trim(),
    target_duration_sec: Number($("duration").value || 60),
    max_shots: Number($("max_shots").value || 12),
    max_retakes: Number($("max_retakes").value || 2),
    narrative_mode: ($("narrative_mode") && $("narrative_mode").value) || "character",
    auto_assemble: true,
    seed_base: 42,
  };
  if (slug) payload.style_slug = slug;
  return payload;
}

/* ---------- Style library picker ---------- */
let styleLibrary = null;
let styleFilterCat = "all";
let styleSearchQ = "";

function openStyleModal() {
  const modal = $("style-modal");
  if (!modal) return;
  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  loadStyleLibrary()
    .then(() => {
      renderStyleFilters();
      renderStyleGrid();
      $("style-search")?.focus();
    })
    .catch((e) => {
      appendLog("Style library error: " + (e.message || e));
      closeStyleModal();
    });
}

function closeStyleModal() {
  const modal = $("style-modal");
  if (!modal) return;
  modal.classList.add("hidden");
  document.body.style.overflow = "";
}

async function loadStyleLibrary() {
  if (styleLibrary && styleLibrary.styles?.length) return styleLibrary;
  const r = await fetch("/api/styles");
  if (!r.ok) throw new Error("Could not load style library");
  styleLibrary = await r.json();
  return styleLibrary;
}

function renderStyleFilters() {
  const el = $("style-cat-filters");
  if (!el || !styleLibrary) return;
  const cats = ["all", ...(styleLibrary.categories || [])];
  el.innerHTML = cats
    .map((c) => {
      const label = c === "all" ? "All" : c.replace(/ Styles$/i, "");
      const active = styleFilterCat === c ? " active" : "";
      return `<button type="button" class="style-cat-btn${active}" data-cat="${escapeHtml(
        c
      )}">${escapeHtml(label)}</button>`;
    })
    .join("");
  el.querySelectorAll("button[data-cat]").forEach((btn) => {
    btn.addEventListener("click", () => {
      styleFilterCat = btn.dataset.cat || "all";
      renderStyleFilters();
      renderStyleGrid();
    });
  });
}

function filteredStyles() {
  if (!styleLibrary?.styles) return [];
  const q = styleSearchQ.trim().toLowerCase();
  return styleLibrary.styles.filter((s) => {
    if (styleFilterCat !== "all" && s.category !== styleFilterCat) return false;
    if (!q) return true;
    const hay = `${s.name} ${s.category} ${s.slug} ${s.description || ""}`.toLowerCase();
    return hay.includes(q);
  });
}

function renderStyleGrid() {
  const el = $("style-grid");
  if (!el) return;
  const selected = ($("style_slug") && $("style_slug").value) || "";
  const list = filteredStyles();
  if (!list.length) {
    el.innerHTML = '<p class="style-grid-empty">No styles match that filter.</p>';
    return;
  }
  el.innerHTML = list
    .map((s) => {
      const sel = s.slug === selected ? " selected" : "";
      const cat = (s.category || "").replace(/ Styles$/i, "");
      return `<button type="button" class="style-card${sel}" data-slug="${escapeHtml(
        s.slug
      )}" title="${escapeHtml(s.name)}">
        <img src="${escapeHtml(s.thumb_url)}?v=3" alt="" loading="lazy" />
        <div class="style-card-body">
          <strong>${escapeHtml(s.name)}</strong>
          <span>${escapeHtml(cat)}</span>
        </div>
      </button>`;
    })
    .join("");
  el.querySelectorAll("button[data-slug]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const slug = btn.dataset.slug;
      const style = styleLibrary.styles.find((x) => x.slug === slug);
      if (style) applyLibraryStyle(style);
      closeStyleModal();
    });
  });
}

function applyLibraryStyle(style) {
  if ($("style")) $("style").value = style.style_prompt || style.sample_prompt || "";
  if ($("style_slug")) $("style_slug").value = style.slug || "";
  const picked = $("style-picked");
  if (picked) {
    picked.classList.remove("hidden");
    if ($("style-picked-thumb")) {
      $("style-picked-thumb").src = (style.thumb_url || "") + "?v=3";
      $("style-picked-thumb").alt = style.name || "";
    }
    if ($("style-picked-name")) $("style-picked-name").textContent = style.name || style.slug;
    if ($("style-picked-cat")) $("style-picked-cat").textContent = style.category || "";
  }
  appendLog(`Style locked: ${style.name || style.slug}`);
  try {
    sessionStorage.setItem(
      "h3vg_style",
      JSON.stringify({ slug: style.slug, name: style.name, thumb_url: style.thumb_url })
    );
  } catch (_) {
    /* ignore */
  }
}

function clearLibraryStyle() {
  if ($("style_slug")) $("style_slug").value = "";
  $("style-picked")?.classList.add("hidden");
  try {
    sessionStorage.removeItem("h3vg_style");
  } catch (_) {
    /* ignore */
  }
}

function restorePickedStyleChip() {
  try {
    const raw = sessionStorage.getItem("h3vg_style");
    if (!raw) return;
    const s = JSON.parse(raw);
    if (!s?.slug) return;
    if ($("style_slug")) $("style_slug").value = s.slug;
    const picked = $("style-picked");
    if (picked) {
      picked.classList.remove("hidden");
      if ($("style-picked-thumb") && s.thumb_url) {
        $("style-picked-thumb").src = s.thumb_url + "?v=3";
        $("style-picked-thumb").alt = s.name || "";
      }
      if ($("style-picked-name")) $("style-picked-name").textContent = s.name || s.slug;
    }
  } catch (_) {
    /* ignore */
  }
}

$("btn-open-styles")?.addEventListener("click", async () => {
  try {
    openStyleModal();
  } catch (e) {
    appendLog("Style library error: " + (e.message || e));
  }
});

document.querySelectorAll("[data-close-styles]").forEach((el) => {
  el.addEventListener("click", closeStyleModal);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("style-modal")?.classList.contains("hidden")) {
    closeStyleModal();
  }
});
$("style-search")?.addEventListener("input", (e) => {
  styleSearchQ = e.target.value || "";
  renderStyleGrid();
});
$("btn-clear-style")?.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  clearLibraryStyle();
  appendLog("Style library selection cleared (custom text kept).");
});
restorePickedStyleChip();

$("btn-plan").addEventListener("click", async () => {
  const body = formPayload();
  if (!body.prompt) return alert("Enter a story prompt");
  $("btn-plan").disabled = true;
  appendLog("Requesting director plan…");
  try {
    const r = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: body.prompt,
        style: body.style,
        target_duration_sec: body.target_duration_sec,
        max_shots: body.max_shots,
        narrative_mode: body.narrative_mode,
      }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    planEl.textContent = JSON.stringify(j, null, 2);
    renderCast(charactersFromPlan(j), null);
    appendLog(
      `Plan: “${j.title}” — ${j.shots?.length || 0} shots, ~${j.target_duration_sec}s · mode=${j.narrative_mode || "character"}`
    );
    if (j.characters?.length) {
      appendLog(`Cast: ${j.characters.map((c) => c.name || c.id).join(", ")}`);
    }
  } catch (e) {
    appendLog("Plan failed: " + e.message);
  } finally {
    $("btn-plan").disabled = false;
  }
});

let pollTimer = null;
let watchingId = null;
let selectedTabKey = null;
let lastJobsPayload = null;

async function syncParallelJobs() {
  const el = $("parallel_jobs");
  if (!el) return;
  const n = Math.max(1, Math.min(8, Number(el.value || 1)));
  el.value = String(n);
  try {
    await fetch("/api/settings/parallel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_parallel_jobs: n }),
    });
  } catch (_) {
    /* ignore */
  }
}

$("parallel_jobs")?.addEventListener("change", () => {
  syncParallelJobs().then(() => {
    appendLog(`Parallel jobs set to ${$("parallel_jobs").value}`);
  });
});

$("btn-run").addEventListener("click", async () => {
  const body = formPayload();
  if (!body.prompt) return alert("Enter a story prompt");

  try {
    const hr = await fetch("/api/health");
    const hj = await hr.json();
    if (hj.essentials && !hj.essentials.ready_for_generate) {
      showEssentialsBanner(hj.essentials, { force: true });
      const blockers = (hj.essentials.blocking || []).join(" · ") || "required tools offline";
      appendLog(`Tools not ready (${blockers}) — self-healing (auto-start) then continuing…`);
      await startEssentialServices();
    }
  } catch (_) {
    /* pipeline will self-heal again at job start */
  }

  resultEl.classList.add("hidden");
  appendLog("Enqueueing job (Director → H3 → Critic → Assemble)…");
  try {
    await syncParallelJobs();
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    appendLog(j.message || "Queued");
    watchingId = j.job_ref || null;
    selectedTabKey = j.job_ref || null;
    saveSession({ job_ref: watchingId, watching: true });
    startPolling(watchingId);
  } catch (e) {
    appendLog("Start failed: " + e.message);
  }
});

$("btn-stop").addEventListener("click", async () => {
  appendLog("Requesting stop…");
  $("btn-stop").disabled = true;
  try {
    const body = {};
    if (selectedTabKey) {
      body.job_ref = selectedTabKey;
      // Also target project id if known
      const tab = (lastJobsPayload?.tabs || []).find((t) => t.job_key === selectedTabKey);
      if (tab?.project_id && !String(tab.project_id).startsWith("pending_")) {
        body.project_id = tab.project_id;
      }
    } else {
      body.stop_all = true;
    }
    const r = await fetch("/api/generate/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    appendLog(j.message || "Stop requested");
    if (j.orphaned?.length) {
      appendLog("Cleared stuck project(s): " + j.orphaned.join(", "));
    }
    if (!pollTimer) startPolling(watchingId);
  } catch (e) {
    appendLog("Stop failed: " + e.message);
    $("btn-stop").disabled = false;
  }
});

function resolveJobFromPayload(payload, preferredId) {
  const active = payload.active || {};
  if (preferredId && active[preferredId]) return active[preferredId];
  const tabs = payload.tabs || [];
  if (preferredId) {
    const tab = tabs.find(
      (t) => t.job_key === preferredId || t.project_id === preferredId
    );
    if (tab) {
      const hit =
        active[tab.job_key] ||
        active[tab.project_id] ||
        Object.values(active).find(
          (x) => x.job_key === tab.job_key || x.project_id === tab.project_id
        );
      if (hit) return hit;
    }
  }
  return pickJob(payload, preferredId);
}

function pickJob(payload, preferredId) {
  const active = Object.values(payload.active || {});
  const workerAlive = payload.worker_alive !== false;

  if (workerAlive) {
    const lives = [];
    const seen = new Set();
    const push = (x) => {
      if (!x || !isWorkerLiveStatus(x.status)) return;
      const id = x.job_key || x.project_id || "";
      if (!id || seen.has(id)) return;
      seen.add(id);
      lives.push(x);
    };
    push(payload.current);
    for (const x of active) push(x);

    if (lives.length) {
      if (preferredId) {
        const byProject = lives.find(
          (x) =>
            x.project_id === preferredId ||
            x.job_key === preferredId
        );
        if (byProject) return byProject;
        const byKey = payload.active?.[preferredId];
        if (byKey && isWorkerLiveStatus(byKey.status)) return byKey;
      }
      return lives[0];
    }
  }

  // Queued preferred
  if (preferredId) {
    const hit =
      payload.active?.[preferredId] ||
      active.find(
        (x) => x.project_id === preferredId || x.job_key === preferredId
      );
    if (hit) return hit;
  }
  const queued = active.find((x) => String(x.status).toLowerCase() === "queued");
  if (queued) return queued;
  if (payload.current) return payload.current;
  return null;
}

function tabLabel(tab) {
  const title = (tab.title || tab.prompt_preview || tab.project_id || tab.job_key || "Job")
    .toString()
    .trim();
  const short = title.length > 28 ? title.slice(0, 26) + "…" : title;
  return short;
}

function renderJobTabs(payload) {
  const el = $("job-tabs");
  const meta = $("queue-meta");
  if (!el) return;
  const tabs = Array.isArray(payload.tabs) ? payload.tabs : [];
  const maxP = payload.max_parallel_jobs ?? 1;
  const workers = payload.workers_alive ?? (payload.worker_alive ? 1 : 0);
  const qn = (payload.queue || []).length;

  if (meta) {
    if (!tabs.length) meta.textContent = "No active jobs.";
    else
      meta.textContent = `${workers}/${maxP} running · ${qn} queued · ${tabs.length} tab(s)`;
  }

  if (!tabs.length) {
    el.innerHTML = "";
    return;
  }

  const sel = selectedTabKey || watchingId;
  el.innerHTML = tabs
    .map((t) => {
      const key = t.job_key || t.project_id;
      const st = String(t.status || "").toLowerCase();
      const active = key === sel ? " active" : "";
      const pos =
        st === "queued" && t.queue_position != null ? ` #${t.queue_position}` : "";
      return `<button type="button" class="job-tab ${escapeHtml(st)}${active}" data-job-key="${escapeHtml(
        key
      )}" role="tab" aria-selected="${key === sel ? "true" : "false"}">
        <strong>${escapeHtml(tabLabel(t))}</strong>
        <span class="tab-sub">${escapeHtml(st)}${escapeHtml(pos)}</span>
      </button>`;
    })
    .join("");

  el.querySelectorAll("button[data-job-key]").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedTabKey = btn.dataset.jobKey;
      watchingId = selectedTabKey;
      saveSession({ job_ref: watchingId, watching: true });
      lastUiProjectId = null;
      if (lastJobsPayload) {
        const job = resolveJobFromPayload(lastJobsPayload, selectedTabKey);
        if (job) applyLiveJobUi(job);
      }
      startPolling(selectedTabKey);
    });
  });
}

/** Track which project last painted cast / plan so we can switch when live job changes. */
let lastUiProjectId = null;

function applyLiveJobUi(job) {
  const pid = job.project_id || null;
  const projectChanged =
    pid && !String(pid).startsWith("pending_") && pid !== lastUiProjectId;
  if (projectChanged) {
    lastUiProjectId = pid;
    renderCast([], null);
  }
  setLogLines(job.log || []);
  renderJobStatus(job);
  if (Array.isArray(job.characters)) {
    renderCast(job.characters, pid);
  } else if (projectChanged) {
    renderCast([], pid);
  }
  return projectChanged;
}

function anyActiveWork(payload) {
  if (payload.worker_alive) return true;
  if ((payload.queue || []).length) return true;
  if ((payload.tabs || []).length) return true;
  return Object.values(payload.active || {}).some((j) => isLiveStatus(j.status));
}

function startPolling(preferredId) {
  if (pollTimer) clearInterval(pollTimer);
  watchingId = preferredId || watchingId;
  if (watchingId) selectedTabKey = watchingId;

  const tick = async () => {
    try {
      const r = await fetch("/api/jobs");
      const j = await r.json();
      lastJobsPayload = j;
      renderJobTabs(j);

      // Keep parallel input in sync with server
      if ($("parallel_jobs") && j.max_parallel_jobs != null && document.activeElement !== $("parallel_jobs")) {
        $("parallel_jobs").value = String(j.max_parallel_jobs);
      }

      const job = resolveJobFromPayload(j, selectedTabKey || watchingId);
      const live = job && isLiveStatus(job.status);
      const workerLive =
        job && isWorkerLiveStatus(job.status) && (j.worker_alive || String(job.status).toLowerCase() === "cancelling");

      if (anyActiveWork(j)) {
        setBusy(true, { keepGenerate: true });
      }

      if (live) {
        if (job.project_id && !String(job.project_id).startsWith("pending_")) {
          if (selectedTabKey === (job.job_key || watchingId) || !selectedTabKey) {
            watchingId = job.job_key || job.project_id;
            saveSession({
              job_ref: watchingId,
              watching: true,
              last_project_id: job.project_id,
            });
          }
        }
        const projectChanged = applyLiveJobUi(job);
        if (job.status === "cancelling") {
          $("btn-stop").disabled = true;
        } else {
          $("btn-stop").disabled = false;
        }
        const needPlan =
          workerLive &&
          job.project_id &&
          !String(job.project_id).startsWith("pending_") &&
          (projectChanged ||
            !job.characters?.length ||
            planEl.textContent === "No plan yet." ||
            (job.title && !String(planEl.textContent || "").includes(String(job.title))));
        if (needPlan) {
          try {
            const pr = await fetch(`/api/projects/${encodeURIComponent(job.project_id)}`);
            if (pr.ok) {
              const detail = await pr.json();
              if (detail.plan) {
                planEl.textContent = JSON.stringify(detail.plan, null, 2);
                renderCast(charactersFromPlan(detail.plan), job.project_id);
              }
              if (detail.stage_timings?.length || detail.log?.length) {
                renderJobStatus({
                  ...job,
                  stage_timings: detail.stage_timings || job.stage_timings,
                  job_started_at: detail.job_started_at || job.job_started_at,
                  characters: charactersFromPlan(detail.plan),
                  title: detail.plan?.title || job.title,
                });
              }
              if (detail.log?.length > (job.log || []).length) setLogLines(detail.log);
            }
          } catch (_) {
            /* keep polled log */
          }
        }
        return;
      }

      // Selected job finished, but others may still be active
      if (anyActiveWork(j)) {
        // Switch selection to another live tab if current died
        const next = (j.tabs || [])[0];
        if (next) {
          selectedTabKey = next.job_key || next.project_id;
          watchingId = selectedTabKey;
          const nj = resolveJobFromPayload(j, selectedTabKey);
          if (nj) applyLiveJobUi(nj);
        }
        return;
      }

      // All jobs finished or idle
      clearInterval(pollTimer);
      pollTimer = null;
      setBusy(false);
      saveSession({ watching: false });
      await refreshProjects();

      const projectId =
        (job && job.project_id && !String(job.project_id).startsWith("pending_")
          ? job.project_id
          : null) ||
        (j.projects && j.projects[0] && j.projects[0].project_id) ||
        loadSession()?.last_project_id;

      if (projectId) {
        try {
          await openProject(projectId);
          lastUiProjectId = projectId;
        } catch (_) {
          if (job?.log?.length) setLogLines(job.log);
        }
        if (job?.master_path || (j.projects && j.projects[0]?.master_path)) {
          const mp = job?.master_path || j.projects[0].master_path;
          showMaster(projectId, mp);
        }
      } else if (job?.log?.length) {
        setLogLines(job.log);
      }
      if (job?.status === "cancelled") {
        appendLog("Generation stopped.");
      } else if (job?.status === "failed") {
        appendLog("Job failed (shot exhausted retakes or no masters).");
      } else {
        appendLog("Job finished or idle.");
      }
    } catch (e) {
      appendLog("Poll error: " + e.message);
    }
  };

  tick();
  pollTimer = setInterval(tick, 1500);
}

/** On tab open: restore last logs and resume follow if generation still running. */
async function resumeSession() {
  try {
    const r = await fetch("/api/jobs");
    const j = await r.json();
    lastJobsPayload = j;
    renderJobTabs(j);
    if ($("parallel_jobs") && j.max_parallel_jobs != null) {
      $("parallel_jobs").value = String(j.max_parallel_jobs);
    }
    const session = loadSession();
    const preferred = session?.job_ref || session?.last_project_id || null;
    const job = resolveJobFromPayload(j, preferred) || pickJob(j, preferred);

    if (anyActiveWork(j)) {
      setBusy(true, { keepGenerate: true });
      watchingId =
        job?.job_key || job?.project_id || preferred || (j.tabs?.[0]?.job_key);
      selectedTabKey = watchingId;
      lastUiProjectId = null;
      if (job) applyLiveJobUi(job);
      appendLog("Reconnected — following job queue.");
      saveSession({ job_ref: watchingId, watching: true, last_project_id: watchingId });
      if (watchingId && !String(watchingId).startsWith("pending_")) {
        try {
          const pr = await fetch(`/api/projects/${encodeURIComponent(job?.project_id || watchingId)}`);
          if (pr.ok) {
            const detail = await pr.json();
            if (detail.plan) {
              planEl.textContent = JSON.stringify(detail.plan, null, 2);
              renderCast(charactersFromPlan(detail.plan), job?.project_id || watchingId);
            }
            if (detail.log?.length) setLogLines(detail.log);
          }
        } catch (_) {
          /* ignore */
        }
      }
      startPolling(watchingId);
      await refreshProjects();
      return;
    }

    const projectId =
      (job && job.project_id && !String(job.project_id).startsWith("pending_")
        ? job.project_id
        : null) ||
      preferred ||
      (j.projects && j.projects[0] && j.projects[0].project_id);

    if (projectId && !String(projectId).startsWith("pending_")) {
      try {
        await openProject(projectId);
        lastUiProjectId = projectId;
        appendLog("Restored logs for " + projectId);
      } catch (_) {
        if (job?.log?.length) {
          setLogLines(job.log);
          appendLog("Restored in-memory job log.");
        }
      }
    } else if (job?.log?.length) {
      setLogLines(job.log);
    }
  } catch (e) {
    appendLog("Could not restore session: " + e.message);
  }
}

$("btn-refresh").addEventListener("click", refreshProjects);
$("btn-copy-log")?.addEventListener("click", async () => {
  const text = (logEl?.textContent || "").trim();
  const btn = $("btn-copy-log");
  if (!text) {
    appendLog("Nothing to copy yet.");
    return;
  }
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      // Fallback for older environments
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    if (btn) {
      const prev = btn.textContent;
      btn.textContent = "Copied";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = prev || "Copy log";
        btn.classList.remove("copied");
      }, 1500);
    }
  } catch (e) {
    appendLog("Copy failed: " + (e.message || e));
  }
});
$("btn-essentials-dismiss")?.addEventListener("click", () => {
  const banner = $("essentials-banner");
  const body = $("essentials-body");
  essentialsDismissedKey = body?.textContent || "dismissed";
  banner?.classList.add("hidden");
});
$("btn-start-services")?.addEventListener("click", () => startEssentialServices());
startHeartbeat();
refreshProjects();
resumeSession();
