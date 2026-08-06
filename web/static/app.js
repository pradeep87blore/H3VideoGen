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

function setBusy(busy) {
  $("btn-run").disabled = busy;
  $("btn-plan").disabled = busy;
  const stop = $("btn-stop");
  if (stop) stop.disabled = !busy;
}

function isLiveStatus(status) {
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
      const hasImg = !!(c.image_path && projectId && !String(projectId).startsWith("pending_"));
      const initial = escapeHtml((name || "?").slice(0, 1).toUpperCase());
      let avatar = `<div class="cast-avatar" aria-hidden="true">${initial}</div>`;
      if (hasImg) {
        const base = String(c.image_path).replace(/\\/g, "/").split("/").pop();
        if (base) {
          const src = `/api/projects/${encodeURIComponent(projectId)}/file?path=${encodeURIComponent(
            "character_board/" + base
          )}`;
          avatar = `<div class="cast-avatar"><img src="${src}" alt="" loading="lazy" /></div>`;
        }
      }
      return `<article class="cast-card">
        ${avatar}
        <div class="cast-meta">
          <div class="cast-id">${escapeHtml(id)}</div>
          <p class="cast-name">${escapeHtml(name)}</p>
          ${look ? `<p class="cast-look">${escapeHtml(look)}</p>` : ""}
          ${
            sheets != null
              ? `<p class="cast-sheet">${sheets} sheet view${sheets === 1 ? "" : "s"}</p>`
              : ""
          }
        </div>
      </article>`;
    })
    .join("");
}

function charactersFromPlan(plan) {
  if (!plan || !Array.isArray(plan.characters)) return [];
  return plan.characters.map((c) => ({
    id: c.id,
    name: c.name,
    look: c.look || "",
    image_path: c.image_path,
    sheet_count: Array.isArray(c.sheet)
      ? c.sheet.filter((p) => p && p.image_path).length
      : undefined,
  }));
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
    ? `Prerequisites missing — will fail within ~${waitMin} min if not fixed`
    : "Prerequisites: soft warnings";
  body.textContent =
    ess.prompt ||
    [...blocking.map((b) => "• " + b), ...warnings.map((w) => "• " + w)].join("\n");
  banner.classList.toggle("warn", !blocking.length);
  banner.classList.remove("hidden");

  if (blocking.length && sig !== essentialsAlertedKey) {
    essentialsAlertedKey = sig;
    const msg =
      "H3 Video Gen needs tools that are not ready:\n\n" +
      blocking.map((b) => "• " + b).join("\n") +
      `\n\nAuto-start waits up to ~${waitMin} minutes, then stops with an error in the log.\n` +
      "Use “Start services” on the banner, or start ComfyUI / Ollama manually.";
    try {
      window.alert(msg);
    } catch (_) {
      /* ignore if blocked */
    }
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
  setBusy(true);
  resultEl.classList.add("hidden");
  appendLog(`Resuming project ${projectId} (auto-starts ComfyUI/Ollama if needed)…`);
  try {
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
    appendLog(j.message || "Resume started");
    watchingId = j.job_ref || projectId;
    saveSession({ job_ref: watchingId, watching: true, last_project_id: projectId });
    startPolling(watchingId);
  } catch (e) {
    appendLog("Resume failed: " + e.message);
    setBusy(false);
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
  planEl.textContent = JSON.stringify(j.plan || j, null, 2);
  renderCast(charactersFromPlan(j.plan) || j.characters || [], id);
  setLogLines(j.log || []);
  if (!j.log || !j.log.length) {
    logEl.textContent = "No logs stored for this project.";
  }
  if (j.master_path) {
    showMaster(id, j.master_path);
  } else {
    resultEl.classList.add("hidden");
  }
  saveSession({ last_project_id: id, job_ref: id });
  return j;
}

function formPayload() {
  return {
    prompt: $("prompt").value.trim(),
    style: $("style").value.trim(),
    target_duration_sec: Number($("duration").value || 60),
    max_shots: Number($("max_shots").value || 12),
    max_retakes: Number($("max_retakes").value || 2),
    auto_assemble: true,
    seed_base: 42,
  };
}

$("btn-plan").addEventListener("click", async () => {
  const body = formPayload();
  if (!body.prompt) return alert("Enter a story prompt");
  setBusy(true);
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
      }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    planEl.textContent = JSON.stringify(j, null, 2);
    renderCast(charactersFromPlan(j), null);
    appendLog(`Plan: “${j.title}” — ${j.shots?.length || 0} shots, ~${j.target_duration_sec}s`);
    if (j.characters?.length) {
      appendLog(`Cast: ${j.characters.map((c) => c.name || c.id).join(", ")}`);
    }
  } catch (e) {
    appendLog("Plan failed: " + e.message);
  } finally {
    setBusy(false);
  }
});

let pollTimer = null;
let watchingId = null;

$("btn-run").addEventListener("click", async () => {
  const body = formPayload();
  if (!body.prompt) return alert("Enter a story prompt");

  // Preflight: surface essentials before long GPU wait
  try {
    const hr = await fetch("/api/health");
    const hj = await hr.json();
    if (hj.essentials && !hj.essentials.ready_for_generate) {
      showEssentialsBanner(hj.essentials, { force: true });
      const waitMin = Math.max(1, Math.round((hj.essentials.wait_limit_sec || 300) / 60));
      const okContinue = window.confirm(
        "Required tools are not ready:\n\n" +
          (hj.essentials.blocking || []).map((b) => "• " + b).join("\n") +
          `\n\nGenerate will try auto-start and stop within ~${waitMin} minutes if they stay down.\n\nContinue?`
      );
      if (!okContinue) return;
    }
  } catch (_) {
    /* proceed; pipeline will fail clearly */
  }

  setBusy(true);
  resultEl.classList.add("hidden");
  logEl.textContent = "";
  renderCast([], null);
  appendLog("Starting full pipeline (Director → H3 → Critic → Assemble)…");
  appendLog("Will auto-start ComfyUI / Ollama if they are not running.");
  try {
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    appendLog(j.message || "Started");
    watchingId = j.job_ref || null;
    saveSession({ job_ref: watchingId, watching: true });
    startPolling(watchingId);
  } catch (e) {
    appendLog("Start failed: " + e.message);
    setBusy(false);
    clearSession();
  }
});

$("btn-stop").addEventListener("click", async () => {
  appendLog("Requesting stop…");
  $("btn-stop").disabled = true;
  try {
    const r = await fetch("/api/generate/stop", { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    appendLog(j.message || "Stop requested");
    if (j.orphaned?.length) {
      appendLog("Cleared stuck project(s): " + j.orphaned.join(", "));
    }
    // Always re-poll so UI unlocks when worker is already gone / orphans cleared
    if (!pollTimer) startPolling(watchingId);
    // If nothing was live, unlock UI immediately
    if (!j.stopped || j.orphaned?.length) {
      // Give one fast poll cycle; tick will clear busy if not live
    }
  } catch (e) {
    appendLog("Stop failed: " + e.message);
    $("btn-stop").disabled = false;
  }
});

function pickJob(payload, preferredId) {
  const active = Object.values(payload.active || {});
  if (preferredId) {
    const hit =
      payload.active?.[preferredId] ||
      active.find((x) => x.project_id === preferredId);
    if (hit) return hit;
  }
  // Prefer truly live jobs only when worker says so
  if (payload.worker_alive !== false) {
    const live = active.find((x) => isLiveStatus(x.status));
    if (live) return live;
  } else {
    // Worker down: never treat cancelled/orphaned as live-polling forever
    const live = active.find(
      (x) => isLiveStatus(x.status) && String(x.status).toLowerCase() === "cancelling"
    );
    if (live && payload.cancel_requested) return live;
  }
  if (payload.current && isLiveStatus(payload.current.status) && payload.worker_alive) {
    return payload.current;
  }
  if (payload.current) return payload.current;
  return null;
}

function startPolling(preferredId) {
  if (pollTimer) clearInterval(pollTimer);
  watchingId = preferredId || watchingId;

  const tick = async () => {
    try {
      const r = await fetch("/api/jobs");
      const j = await r.json();
      const job = pickJob(j, watchingId);
      const live =
        job &&
        isLiveStatus(job.status) &&
        (j.worker_alive || String(job.status).toLowerCase() === "cancelling");

      if (live) {
        setBusy(true);
        if (job.project_id && !String(job.project_id).startsWith("pending_")) {
          watchingId = job.project_id;
          saveSession({ job_ref: watchingId, watching: true, last_project_id: watchingId });
        }
        setLogLines(job.log || []);
        if (job.characters?.length) {
          renderCast(job.characters, job.project_id);
        }
        if (job.status === "cancelling") {
          $("btn-stop").disabled = true;
        }
        if (job.title && planEl.textContent === "No plan yet.") {
          try {
            const pr = await fetch(`/api/projects/${encodeURIComponent(job.project_id)}`);
            if (pr.ok) {
              const detail = await pr.json();
              if (detail.plan) {
                planEl.textContent = JSON.stringify(detail.plan, null, 2);
                renderCast(charactersFromPlan(detail.plan), job.project_id);
              }
              if (detail.log?.length > (job.log || []).length) setLogLines(detail.log);
            }
          } catch (_) {
            /* keep polled log */
          }
        } else if (
          job.project_id &&
          !String(job.project_id).startsWith("pending_") &&
          (!job.characters || !job.characters.length)
        ) {
          // Plan may exist on disk before job snapshot includes cast
          try {
            const pr = await fetch(`/api/projects/${encodeURIComponent(job.project_id)}`);
            if (pr.ok) {
              const detail = await pr.json();
              if (detail.plan?.characters?.length) {
                if (planEl.textContent === "No plan yet.") {
                  planEl.textContent = JSON.stringify(detail.plan, null, 2);
                }
                renderCast(charactersFromPlan(detail.plan), job.project_id);
              }
            }
          } catch (_) {
            /* ignore */
          }
        }
        return;
      }

      // Job finished or idle
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
    const session = loadSession();
    const preferred = session?.job_ref || session?.last_project_id || null;
    const job = pickJob(j, preferred);

    if (job && isLiveStatus(job.status) && j.worker_alive) {
      setBusy(true);
      setLogLines(job.log || ["Resuming live generation…"]);
      appendLog("Reconnected — following live generation logs.");
      watchingId = job.project_id || preferred;
      saveSession({ job_ref: watchingId, watching: true, last_project_id: watchingId });
      startPolling(watchingId);
      await refreshProjects();
      return;
    }

    // Restore finished / last project logs
    const projectId =
      (job && job.project_id && !String(job.project_id).startsWith("pending_")
        ? job.project_id
        : null) ||
      preferred ||
      (j.projects && j.projects[0] && j.projects[0].project_id);

    if (projectId && !String(projectId).startsWith("pending_")) {
      try {
        await openProject(projectId);
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
