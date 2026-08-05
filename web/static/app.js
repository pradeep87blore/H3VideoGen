const $ = (id) => document.getElementById(id);
const logEl = $("log");
const planEl = $("plan");
const resultEl = $("result");
const projectsEl = $("projects");
const healthEl = $("health");

const STORAGE_KEY = "h3vg_job";

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

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    const bits = [];
    bits.push(j.gemini_key_set ? "Gemini key ✓" : "Gemini key missing");
    bits.push(j.comfy_ok ? "ComfyUI ✓" : "ComfyUI ✗");
    bits.push(j.ollama_ok ? "Ollama ✓" : "Ollama ·");
    bits.push(j.voice_enabled ? "Voice on" : "Voice off");
    healthEl.textContent = bits.join(" · ");
    healthEl.className = "health " + (j.comfy_ok ? "ok" : "bad");
  } catch (e) {
    healthEl.textContent = "API unreachable";
    healthEl.className = "health bad";
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
    appendLog(`Plan: “${j.title}” — ${j.shots?.length || 0} shots, ~${j.target_duration_sec}s`);
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
  setBusy(true);
  resultEl.classList.add("hidden");
  logEl.textContent = "";
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
        if (job.status === "cancelling") {
          $("btn-stop").disabled = true;
        }
        if (job.title && planEl.textContent === "No plan yet.") {
          try {
            const pr = await fetch(`/api/projects/${encodeURIComponent(job.project_id)}`);
            if (pr.ok) {
              const detail = await pr.json();
              if (detail.plan) planEl.textContent = JSON.stringify(detail.plan, null, 2);
              if (detail.log?.length > (job.log || []).length) setLogLines(detail.log);
            }
          } catch (_) {
            /* keep polled log */
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
refreshHealth();
refreshProjects();
resumeSession();
setInterval(refreshHealth, 15000);
