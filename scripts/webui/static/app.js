/* WARP-RM inspector client. Single-file vanilla JS + Plotly. */
(() => {
  const els = {
    ckptSelect:     document.getElementById("ckpt-select"),
    status:         document.getElementById("status"),
    datasetToggle:  document.getElementById("dataset-toggle"),
    datasetSummary: document.getElementById("dataset-summary"),
    datasetPopover: document.getElementById("dataset-popover"),
    datasetList:    document.getElementById("dataset-list"),
    datasetSearch:  document.getElementById("dataset-search"),
    datasetClear:   document.getElementById("dataset-clear"),
    epList:         document.getElementById("episode-list"),
    epFooter:       document.getElementById("episode-footer"),
    epSearch:       document.getElementById("episode-search"),
    epSort:         document.getElementById("episode-sort"),
    video:          document.getElementById("video"),
    videoTitle:     document.getElementById("video-title"),
    videoMeta:      document.getElementById("video-meta"),
    frameLabel:     document.getElementById("frame-label"),
    timeLabel:      document.getElementById("time-label"),
    weightLabel:    document.getElementById("weight-label"),
    stats:          document.getElementById("stats"),
    plotProgress:   document.getElementById("plot-progress"),
    plotVelocity:   document.getElementById("plot-velocity"),
    plotZoom:       document.getElementById("plot-zoom"),
    cursorProgress: document.getElementById("cursor-progress"),
    cursorVelocity: document.getElementById("cursor-velocity"),
    cursorZoom:     document.getElementById("cursor-zoom"),
    rabcField:      document.querySelector(".rabc-field"),
    rabcMode:       document.getElementById("rabc-mode"),
    rabcClipMin:    document.getElementById("rabc-clip-min"),
    rabcClipMax:    document.getElementById("rabc-clip-max"),
    rabcThresh:     document.getElementById("rabc-thresh"),
    signalSource:   document.getElementById("signal-source"),
    cameraSelect:   document.getElementById("camera-select"),
  };

  const state = {
    checkpoints: [],
    currentCkpt: null,
    datasets: [],
    activeRepos: new Set(),
    episodes: [],
    selectedEp: null,
    inference: null,
    fps: 30,
    tMark: 0,
    rabc: { mode: "off", clipMin: 0.0, clipMax: 1.0, thresh: null },
    // Per-episode velocity summaries.
    // Map<"<ckpt>::<repo>::<ep_idx>", number|null>: null = in-flight;
    // undefined = not started; number = mean_velocity.
    summary: new Map(),
    summaryJob: null, // current scoring generation token
    // Video feature key to score. Datasets do not agree on a naming
    // convention (`top_camera-images-rgb` vs `observation.images.top`), so this
    // is chosen from the selected datasets' declared video_keys rather than
    // assumed — an unavailable key makes /api/episodes 500 with an empty list.
    cameraKey: null,
  };
  const SUMMARY_CONCURRENCY = 4;

  // Per-frame weight under the selected WARP-BC scheme, applied directly to
  // velocity (no chunk integration) so the plots show what the downstream
  // weight would look like if that frame's value were the sample weight.
  //
  // Returns { vs, ws } — parallel arrays. `vs` is what the velocity plot draws,
  // `ws` is the per-frame weight that drives the color shading.
  function applyScheme(vel) {
    const r = state.rabc;
    const n = vel.length;
    const cMin = isFinite(r.clipMin) ? r.clipMin : 0;
    const cMax = isFinite(r.clipMax) ? r.clipMax : 1;

    if (r.mode === "off") {
      const ws = new Float32Array(n);
      for (let i = 0; i < n; i++) ws[i] = Math.max(0, vel[i]);
      return { vs: vel, ws };
    }

    if (r.mode === "velocity_only") {
      const hasThresh = r.thresh !== null && isFinite(r.thresh);
      const vs = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const v = vel[i];
        if (hasThresh) vs[i] = v < r.thresh ? 0 : Math.min(v, cMax);
        else           vs[i] = Math.min(Math.max(v, cMin), cMax);
      }
      return { vs, ws: vs };
    }

    // Unknown mode: fall back to off.
    const ws = new Float32Array(n);
    for (let i = 0; i < n; i++) ws[i] = Math.max(0, vel[i]);
    return { vs: vel, ws };
  }

  // ── Palette (mirrors warp_rm/visualization/plotting.py) ─────────────
  const FG      = "#EAEAEA";
  const FG_DIM  = "#63637A";
  const BG_PLOT = "#0D0D1A";
  const GRID    = "#222244";
  const SPINE   = "#444466";
  const CUR     = "#F4D35E";
  const PROG    = "#9B59B6"; // C_RECON
  const VEL_FG  = "rgba(234,234,234,0.75)";
  const DISPLAY = "Fraunces, serif";
  const MONO    = "'IBM Plex Mono', ui-monospace, SF Mono, Menlo, monospace";

  // Normalized ramp t∈[0, 1]: red → yellow → green. Below/above clamp.
  function rampTToRgb(t) {
    t = Math.min(1, Math.max(0, t));
    if (t <= 0) return "rgb(230,50,50)";
    if (t < 0.5) {
      const g = Math.round(255 * (t * 1.6));
      return `rgb(255,${g},25)`;
    }
    const s = (t - 0.5) * 2;
    const r = Math.round(255 * (1 - s * 0.82));
    const g = Math.round(255 * (0.8 + s * 0.1));
    const b = Math.round(255 * (0.1 + s * 0.2));
    return `rgb(${r},${g},${b})`;
  }
  // Weight → RGB, remapped to the RABC clip band. `clipMin` anchors red,
  // `clipMax` anchors green; anything outside the band saturates. When
  // RABC is off, the inputs still default to (0, 1) so the gradient reads
  // as red-at-0 / green-at-1 — which is what most callers want.
  function weightToRgb(w) {
    const lo = state.rabc.clipMin;
    const hi = state.rabc.clipMax;
    const span = hi - lo;
    if (!isFinite(span) || span <= 0) return rampTToRgb(w >= hi ? 1 : 0);
    return rampTToRgb((w - lo) / span);
  }
  function weightToRgba(w, a) {
    const c = weightToRgb(w).match(/\d+/g).map(Number);
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }

  // Static Plotly colorscale over normalized t∈[0, 1]. The heatmap's
  // zmin/zmax (see heatmapBgShaped) drive the actual w→t mapping — which
  // keeps the gradient in lock-step with the RABC inputs without having
  // to rebuild the colorscale on every change.
  const WEIGHT_COLORSCALE = (() => {
    const stops = [];
    for (let i = 0; i <= 10; i++) {
      stops.push([i / 10, rampTToRgb(i / 10)]);
    }
    return stops;
  })();

  function baseLayout(title, titleColor) {
    return {
      title: {
        text: title,
        font: { size: 13, color: titleColor || FG, family: DISPLAY, weight: 500 },
        x: 0.01, xanchor: "left", y: 0.98, yanchor: "top",
      },
      paper_bgcolor: BG_PLOT,
      plot_bgcolor:  BG_PLOT,
      margin: { l: 48, r: 14, t: 26, b: 30 },
      font: { color: FG, family: MONO, size: 10 },
      xaxis: {
        color: "#A8A8BC", gridcolor: GRID, zerolinecolor: GRID,
        linecolor: SPINE, linewidth: 1, mirror: true,
        ticks: "outside", tickcolor: "#8A8A9E", ticklen: 4, tickwidth: 1,
        showgrid: true, gridwidth: 0.5, range: [0, 1],
        tickfont: { color: "#C5C5D4", size: 10, family: MONO },
      },
      yaxis: {
        color: "#A8A8BC", gridcolor: GRID, zerolinecolor: GRID,
        linecolor: SPINE, linewidth: 1, mirror: true,
        ticks: "outside", tickcolor: "#8A8A9E", ticklen: 4, tickwidth: 1,
        showgrid: true, gridwidth: 0.5,
        tickfont: { color: "#C5C5D4", size: 10, family: MONO },
      },
      showlegend: false,
      hovermode: "x unified",
    };
  }

  const PLOT_CFG = { responsive: true, displayModeBar: false, doubleClick: "reset" };

  // ── API helpers ────────────────────────────────────────────────────
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  }
  function setStatus(text, kind = "") {
    els.status.textContent = text;
    els.status.className = "status" + (kind ? " " + kind : "");
  }
  function msTime(seconds) {
    if (!isFinite(seconds)) return "-:--";
    const m = Math.floor(seconds / 60);
    const s = seconds - 60 * m;
    return `${m}:${s.toFixed(2).padStart(5, "0")}`;
  }

  // ── Checkpoints ────────────────────────────────────────────────────
  async function loadCheckpoints() {
    const r = await api("/api/checkpoints");
    state.checkpoints = r.checkpoints;
    state.currentCkpt = r.current;
    els.ckptSelect.innerHTML = "";
    if (!r.checkpoints.length) {
      els.ckptSelect.innerHTML = `<option value="">no checkpoints found</option>`;
      els.ckptSelect.disabled = true;
      return;
    }
    els.ckptSelect.disabled = false;
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = r.current ? "— switch —" : "— choose —";
    els.ckptSelect.appendChild(blank);
    for (const c of r.checkpoints) {
      const o = document.createElement("option");
      o.value = c.path;
      o.textContent = c.name;
      if (c.path === r.current) o.selected = true;
      els.ckptSelect.appendChild(o);
    }
  }

  // ── Datasets popover ───────────────────────────────────────────────
  async function loadDatasets() {
    const r = await api("/api/datasets");
    state.datasets = r.datasets;
    renderDatasetList();
    updateDatasetSummary();
  }
  function renderDatasetList() {
    const q = (els.datasetSearch.value || "").trim().toLowerCase();
    const rows = state.datasets.filter(d => !q || d.name.toLowerCase().includes(q));
    els.datasetList.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (const d of rows) {
      const on = state.activeRepos.has(d.path);
      const row = document.createElement("div");
      row.className = "popover-row" + (on ? " on" : "");
      const splitNames = d.splits ? Object.keys(d.splits) : [];
      const splitTag = splitNames.length > 1 ? ` <span class="tag-splits">${splitNames.join("/")}</span>` : "";
      row.innerHTML = `
        <span class="mark">${on ? "●" : "○"}</span>
        <span class="label">${d.name}${splitTag}</span>
        <span class="count">${d.n_episodes}</span>
      `;
      row.onclick = () => {
        if (state.activeRepos.has(d.path)) state.activeRepos.delete(d.path);
        else state.activeRepos.add(d.path);
        renderDatasetList();
        updateDatasetSummary();
        rebuildCameraOptions();
        refreshEpisodes();
      };
      frag.appendChild(row);
    }
    els.datasetList.appendChild(frag);
  }
  // Video keys common to EVERY selected dataset. Intersection, not union: a key
  // missing from one selected repo would make that repo's episode query fail.
  function commonCameraKeys() {
    const sel = state.datasets.filter(d => state.activeRepos.has(d.path));
    if (!sel.length) return [];
    return sel.reduce(
      (acc, d) => acc.filter(k => (d.video_keys || []).includes(k)),
      [...(sel[0].video_keys || [])],
    );
  }
  function rebuildCameraOptions() {
    const keys = commonCameraKeys();
    const prev = state.cameraKey;
    els.cameraSelect.innerHTML = "";
    if (!keys.length) {
      els.cameraSelect.innerHTML = `<option value="">—</option>`;
      els.cameraSelect.disabled = true;
      state.cameraKey = null;
      if (state.activeRepos.size > 1) {
        setStatus("selected datasets share no camera key", "err");
      }
      return;
    }
    els.cameraSelect.disabled = false;
    // Keep the current choice when still valid; otherwise prefer the historical
    // default, then any "top" view, then whatever is first.
    let pick = keys.includes(prev) ? prev
      : (keys.includes("top_camera-images-rgb") ? "top_camera-images-rgb"
      : (keys.find(k => k.toLowerCase().includes("top")) || keys[0]));
    for (const k of keys) {
      const o = document.createElement("option");
      o.value = k; o.textContent = k;
      if (k === pick) o.selected = true;
      els.cameraSelect.appendChild(o);
    }
    state.cameraKey = pick;
  }
  function camQ() {
    return state.cameraKey ? `&camera_key=${encodeURIComponent(state.cameraKey)}` : "";
  }

  function updateDatasetSummary() {
    const n = state.activeRepos.size;
    if (n === 0) {
      els.datasetSummary.textContent = "none";
    } else if (n === 1) {
      const path = [...state.activeRepos][0];
      const d = state.datasets.find(x => x.path === path);
      els.datasetSummary.textContent = d ? d.name : "1 selected";
    } else {
      els.datasetSummary.textContent = `${n} selected`;
    }
  }
  function togglePopover(force) {
    const shown = !els.datasetPopover.hasAttribute("hidden");
    const next = force === undefined ? !shown : force;
    if (next) {
      els.datasetPopover.removeAttribute("hidden");
      els.datasetToggle.setAttribute("aria-expanded", "true");
      els.datasetSearch.focus();
    } else {
      els.datasetPopover.setAttribute("hidden", "");
      els.datasetToggle.setAttribute("aria-expanded", "false");
    }
  }
  els.datasetToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    togglePopover();
  });
  els.datasetSearch.addEventListener("input", renderDatasetList);
  els.datasetClear.addEventListener("click", () => {
    state.activeRepos.clear();
    renderDatasetList();
    updateDatasetSummary();
    rebuildCameraOptions();
    refreshEpisodes();
  });
  document.addEventListener("click", (e) => {
    if (!els.datasetPopover.contains(e.target) && e.target !== els.datasetToggle) {
      togglePopover(false);
    }
  });

  // ── Episode list ────────────────────────────────────────────────────
  async function refreshEpisodes() {
    if (state.activeRepos.size === 0) {
      state.episodes = [];
      renderEpisodeList();
      return;
    }
    setStatus("discovering", "busy");
    const all = [];
    const errs = [];
    for (const repo of state.activeRepos) {
      try {
        const r = await api(`/api/episodes?repo=${encodeURIComponent(repo)}${camQ()}`);
        const meta = state.datasets.find(d => d.path === repo) || {};
        for (const ep of r.episodes) {
          all.push({ repo, repo_name: meta.name || repo, fps: meta.fps || 30, ...ep });
        }
      } catch (e) {
        console.error(e);
        // Previously this failed silently and the user saw an empty table with
        // no explanation (the usual cause is a camera key the dataset lacks).
        errs.push(String(e.message || e).slice(0, 200));
      }
    }
    state.episodes = all;
    if (!all.length && errs.length) setStatus(errs[0], "err");
    else setStatus("idle");
    renderEpisodeList();
    if (sortNeedsSummary(els.epSort.value)) ensureSummaries();
  }

  function sortNeedsSummary(sort) {
    return sort === "velocity_asc" || sort === "velocity_desc";
  }
  function summaryKey(ckpt, repo, epIdx) {
    return `${ckpt || ""}::${repo}::${epIdx}`;
  }
  function lookupSummary(ep) {
    return state.summary.get(summaryKey(state.currentCkpt, ep.repo, ep.idx));
  }

  // Coalesce high-frequency renderEpisodeList() calls (e.g. one per quality
  // response during bulk scoring) into at-most-one render per animation frame.
  // Use this instead of calling renderEpisodeList() directly when mutating
  // state in a tight loop.
  let _renderPending = false;
  function scheduleRenderEpisodeList() {
    if (_renderPending) return;
    _renderPending = true;
    requestAnimationFrame(() => {
      _renderPending = false;
      renderEpisodeList();
    });
  }

  function renderEpisodeList() {
    const q = (els.epSearch.value || "").trim();
    let rows = state.episodes.slice();
    if (q) rows = rows.filter(r => String(r.idx).includes(q) || r.name.includes(q));
    const sort = els.epSort.value;
    const sortByV = sortNeedsSummary(sort);
    rows.sort((a, b) => {
      if (sort === "length_asc")  return a.n_frames - b.n_frames;
      if (sort === "length_desc") return b.n_frames - a.n_frames;
      if (sortByV) {
        const qa = lookupSummary(a);
        const qb = lookupSummary(b);
        const na = typeof qa === "number" ? qa : null;
        const nb = typeof qb === "number" ? qb : null;
        if (na === null && nb === null) return a.idx - b.idx;
        if (na === null) return 1;   // un-scored sort to the bottom
        if (nb === null) return -1;
        return sort === "velocity_asc" ? na - nb : nb - na;
      }
      return a.idx - b.idx;
    });
    els.epList.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (const r of rows) {
      const div = document.createElement("div");
      div.className = "episode-row";
      if (state.selectedEp && state.selectedEp.repo === r.repo && state.selectedEp.idx === r.idx) {
        div.classList.add("active");
      }
      const mmss = msTime(r.n_frames / r.fps);
      const short = r.repo_name.split("/").pop().slice(0, 6);
      const qv = lookupSummary(r);
      let qCell = "";
      if (sortByV) {
        if (typeof qv === "number") qCell = `<span class="quality">v̄ ${qv.toFixed(2)}</span>`;
        else if (qv === null)       qCell = `<span class="quality pending">…</span>`;
        else                        qCell = `<span class="quality pending">—</span>`;
      }
      div.innerHTML = `
        <span class="name"><span class="src">${short}</span>${r.name.replace("episode_", "ep ")}</span>
        ${qCell}
        <span class="length">${mmss}</span>
      `;
      div.title = `${r.repo_name}\n${r.name} · ${r.n_frames} frames`
        + (typeof qv === "number" ? `\nmean velocity=${qv.toFixed(3)}` : "");
      div.onclick = () => selectEpisode(r);
      frag.appendChild(div);
    }
    els.epList.appendChild(frag);
    const scored = state.episodes.reduce((acc, r) =>
      acc + (typeof lookupSummary(r) === "number" ? 1 : 0), 0);
    const suffix = sortByV ? ` · v̄ ${scored}/${state.episodes.length}` : "";
    els.epFooter.textContent = `${rows.length} / ${state.episodes.length}${suffix}`;
  }

  // ── Velocity summaries (used when sort is by mean velocity) ─────────
  // Mirrors scripts/eval/score_episodes.py: mean per-frame progress velocity
  // per episode. Requires a checkpoint — the summary is derived from dense
  // inference, not from the injected parquet columns.
  async function ensureSummaries() {
    if (!state.currentCkpt) {
      setStatus("load checkpoint to score", "err");
      return;
    }
    const job = Symbol("sjob");
    state.summaryJob = job;

    // Group pending episodes by repo so we can issue one batch cache-pull
    // per dataset before falling back to per-episode fetches for misses.
    const pendingByRepo = new Map();
    const todo = [];
    for (const ep of state.episodes) {
      const key = summaryKey(state.currentCkpt, ep.repo, ep.idx);
      const cur = state.summary.get(key);
      if (typeof cur === "number" || cur === null) continue;
      state.summary.set(key, null); // in-flight
      todo.push(ep);
      if (!pendingByRepo.has(ep.repo)) pendingByRepo.set(ep.repo, []);
      pendingByRepo.get(ep.repo).push(ep);
    }
    if (todo.length === 0) { renderEpisodeList(); return; }

    setStatus(`scoring 0/${todo.length}`, "busy");
    renderEpisodeList();
    let done = 0;

    // Fast path: pull every already-cached summary for each repo in one
    // request. Anything missing still goes through the per-episode fallback.
    for (const [repo, eps] of pendingByRepo) {
      if (state.summaryJob !== job) return;
      try {
        const r = await api(`/api/summary/all?repo=${encodeURIComponent(repo)}${camQ()}`);
        if (state.summaryJob !== job) return;
        const cachedScores = r.scores || {};
        for (const ep of eps) {
          const sc = cachedScores[ep.idx];
          if (sc && typeof sc.mean_velocity === "number") {
            const key = summaryKey(state.currentCkpt, ep.repo, ep.idx);
            state.summary.set(key, sc.mean_velocity);
            done += 1;
            // Remove from todo so the per-episode worker loop doesn't re-fetch.
            const i = todo.indexOf(ep);
            if (i >= 0) todo.splice(i, 1);
          }
        }
      } catch (e) {
        console.error("summary/all", repo, e);
      }
    }
    setStatus(`scoring ${done}/${done + todo.length}`, "busy");
    scheduleRenderEpisodeList();
    if (todo.length === 0) {
      if (state.summaryJob === job) {
        setStatus(`scored ${done}`);
        renderEpisodeList();
      }
      return;
    }

    async function worker() {
      while (state.summaryJob === job) {
        const ep = todo.shift();
        if (!ep) return;
        const key = summaryKey(state.currentCkpt, ep.repo, ep.idx);
        try {
          const r = await api(
            `/api/summary?repo=${encodeURIComponent(ep.repo)}&ep_idx=${ep.idx}${camQ()}`);
          if (state.summaryJob !== job) return;
          state.summary.set(key, typeof r.mean_velocity === "number" ? r.mean_velocity : NaN);
        } catch (e) {
          console.error("summary", ep.name, e);
          state.summary.set(key, NaN);
        }
        done += 1;
        if (state.summaryJob === job) {
          setStatus(`scoring ${done}/${done + todo.length}`, "busy");
          scheduleRenderEpisodeList();
        }
      }
    }
    await Promise.all(Array.from({ length: SUMMARY_CONCURRENCY }, worker));
    if (state.summaryJob === job) {
      setStatus(`scored ${done}`);
      renderEpisodeList();
    }
  }

  // ── Episode selection ──────────────────────────────────────────────
  async function selectEpisode(ep) {
    state.selectedEp = ep;
    state.fps = ep.fps || 30;
    renderEpisodeList();
    els.videoTitle.textContent = `${ep.repo_name} · ${ep.name.replace("episode_", "ep ")}`;
    els.videoMeta.textContent =
      `${ep.n_frames} frames · ${msTime(ep.n_frames / state.fps)} · ${state.fps} fps`;
    els.video.src = `/api/video?repo=${encodeURIComponent(ep.repo)}&ep_idx=${ep.idx}${camQ()}`;
    els.video.load();

    const source = els.signalSource ? els.signalSource.value : "ckpt";
    const url = source === "sidecar"
      ? `/api/dataset_signals?repo=${encodeURIComponent(ep.repo)}&ep_idx=${ep.idx}${camQ()}`
      : `/api/inference?repo=${encodeURIComponent(ep.repo)}&ep_idx=${ep.idx}${camQ()}`;
    setStatus(source === "sidecar" ? "sidecar" : "inference", "busy");
    try {
      const t0 = performance.now();
      const r = await api(url);
      state.inference = r;
      renderPlots();
      const dt = (performance.now() - t0) / 1000;
      // Sidecar mode: show which ckpt produced the injected signal so the
      // user knows what they're looking at without confusing it with the
      // currently-loaded ckpt (or lack thereof).
      if (source === "sidecar") {
        const cm = r.meta && r.meta.current_model;
        setStatus(cm ? `sidecar · ${cm}` : `sidecar ${dt.toFixed(2)}s`);
      } else {
        setStatus(r.cached ? `cached ${dt.toFixed(2)}s` : `infer ${dt.toFixed(1)}s`);
      }
    } catch (e) {
      console.error(e);
      setStatus("error", "err");
    }
  }

  // ── Plots ──────────────────────────────────────────────────────────
  function downsample(t, ps, vs, ws, maxN) {
    const N = ps.length;
    if (N <= maxN) return {
      xs: Array.from(t), ps: Array.from(ps), vs: Array.from(vs), ws: Array.from(ws),
    };
    const step = N / maxN;
    const xs = new Array(maxN), pps = new Array(maxN), vvs = new Array(maxN), wws = new Array(maxN);
    for (let i = 0; i < maxN; i++) {
      const idx = Math.min(N - 1, Math.floor(i * step));
      xs[i] = t[idx]; pps[i] = ps[idx]; vvs[i] = vs[idx]; wws[i] = ws[idx];
    }
    return { xs, ps: pps, vs: vvs, ws: wws };
  }

  function renderPlots() {
    const inf = state.inference;
    if (!inf) return;
    const N = inf.n_frames;
    const t = new Float32Array(N);
    for (let i = 0; i < N; i++) t[i] = i / Math.max(N - 1, 1);

    // Downsample for fast drawing.
    const ds = downsample(t, inf.abs_progress, inf.velocity, inf.weights, 1800);

    // Abs-head direct progress (enhanced models only). Downsample to the
    // same x-grid as the integrated rel-head curve so they overlay cleanly.
    let dsAbsHead = null;
    if (Array.isArray(inf.abs_head_progress) && inf.abs_head_progress.length === N) {
      const stepDs = N / Math.max(ds.xs.length, 1);
      dsAbsHead = new Array(ds.xs.length);
      for (let i = 0; i < ds.xs.length; i++) {
        const idx = Math.min(N - 1, Math.floor(i * stepDs));
        dsAbsHead[i] = inf.abs_head_progress[idx];
      }
    }

    // Apply the selected weighting scheme (no-op when mode is "off"). Progress
    // plot stays on raw weights; velocity + zoom switch to transformed values
    // so the color shading reflects what a per-frame downstream weight is.
    const dsV = applyScheme(ds.vs);
    const fullV = applyScheme(Array.from(inf.velocity));

    const vMin = Math.min(0, Math.min(...dsV.vs));
    const vMax = Math.max(1e-6, Math.max(...dsV.vs));
    const vRange = vMax - vMin;
    const vPad = vRange * 0.05;
    const vLo = vMin - vPad, vHi = vMax + vPad;

    // Stats — reflect the weights the user will see in the downstream
    // weighting (full-resolution transformed ws).
    const statW = fullV.ws;
    const avgW = mean(statW);
    const maxW = Math.max(...statW);
    const nZero = statW.reduce((a, w) => a + (w < 0.01 ? 1 : 0), 0);
    const statsLabel = state.rabc.mode === "off" ? "raw" : state.rabc.mode;
    els.stats.innerHTML = `<span class="muted">${statsLabel}</span> avg <b>${avgW.toFixed(3)}</b> · max <b>${maxW.toFixed(3)}</b> · zero <b>${nZero}</b>/${N} (${Math.round(100 * nZero / N)}%)`;

    // Progress plot — weight fill bounded by the progress curve + solid line.
    // Trace order: [heatmap bg, abs-head line (behind), integrated rel-head line (front)].
    // (cursor is a DOM overlay, not a Plotly trace.)
    const progressLayout = baseLayout(
      "reconstructed progress — weight-shaded", PROG,
    );
    progressLayout.yaxis.range = [0, 1];
    const progressTraces = [
      heatmapBgShaped(ds.xs, ds.ps, ds.ws, 0, 1, 120, 0.55, 0),
    ];
    if (dsAbsHead) {
      // Abs-head direct progress as a thinner muted line behind the integrated
      // line. Distinct hue so the user can see when the two heads disagree.
      progressTraces.push({
        x: ds.xs, y: dsAbsHead, mode: "lines", type: "scatter",
        line: { color: "#9aa3b8", width: 1.4, dash: "solid" },
        name: "abs head", hoverinfo: "skip", showlegend: false,
      });
    }

    progressTraces.push({
      x: ds.xs, y: ds.ps, mode: "lines", type: "scatter",
      line: { color: PROG, width: 2 },
      name: "integrated rel head", hoverinfo: "skip", showlegend: false,
    });
    Plotly.react(els.plotProgress, progressTraces, progressLayout, PLOT_CFG);

    // Velocity plot — weight fill between 0 and the velocity curve + gray line.
    // Zero reference comes from the y-axis zeroline, promoted to high contrast
    // via zerolinecolor/width.
    let velTitle;
    if (state.rabc.mode === "off") {
      velTitle = "per-frame velocity — weight-shaded";
    } else if (state.rabc.mode === "velocity_only") {
      velTitle = `per-frame velocity — clipped [${fmtNum(state.rabc.clipMin)}, ${fmtNum(state.rabc.clipMax)}]${state.rabc.thresh != null ? ` · thresh ${fmtNum(state.rabc.thresh)}` : ""}`;
    } else {
      velTitle = "per-frame velocity";
    }
    const velocityLayout = baseLayout(velTitle, FG);
    velocityLayout.yaxis.range = [vLo, vHi];
    velocityLayout.yaxis.zeroline = true;
    velocityLayout.yaxis.zerolinecolor = "#C5C5D4";
    velocityLayout.yaxis.zerolinewidth = 1.5;
    Plotly.react(els.plotVelocity, [
      heatmapBgShaped(ds.xs, dsV.vs, dsV.ws, vLo, vHi, 100, 0.55, 0),
      { x: ds.xs, y: dsV.vs, mode: "lines", type: "scatter",
        line: { color: VEL_FG, width: 1.3 },
        hoverinfo: "skip" },
    ], velocityLayout, PLOT_CFG);

    // Zoom plot — drawn once with FULL signal; scrubbing just pans its x-range.
    // For the zoom's heatmap background we use the full-resolution weights so
    // the gradient stays crisp as you pan across.
    state._plotCache = {
      xs: ds.xs, vs: dsV.vs, ws: dsV.ws, ps: ds.ps,
      vLo, vHi, N, t,
      xsFull: Array.from(t),
      vsFull: Array.from(fullV.vs),
      wsFull: Array.from(fullV.ws),
    };
    renderZoomPlotOnce();

    // Click-to-seek on the overview plots
    for (const plotEl of [els.plotProgress, els.plotVelocity, els.plotZoom]) {
      plotEl.on?.("plotly_click", (evt) => {
        if (!evt.points || !evt.points.length) return;
        const x = evt.points[0].x;
        if (els.video.duration) els.video.currentTime = x * els.video.duration;
      });
    }
    requestAnimationFrame(() => {
      for (const el of [els.plotProgress, els.plotVelocity, els.plotZoom]) {
        Plotly.Plots.resize(el);
      }
      moveCursors(state.tMark);
    });
  }

  // Heatmap whose weight-coloring is clipped to the band between `baseline`
  // and the prediction curve `ys`. Cells outside that band are null (renders
  // transparent) so the color follows the curve instead of painting the whole
  // column — mirroring render_rabc.py's fill_between / axvspan behavior.
  function heatmapBgShaped(xs, ys, ws, yLo, yHi, nY, opacity, baseline = 0) {
    const dy = (yHi - yLo) / nY;
    const yBins = new Array(nY);
    for (let j = 0; j < nY; j++) yBins[j] = yLo + (j + 0.5) * dy;
    const nX = xs.length;
    const z = new Array(nY);
    for (let j = 0; j < nY; j++) {
      const row = new Array(nX);
      const yj = yBins[j];
      for (let i = 0; i < nX; i++) {
        const yi = ys[i];
        const lo = yi < baseline ? yi : baseline;
        const hi = yi < baseline ? baseline : yi;
        row[i] = (yj >= lo && yj <= hi) ? ws[i] : null;
      }
      z[j] = row;
    }
    // zmin/zmax track the RABC clip band so heatmap coloring matches the
    // weightToRgb() ramp used for cursor labels and colorscale stops.
    const zLo = state.rabc.clipMin;
    const zHi = state.rabc.clipMax;
    const zSpan = zHi - zLo;
    return {
      type: "heatmap",
      x: xs, y: yBins, z,
      zmin: zLo,
      zmax: (isFinite(zSpan) && zSpan > 0) ? zHi : zLo + 1e-6,
      colorscale: WEIGHT_COLORSCALE,
      showscale: false,
      opacity,
      hoverinfo: "skip",
    };
  }
  // Draw the zoom plot once per episode, with the full signal. Scrubbing just
  // calls relayout to pan the xaxis range — no re-diffing of traces per frame.
  function renderZoomPlotOnce() {
    const c = state._plotCache;
    if (!c) return;
    const tc = state.tMark;
    const half = 0.05;
    const lo = tc - half;
    const hi = tc + half;

    const zoomTitle = state.rabc.mode === "off"
      ? "zoomed velocity · ±5%"
      : `zoomed velocity · ±5% · ${state.rabc.mode}`;
    const layout = baseLayout(zoomTitle, FG);
    layout.xaxis.range = [lo, hi];
    layout.yaxis.range = [c.vLo, c.vHi];
    layout.yaxis.zeroline = true;
    layout.yaxis.zerolinecolor = "#C5C5D4";
    layout.yaxis.zerolinewidth = 1.5;
    Plotly.react(els.plotZoom, [
      heatmapBgShaped(c.xsFull, c.vsFull, c.wsFull, c.vLo, c.vHi, 100, 0.6, 0),
      { x: c.xsFull, y: c.vsFull, mode: "lines", type: "scatter",
        line: { color: VEL_FG, width: 1.5 }, hoverinfo: "skip" },
    ], layout, PLOT_CFG);
  }

  function panZoomPlot() {
    const c = state._plotCache;
    if (!c) return;
    const tc = state.tMark;
    const half = 0.05;
    const lo = tc - half;
    const hi = tc + half;
    Plotly.relayout(els.plotZoom, { "xaxis.range": [lo, hi] });
  }

  // Return the pixel x-range of a plot's data area by reading Plotly's
  // computed layout. Falls back to margin-based estimates if unavailable.
  function plotXPx(plotEl) {
    const gd = plotEl && plotEl._fullLayout;
    if (gd && gd.xaxis && gd.xaxis._length != null) {
      return { x0: gd.xaxis._offset, x1: gd.xaxis._offset + gd.xaxis._length };
    }
    const w = plotEl?.clientWidth || 0;
    return { x0: 46, x1: Math.max(46, w - 14) };
  }

  // Move the DOM cursor overlays (CSS transform — GPU-accelerated, instant).
  // For progress/velocity the cursor is proportional to tMark across the full
  // [0, 1] data axis. For the zoom plot the cursor is always centered within
  // the current ±5% window.
  function moveCursors(tc) {
    const place = (plotEl, cursorEl, frac) => {
      const { x0, x1 } = plotXPx(plotEl);
      const x = x0 + frac * (x1 - x0);
      cursorEl.style.transform = `translateX(${x}px)`;
      cursorEl.style.left = "0px";  // we set translateX in absolute px
    };
    place(els.plotProgress, els.cursorProgress, tc);
    place(els.plotVelocity, els.cursorVelocity, tc);
    place(els.plotZoom,     els.cursorZoom,     0.5);  // always center
  }

  // Pan the zoom plot's x-axis. Throttled to ~20Hz so we don't saturate
  // Plotly with relayout work during continuous scrubbing.
  let _zoomPanPending = false;
  let _zoomPanLastTs = 0;
  function schedulePanZoom() {
    if (_zoomPanPending) return;
    _zoomPanPending = true;
    const now = performance.now();
    const dt = now - _zoomPanLastTs;
    const wait = Math.max(0, 50 - dt);  // at most ~20 Hz
    setTimeout(() => {
      _zoomPanPending = false;
      _zoomPanLastTs = performance.now();
      panZoomPlot();
    }, wait);
  }

  function updateCursor() {
    const inf = state.inference;
    if (!inf) return;
    const c = state._plotCache;
    if (!c) return;
    moveCursors(state.tMark);
    schedulePanZoom();
  }

  // ── Video sync — continuous cursor via wall-clock interpolation ─────
  //
  // The browser only updates `video.currentTime` at the video's decode rate
  // (typically 30 Hz here). To make the cursor look continuous during play
  // we linearly extrapolate between video-frame timestamps using the wall
  // clock: effective_t = last_report + (now - report_wall_time) * rate.
  //
  // When paused or seeking we fall back to the raw currentTime.
  let _vidLastT = 0;          // last raw video.currentTime we sampled
  let _vidLastWall = 0;       // wall-clock time of that sample
  let _cursorDisplayLastT = -1;

  function syncFrame() {
    if (state.inference && els.video.duration) {
      const reportedT = els.video.currentTime;
      const wallNow = performance.now();

      // If the video ticked to a new timestamp, anchor the interpolator.
      if (reportedT !== _vidLastT) {
        _vidLastT = reportedT;
        _vidLastWall = wallNow;
      }

      // If playing, extrapolate; otherwise use the raw value.
      let displayT = reportedT;
      if (!els.video.paused && !els.video.seeking && els.video.readyState >= 2) {
        const rate = els.video.playbackRate || 1;
        // Cap extrapolation at one video frame (avoids overshoot when the
        // browser falls behind). Assume 30 fps if we don't know.
        const fps = state.fps || 30;
        const maxAhead = 1 / fps;
        const elapsed = Math.min(maxAhead, (wallNow - _vidLastWall) / 1000);
        displayT = reportedT + elapsed * rate;
        displayT = Math.min(displayT, els.video.duration);
      }

      if (Math.abs(displayT - _cursorDisplayLastT) > 1e-4) {
        _cursorDisplayLastT = displayT;
        const tn = Math.max(0, Math.min(1, displayT / els.video.duration));
        state.tMark = tn;
        const N = state.inference.n_frames;
        const fi = Math.min(N - 1, Math.round(tn * (N - 1)));
        // When rabc toggle is on, show the transformed weight/velocity the
        // downstream pipeline would see; otherwise the raw values.
        const cache = state._plotCache;
        const w = cache ? cache.wsFull[fi] : state.inference.weights[fi];
        const v = cache ? cache.vsFull[fi] : state.inference.velocity[fi];
        els.frameLabel.textContent = `frame ${fi} / ${N - 1}`;
        els.timeLabel.textContent  = `${msTime(displayT)} / ${msTime(els.video.duration)}`;
        els.weightLabel.textContent = `w ${w.toFixed(2)}  ·  v ${v.toFixed(2)}`;
        els.weightLabel.style.color = weightToRgb(w);
        els.weightLabel.style.borderColor = weightToRgba(w, 0.8);
        updateCursor();
      }
    }
    requestAnimationFrame(syncFrame);
  }
  requestAnimationFrame(syncFrame);

  // Re-anchor interpolator on seek / load / rate change.
  const reanchor = () => { _vidLastT = -1; _cursorDisplayLastT = -1; };
  els.video.addEventListener("seeking",        reanchor);
  els.video.addEventListener("seeked",         reanchor);
  els.video.addEventListener("loadedmetadata", reanchor);
  els.video.addEventListener("ratechange",     reanchor);
  els.video.addEventListener("play",           reanchor);
  els.video.addEventListener("pause",          reanchor);

  function mean(arr) { let s = 0; for (const x of arr) s += x; return s / arr.length; }
  function fmtNum(x) {
    if (!isFinite(x)) return "—";
    return (Math.abs(x) < 10 && Math.round(x) !== x)
      ? x.toFixed(2).replace(/\.?0+$/, "") || "0"
      : String(x);
  }

  // ── RABC controls ──────────────────────────────────────────────────
  function readRabcFromInputs() {
    const cMin = parseFloat(els.rabcClipMin.value);
    const cMax = parseFloat(els.rabcClipMax.value);
    const thRaw = els.rabcThresh.value.trim();
    const th = thRaw === "" ? null : parseFloat(thRaw);
    state.rabc.clipMin = isFinite(cMin) ? cMin : 0;
    state.rabc.clipMax = isFinite(cMax) ? cMax : 1;
    state.rabc.thresh  = (th !== null && isFinite(th)) ? th : null;
    state.rabc.mode = els.rabcMode ? els.rabcMode.value : "off";
    els.rabcField.classList.toggle("disabled", state.rabc.mode === "off");
  }
  function onRabcChange() {
    readRabcFromInputs();
    if (state.inference) renderPlots();
  }
  if (els.rabcMode) els.rabcMode.addEventListener("change", onRabcChange);
  els.rabcClipMin.addEventListener("change", onRabcChange);
  els.rabcClipMax.addEventListener("change", onRabcChange);
  els.rabcThresh.addEventListener("change", onRabcChange);
  readRabcFromInputs();

  // Source toggle: switching between live ckpt inference and the dataset's
  // injected sidecar re-fetches the signal for the currently-selected
  // episode so the plots update immediately. Persists across reloads.
  if (els.signalSource) {
    const SAVED_SOURCE = (typeof localStorage !== "undefined"
                          && localStorage.getItem("inspector_source")) || "ckpt";
    if (SAVED_SOURCE === "sidecar" || SAVED_SOURCE === "ckpt") {
      els.signalSource.value = SAVED_SOURCE;
    }
    els.signalSource.addEventListener("change", () => {
      try { localStorage.setItem("inspector_source", els.signalSource.value); } catch (_) {}
      // Re-fetch the inspector plot for the current episode. The velocity
      // summary column is checkpoint-derived either way, so it is unaffected
      // by the source toggle and needs no re-scoring.
      if (state.selectedEp) selectEpisode(state.selectedEp);
    });
  }

  // ── Control wiring ─────────────────────────────────────────────────
  els.ckptSelect.addEventListener("change", async (e) => {
    const path = e.target.value;
    if (!path) return;
    setStatus("loading ckpt", "busy");
    try {
      const r = await api("/api/checkpoint", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ path }),
      });
      state.currentCkpt = r.current;
      state.summaryJob = null;     // cancel any in-flight scoring job
      state.summary.clear();       // summaries depend on the checkpoint
      setStatus("ckpt loaded");
      renderEpisodeList();
      if (state.selectedEp) selectEpisode(state.selectedEp);
      if (sortNeedsSummary(els.epSort.value)) ensureSummaries();
    } catch (err) {
      console.error(err);
      setStatus("load failed", "err");
    }
  });

  els.cameraSelect.addEventListener("change", () => {
    state.cameraKey = els.cameraSelect.value || null;
    state.summaryJob = null;   // summaries are per-camera
    state.summary.clear();
    state.selectedEp = null;
    refreshEpisodes();
  });

  els.epSearch.addEventListener("input", renderEpisodeList);
  els.epSort.addEventListener("change", () => {
    renderEpisodeList();
    if (sortNeedsSummary(els.epSort.value)) ensureSummaries();
  });

  window.addEventListener("resize", () => {
    for (const el of [els.plotProgress, els.plotVelocity, els.plotZoom]) {
      if (el && el.data) Plotly.Plots.resize(el);
    }
    moveCursors(state.tMark);
  });

  // ── Init ───────────────────────────────────────────────────────────
  (async () => {
    setStatus("loading");
    try {
      await Promise.all([loadCheckpoints(), loadDatasets()]);
      rebuildCameraOptions();
      setStatus("idle");
    } catch (e) {
      console.error(e);
      setStatus("init failed", "err");
    }
  })();
})();
