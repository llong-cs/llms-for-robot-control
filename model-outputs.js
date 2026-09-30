"use strict";

(() => {
  const section = document.getElementById("model-output-section");
  const options = document.getElementById("model-output-options");
  const toggle = document.getElementById("model-output-toggle");
  const content = document.getElementById("model-output-content");
  const status = document.getElementById("model-output-status");
  const controls = document.getElementById("model-output-controls");
  const previous = document.getElementById("model-output-previous");
  const next = document.getElementById("model-output-next");
  const select = document.getElementById("model-output-turn");
  const timeLabel = document.getElementById("model-output-time");
  const jump = document.getElementById("model-output-jump");
  const download = document.getElementById("model-output-download");
  const code = document.getElementById("model-output-code");
  const caption = section.querySelector(".model-output-caption");
  const player = document.getElementById("rollout-player");
  const requests = new Map();
  let clip = null;
  let trace = null;
  let version = 0;

  function formatVideoTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return null;
    const centiseconds = Math.round(seconds * 100);
    const minutes = Math.floor(centiseconds / 6000);
    const remainder = Math.floor(centiseconds % 6000 / 100);
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}.${String(centiseconds % 100).padStart(2, "0")}`;
  }

  function getJSON(path) {
    if (!requests.has(path)) {
      const request = fetch(path).then((response) => {
        if (!response.ok) throw new Error("Model output unavailable");
        return response.json();
      }).catch((error) => {
        requests.delete(path);
        throw error;
      });
      requests.set(path, request);
    }
    return requests.get(path);
  }

  function renderTurn() {
    const index = select.selectedIndex;
    const turn = trace?.turns[index];
    if (!turn) return;
    // Expand argument JSON for reading, retaining every field and full note.
    // The download keeps the exact original tool-call representation.
    const output = { tool_calls: turn.toolCalls.map((call) => {
      if (typeof call.arguments !== "string") return call;
      try { return { ...call, arguments: JSON.parse(call.arguments) }; }
      catch (error) { return call; }
    }) };
    if (turn.responseText) output.response_text = turn.responseText;
    if (turn.error) output.error = turn.error;
    if (turn.errorKind) output.error_kind = turn.errorKind;
    code.textContent = JSON.stringify(output, null, 2);
    caption.textContent = turn.toolCalls.length
      ? "Complete tool calls, including action parameters and notes."
      : "No tool call was recorded for this turn.";
    code.scrollTop = 0;
    code.hidden = false;
    previous.disabled = index === 0;
    next.disabled = index === trace.turns.length - 1;
    const time = formatVideoTime(turn.timeSeconds);
    timeLabel.textContent = time ? `Video time: ${time}` : "Video time unavailable";
    jump.hidden = time === null;
    jump.textContent = time ? `Jump to ${time}` : "Jump to this turn";
    jump.disabled = player.readyState < 1;
  }

  async function loadTurns() {
    const requestVersion = ++version;
    status.textContent = "Loading model outputs…";
    controls.hidden = true;
    code.hidden = true;
    try {
      const index = await getJSON("model-outputs/index.json");
      if (requestVersion !== version) return;
      const entry = index.clips[clip.mp4];
      if (!entry || entry.status !== "available") {
        status.textContent = entry?.reason || "No model output recording is available for this video.";
        return;
      }
      const result = await getJSON(entry.trace);
      if (requestVersion !== version) return;
      if (!Array.isArray(result.turns) || !result.turns.length) {
        status.textContent = "No model turns were recorded for this video.";
        return;
      }
      trace = result;
      download.href = entry.trace;
      select.replaceChildren();
      trace.turns.forEach((turn) => {
        const option = document.createElement("option");
        option.textContent = `Turn ${turn.turn} · ${formatVideoTime(turn.timeSeconds) || "Time unavailable"}`;
        select.append(option);
      });
      status.textContent = `${trace.turns.length} recorded turns. ${result.message || ""}`.trim();
      controls.hidden = false;
      renderTurn();
    } catch (error) {
      if (requestVersion !== version) return;
      status.textContent = "The model outputs could not be loaded. Hide and show them again to retry.";
    }
  }

  function setClip(selected, enabled) {
    version += 1;
    clip = selected;
    trace = null;
    options.hidden = !enabled || !selected;
    section.hidden = true;
    content.hidden = true;
    controls.hidden = true;
    code.hidden = true;
    code.textContent = "";
    status.textContent = "";
    timeLabel.textContent = "";
    select.replaceChildren();
    download.removeAttribute("href");
    toggle.textContent = "Show model outputs";
    toggle.setAttribute("aria-expanded", "false");
  }

  function show() {
    if (options.hidden || !content.hidden) return;
    section.hidden = false;
    content.hidden = false;
    toggle.textContent = "Hide model outputs";
    toggle.setAttribute("aria-expanded", "true");
    loadTurns();
  }

  toggle.addEventListener("click", () => {
    const show = content.hidden;
    section.hidden = !show;
    content.hidden = !show;
    toggle.textContent = show ? "Hide model outputs" : "Show model outputs";
    toggle.setAttribute("aria-expanded", String(show));
    if (show) loadTurns();
    else version += 1;
  });
  select.addEventListener("change", renderTurn);
  previous.addEventListener("click", () => {
    if (select.selectedIndex > 0) select.selectedIndex -= 1;
    renderTurn();
  });
  next.addEventListener("click", () => {
    if (trace && select.selectedIndex < trace.turns.length - 1) select.selectedIndex += 1;
    renderTurn();
  });
  jump.addEventListener("click", () => {
    const turn = trace?.turns[select.selectedIndex];
    if (!Number.isFinite(turn?.timeSeconds) || player.readyState < 1 || !Number.isFinite(player.duration)) return;
    player.pause();
    player.currentTime = Math.min(turn.timeSeconds, player.duration);
  });
  player.addEventListener("loadedmetadata", () => { jump.disabled = false; });
  window.RolloutOutputs = { setClip, show };
})();
