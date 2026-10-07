"use strict";

(() => {
  const article = document.getElementById("article");
  const panel = document.getElementById("rollout-panel");
  const backdrop = document.getElementById("rollout-backdrop");
  const closeButton = document.getElementById("rollout-close");
  const player = document.getElementById("rollout-player");
  const loaded = document.getElementById("rollout-loaded");
  const status = document.getElementById("rollout-status");
  const clipsList = document.getElementById("rollout-clips");
  const mediaError = document.getElementById("rollout-video-error");
  const triggerSelector = ".chart-variant, .performance-cell, .demo-link, .model-output-open";
  const resultTables = new Set(["main-results", "molmoact2-results"]);
  const manifestPromises = new Map();
  const taskNames = new Map([
    ["slanted board", 0], ["constrained extraction", 0],
    ["odd geometry", 1], ["odd-object grasping", 1],
    ["precision insertion", 2], ["precision insert", 2],
    ["stand object", 3], ["standing stability", 3],
    ["tool drawer", 4], ["tool-assisted drawer", 4],
    ["unlock ring", 5], ["ring release", 5],
  ]);
  let activeGroup = null;
  let activeClip = null;
  let returnFocus = null;
  let requestVersion = 0;

  function syncLayout() {
    const isOpen = !panel.hidden;
    // Use the space beside the centered article without changing its layout.
    const modal = isOpen && panel.getBoundingClientRect().left < article.getBoundingClientRect().right + 28;
    document.body.classList.toggle("rollout-modal-open", modal);
    backdrop.hidden = !modal;
    article.inert = modal;
    panel.setAttribute("role", modal ? "dialog" : "complementary");
    if (modal) panel.setAttribute("aria-modal", "true");
    else panel.removeAttribute("aria-modal");
    if (modal && !window.MediaViewer?.isOpen() && !panel.contains(document.activeElement)) closeButton.focus({ preventScroll: true });
  }

  function stopVideo() {
    window.MediaViewer?.close({ restoreFocus: false });
    window.RolloutOutputs.setClip(null, false);
    player.pause();
    activeClip = null;
    player.removeAttribute("src");
    player.removeAttribute("poster");
    player.replaceChildren();
    player.load();
    mediaError.hidden = true;
  }

  function closePanel({ restoreFocus = true } = {}) {
    if (panel.hidden) return;
    const trigger = returnFocus;
    requestVersion += 1;
    stopVideo();
    panel.hidden = true;
    activeGroup = null;
    syncLayout();
    document.querySelectorAll(triggerSelector).forEach((control) => {
      control.classList.remove("has-rollouts-open");
      control.setAttribute("aria-expanded", "false");
    });
    const collapsedSection = trigger?.closest("details:not([open])");
    const focusTarget = collapsedSection ? collapsedSection.querySelector("summary")
      : trigger?.isConnected && trigger.getBoundingClientRect().width > 0
      ? trigger : document.getElementById(trigger?.dataset.filterControl);
    if (restoreFocus && focusTarget) focusTarget.focus({ preventScroll: true });
    returnFocus = null;
  }

  function outcome(clip) {
    if (typeof clip.success !== "boolean") return "";
    return clip.success ? "Success" : "Failure";
  }

  function displayLabel(value = "") {
    const parts = String(value).split("·").map((part) => {
      const label = part.trim();
      const taskId = taskNames.get(label.toLowerCase().replaceAll("_", " "));
      return taskId !== undefined ? `task ${taskId}`
        : label.replace(/\btask\s*0*(\d+)(?:\s*(?:[-/@]\s*|\s+)(easy|medium|extra[ -]?hard|xhard|hard)\b)?/gi,
          (_, id, difficulty) => `task ${Number(id)}${difficulty ? ` / ${difficultyLabel(difficulty)}` : ""}`);
    });
    const unique = [...new Set(parts.filter(Boolean))];
    const combined = [];
    for (let index = 0; index < unique.length; index += 1) {
      const difficulty = difficultyLabel(unique[index + 1]);
      if (/\btask \d+$/.test(unique[index]) && difficulty) {
        combined.push(`${unique[index]} / ${difficulty}`);
        index += 1;
      } else combined.push(unique[index]);
    }
    return combined.join(" · ");
  }

  function difficultyLabel(value = "") {
    const label = String(value).trim().toLowerCase().replace(/[ -]/g, "");
    return label === "extrahard" ? "xhard" : ["easy", "medium", "hard", "xhard"].includes(label) ? label : "";
  }

  function setDisplayLabel(element, value) {
    const label = displayLabel(value);
    const pattern = /\btask \d+(?: \/ (?:easy|medium|hard|xhard))?\b/g;
    const fragment = document.createDocumentFragment();
    const appendLabelText = (value) => {
      const span = document.createElement("span");
      window.BlogMath.setText(span, value);
      fragment.append(...span.childNodes);
    };
    let cursor = 0;
    for (const match of label.matchAll(pattern)) {
      appendLabelText(label.slice(cursor, match.index));
      const task = document.createElement("code");
      task.className = "task-label";
      task.textContent = match[0];
      fragment.append(task);
      cursor = match.index + match[0].length;
    }
    appendLabelText(label.slice(cursor));
    element.replaceChildren(fragment);
  }

  function groupContext(group) {
    if (group.context) return group.context;
    const first = group.clips[0];
    const benchmark = String(group.benchmark || first.benchmark || "");
    const taskName = String(first.task ?? "").replaceAll("_", " ").toLowerCase();
    const taskId = first.taskId ?? (/^\d+$/.test(taskName) ? Number(taskName) : taskNames.get(taskName));
    if (taskId === undefined || taskId === null) return "";
    const difficulty = difficultyLabel(first.difficulty);
    const benchmarkLabel = /libero/i.test(benchmark) ? `${benchmark} · ` : "";
    return `${benchmarkLabel}task ${taskId}${difficulty ? ` / ${difficulty}` : ""}`;
  }

  function normalizedLabel(value) {
    return String(value).normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  }

  function contextWithoutTitle(context, title) {
    const heading = ` ${normalizedLabel(title)} `;
    return displayLabel(context).split(" · ")
      .filter((part) => !heading.includes(` ${normalizedLabel(part)} `)).join(" · ");
  }

  function clipTitle(clip, index) {
    return displayLabel(clip.label || `Example ${index + 1} · Seed ${clip.seed}`);
  }

  function clipMetrics(clip) {
    if (Number.isFinite(clip.progress) && Number.isFinite(clip.finalProgress) && Number.isFinite(clip.steps)) {
      return `Peak PS ${clip.progress.toFixed(2)} · Final PS ${clip.finalProgress.toFixed(2)} · ${clip.steps.toLocaleString("en-US")} execution steps`;
    }
    return Number.isFinite(clip.durationSeconds) ? `Duration ${clip.durationSeconds.toFixed(1)} s` : "";
  }

  function selectClip(clip, play = false) {
    if (activeClip && activeClip.id === clip.id) {
      if (play) player.play().catch(() => {});
      return;
    }
    stopVideo();
    activeClip = clip;
    window.RolloutOutputs.setClip(clip, !resultTables.has(activeGroup.chartTableId));
    player.poster = clip.poster;
    const index = activeGroup.clips.findIndex((item) => item.id === clip.id);
    const title = window.BlogMath.text(document.getElementById("rollout-title"));
    const selectedTitle = clipTitle(clip, index);
    player.setAttribute("aria-label", [...new Set([title, selectedTitle, outcome(clip)].filter(Boolean))].join(", "));
    // VP9 is supported by the embedded browser even when H.264 is unavailable.
    [[clip.webm, "video/webm"], [clip.mp4, "video/mp4"]].forEach(([src, type]) => {
      const source = document.createElement("source");
      source.src = src;
      source.type = type;
      if (type === "video/mp4") source.addEventListener("error", () => {
        if (activeClip?.id === clip.id) mediaError.hidden = false;
      });
      player.append(source);
    });
    const currentTitle = document.getElementById("rollout-current-title");
    setDisplayLabel(currentTitle, contextWithoutTitle(selectedTitle, title));
    currentTitle.hidden = !currentTitle.textContent;
    const badge = document.getElementById("rollout-current-outcome");
    badge.textContent = outcome(clip);
    badge.hidden = typeof clip.success !== "boolean";
    badge.classList.toggle("is-success", clip.success === true);
    const metrics = document.getElementById("rollout-current-metrics");
    metrics.textContent = clipMetrics(clip);
    metrics.hidden = !metrics.textContent;
    const currentHeading = panel.querySelector(".rollout-current-heading");
    currentHeading.hidden = currentTitle.hidden && badge.hidden;
    currentHeading.closest("figcaption").hidden = currentHeading.hidden && metrics.hidden;
    clipsList.querySelectorAll("button").forEach((button) => {
      const selected = button.dataset.clipId === clip.id;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    player.load();
    if (play) player.play().catch(() => {});
  }

  function populate(group, summary) {
    activeGroup = group;
    const mainSamples = resultTables.has(group.chartTableId);
    const failureDemo = group.chartTableId === "failure-behaviors";
    const demo = failureDemo || group.chartTableId === "promising-behaviors" || group.chartTableId === "article-demos";
    const singleDemo = demo && group.clips.length === 1;
    const title = displayLabel(group.title || group.variant);
    setDisplayLabel(document.getElementById("rollout-title"), title);
    panel.querySelector(".rollout-eyebrow").textContent = failureDemo ? "Failure examples" : group.chartTableId === "article-demos" ? "Demo" : demo ? "Promising behaviors" : mainSamples ? "Task samples" : "Rollout examples";
    const first = group.clips[0];
    const context = document.getElementById("rollout-context");
    setDisplayLabel(context, contextWithoutTitle(groupContext(group), title));
    context.hidden = mainSamples || !context.textContent;
    const scope = document.getElementById("rollout-scope");
    setDisplayLabel(scope, demo ? summary || group.description || "" : "");
    scope.hidden = !scope.textContent || normalizedLabel(window.BlogMath.text(scope)) === normalizedLabel(title);
    const listTitle = panel.querySelector(".rollout-list-title");
    listTitle.hidden = singleDemo;
    clipsList.hidden = singleDemo;
    listTitle.firstChild.textContent = demo ? "Choose an example " : mainSamples ? "Choose a sample " : "Choose a rollout ";
    listTitle.querySelector("span").textContent = `${group.clips.length} ${mainSamples ? (group.clips.length === 1 ? "sample" : "samples") : "examples"}`;
    clipsList.setAttribute("aria-label", demo ? "Demo choices" : mainSamples ? "Sample choices" : "Rollout choices");
    clipsList.replaceChildren();
    (singleDemo ? [] : group.clips).forEach((clip, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "rollout-clip";
      button.dataset.clipId = clip.id;
      button.setAttribute("aria-label", `Play ${clipTitle(clip, index)}, ${outcome(clip)}. ${clipMetrics(clip)}.`);
      button.setAttribute("aria-controls", "rollout-player");
      const image = document.createElement("img");
      image.src = clip.poster;
      image.alt = "";
      image.width = 1280;
      image.height = 360;
      image.decoding = "async";
      const text = document.createElement("span");
      text.className = "rollout-clip-text";
      const label = document.createElement("strong");
      setDisplayLabel(label, clip.label || `${index + 1}. Seed ${clip.seed}`);
      const detail = document.createElement("span");
      detail.textContent = Number.isFinite(clip.progress)
        ? `${outcome(clip)} · PS ${clip.progress.toFixed(2)}` : clipMetrics(clip);
      text.append(label, detail);
      button.append(image, text);
      button.addEventListener("click", () => selectClip(clip, true));
      clipsList.append(button);
    });
    loaded.hidden = false;
    status.textContent = "";
    selectClip(first);
  }

  function getManifest(path) {
    if (!manifestPromises.has(path)) {
      const promise = fetch(path)
        .then((response) => {
          if (!response.ok) throw new Error("Video list unavailable");
          return response.json();
        })
        .catch((error) => {
          manifestPromises.delete(path);
          throw error;
        });
      manifestPromises.set(path, promise);
    }
    return manifestPromises.get(path);
  }

  window.addEventListener("experiment:open-rollouts", async (event) => {
    const { tableId, variant, trigger, manifest = "rollout-videos.json", title = variant, summary, group: providedGroup, showModelOutputs = false } = event.detail;
    if (!panel.hidden && activeGroup?.chartTableId === tableId && activeGroup.variant === variant && returnFocus === trigger) {
      if (showModelOutputs) window.RolloutOutputs.show();
      closeButton.focus({ preventScroll: true });
      return;
    }
    const version = ++requestVersion;
    returnFocus = trigger;
    stopVideo();
    activeGroup = null;
    loaded.hidden = true;
    status.textContent = tableId === "promising-behaviors" || tableId === "failure-behaviors" ? "Loading demo…" : "Loading samples…";
    setDisplayLabel(document.getElementById("rollout-title"), title);
    panel.querySelector(".rollout-eyebrow").textContent = tableId === "failure-behaviors" ? "Failure examples" : tableId === "promising-behaviors" ? "Promising behaviors" : resultTables.has(tableId) ? "Task samples" : "Rollout examples";
    panel.hidden = false;
    panel.querySelector(".rollout-content").scrollTop = 0;
    syncLayout();
    document.querySelectorAll(triggerSelector).forEach((control) => {
      const selected = control === trigger;
      control.classList.toggle("has-rollouts-open", selected);
      control.setAttribute("aria-expanded", String(selected));
    });
    requestAnimationFrame(() => {
      if (version !== requestVersion || panel.hidden) return;
      closeButton.focus({ preventScroll: true });
    });
    try {
      const data = providedGroup ? null : await getManifest(manifest);
      if (version !== requestVersion || panel.hidden) return;
      const group = providedGroup || data.groups.find((item) => item.chartTableId === tableId && item.variant === variant);
      if (!group || !Array.isArray(group.clips) || !group.clips.length) throw new Error("No matching samples");
      populate({ ...group, title: group.title || title }, summary);
      if (showModelOutputs) window.RolloutOutputs.show();
    } catch (error) {
      if (version !== requestVersion || panel.hidden) return;
      status.textContent = "The videos could not be loaded. Close this panel and select the link, cell, or bar again to retry.";
    }
  });

  closeButton.addEventListener("click", closePanel);
  backdrop.addEventListener("click", closePanel);
  window.addEventListener("article:navigate", () => closePanel({ restoreFocus: false }));
  window.addEventListener("resize", syncLayout);
  player.addEventListener("error", () => {
    if (activeClip) mediaError.hidden = false;
  });
  player.addEventListener("loadedmetadata", () => {
    mediaError.hidden = true;
  });
  document.addEventListener("keydown", (event) => {
    if (panel.hidden || window.MediaViewer?.isOpen()) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closePanel();
    } else if (event.key === "Tab" && panel.getAttribute("aria-modal") === "true") {
      const focusable = Array.from(panel.querySelectorAll("button, a[href], video, summary, select, pre[tabindex]"))
        .filter((item) => item.getClientRects().length && !item.disabled);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
})();
