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
    if (modal && !panel.contains(document.activeElement)) closeButton.focus({ preventScroll: true });
  }

  function stopVideo() {
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

  function clipTitle(clip, index) {
    return clip.label || `Example ${index + 1} · Seed ${clip.seed}`;
  }

  function clipMetrics(clip) {
    if (Number.isFinite(clip.progress) && Number.isFinite(clip.finalProgress) && Number.isFinite(clip.steps)) {
      return `Peak PS ${clip.progress.toFixed(2)} · Final PS ${clip.finalProgress.toFixed(2)} · ${clip.steps.toLocaleString("en-US")} control steps`;
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
    player.setAttribute("aria-label", [activeGroup.title || activeGroup.variant, clipTitle(clip, index), outcome(clip)].filter(Boolean).join(", "));
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
    document.getElementById("rollout-current-title").textContent = clipTitle(clip, index);
    const badge = document.getElementById("rollout-current-outcome");
    badge.textContent = outcome(clip);
    badge.hidden = typeof clip.success !== "boolean";
    badge.classList.toggle("is-success", clip.success === true);
    const metrics = document.getElementById("rollout-current-metrics");
    metrics.textContent = clipMetrics(clip);
    metrics.hidden = !metrics.textContent;
    const directLink = document.getElementById("rollout-open-video");
    directLink.href = player.canPlayType('video/webm; codecs="vp9"') ? clip.webm : clip.mp4;
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
    document.getElementById("rollout-title").textContent = group.title || group.variant;
    panel.querySelector(".rollout-eyebrow").textContent = failureDemo ? "Failure examples" : group.chartTableId === "article-demos" ? "Demo" : demo ? "Promising behaviors" : mainSamples ? "Task samples" : "Rollout examples";
    const first = group.clips[0];
    document.getElementById("rollout-context").textContent =
      group.context || `Task 0 · Slanted board · ${first.difficulty === "xhard" ? "Extra hard" : first.difficulty}`;
    const scope = document.getElementById("rollout-scope");
    scope.textContent = summary || group.description || (mainSamples ? "Samples for the selected task and difficulty." : group.clipStepLimit
      ? `First ${group.clipStepLimit} control steps, or earlier completion. These clips follow the chart’s cutoff.`
      : `Full rollouts · ${group.sourceMaxSteps.toLocaleString("en-US")} control-step budget.`);
    const listTitle = panel.querySelector(".rollout-list-title");
    listTitle.hidden = singleDemo;
    clipsList.hidden = singleDemo;
    listTitle.firstChild.textContent = demo ? "Choose an example " : mainSamples ? "Choose a sample " : "Choose a rollout ";
    listTitle.querySelector("span").textContent = `${group.clips.length} ${mainSamples ? (group.clips.length === 1 ? "sample" : "samples") : "examples"}`;
    const selectionNote = panel.querySelector(".rollout-selection-note");
    selectionNote.hidden = demo;
    selectionNote.textContent = mainSamples
      ? "Samples illustrate successful and failed attempts where available. The table summarizes all evaluated rollouts."
      : "Four examples selected to show a range of outcomes. Their success rate need not match the chart average.";
    panel.querySelector(".rollout-source").hidden = demo;
    panel.querySelector(".rollout-source summary").textContent = mainSamples ? "Sample files" : "Source experiment";
    document.getElementById("rollout-source-run").textContent = demo ? "" : mainSamples
      ? group.clips.map((clip) => `${clip.source || group.source}/${clip.sourceFile}`).join("\n")
      : group.runName;
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
      label.textContent = clip.label || `${index + 1}. Seed ${clip.seed}`;
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
    document.getElementById("rollout-title").textContent = title;
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
  window.addEventListener("article:open-concept", () => closePanel({ restoreFocus: false }));
  window.addEventListener("article:navigate", () => closePanel({ restoreFocus: false }));
  window.addEventListener("resize", syncLayout);
  player.addEventListener("error", () => {
    if (activeClip) mediaError.hidden = false;
  });
  player.addEventListener("loadedmetadata", () => {
    mediaError.hidden = true;
    if (player.currentSrc) document.getElementById("rollout-open-video").href = player.currentSrc;
  });
  document.addEventListener("keydown", (event) => {
    if (panel.hidden) return;
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
