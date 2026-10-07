"use strict";

// Manual videos share one playback slot; looping illustrations are independent.
document.addEventListener("play", (event) => {
  if (!(event.target instanceof HTMLVideoElement) || event.target.hasAttribute("data-looping-illustration")) return;
  document.querySelectorAll("video:not([data-looping-illustration])").forEach((other) => {
    if (other !== event.target) other.pause();
  });
}, true);

document.querySelectorAll("#article details").forEach((disclosure) => {
  disclosure.addEventListener("toggle", () => {
    if (!disclosure.open) disclosure.querySelectorAll("video").forEach((video) => video.pause());
  });
});

// Present lightweight video assets as noninteractive, visible-only animations.
document.querySelectorAll("video[data-looping-illustration]").forEach((video) => {
  let inView = false;
  const updatePlayback = () => {
    const visible = inView && !document.hidden && !video.closest("details:not([open])");
    if (visible) {
      if (video.paused) video.play().catch(() => { /* Keep the poster if autoplay is unavailable. */ });
    } else {
      video.pause();
    }
  };
  const observer = new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting && entry.intersectionRatio >= 0.05;
    updatePlayback();
  }, { threshold: [0, 0.05] });
  observer.observe(video);
  for (let parent = video.parentElement; parent; parent = parent.parentElement) {
    if (parent instanceof HTMLDetailsElement) parent.addEventListener("toggle", updatePlayback);
  }
  document.addEventListener("visibilitychange", updatePlayback);
});

// Each illustration selects its horizon independently.
document.querySelectorAll("[data-control-demo-group]").forEach((controlDemo) => {
  const selector = controlDemo.querySelector(".control-demo-selector");
  const buttons = Array.from(controlDemo.querySelectorAll("[data-control-demo]"));
  const examples = Array.from(controlDemo.querySelectorAll("[data-control-horizon]"));
  const selectHorizon = (horizon) => {
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.controlDemo === horizon)));
    examples.forEach((figure) => {
      const selected = figure.dataset.controlHorizon === horizon;
      if (!selected) figure.querySelector("video")?.pause();
      figure.hidden = !selected;
    });
  };
  if (selector && buttons.length && examples.length) {
    buttons.forEach((button) => button.addEventListener("click", () => selectHorizon(button.dataset.controlDemo)));
    selectHorizon(buttons[0].dataset.controlDemo);
    selector.hidden = false;
  }
});

const videos = Array.from(document.querySelectorAll("#article video"));
videos.forEach((video) => {
  const figure = video.closest("figure");
  const errorMessage = figure.querySelector(".media-error");
  const showError = () => { errorMessage.hidden = false; };
  video.addEventListener("error", showError);
  const lastSource = video.querySelector("source:last-of-type");
  if (lastSource) lastSource.addEventListener("error", showError);
  video.addEventListener("loadedmetadata", () => {
    errorMessage.hidden = true;
    const directLink = figure.querySelector(".video-link");
    if (directLink) directLink.href = video.currentSrc;
  });
  if (video.error) showError();

  // Scripted control illustrations have no model-generated output to display.
  if (video.dataset.modelOutputs === "none") return;

  const mp4 = video.querySelector('source[type="video/mp4"]')?.getAttribute("src");
  const webm = video.querySelector('source[type="video/webm"]')?.getAttribute("src");
  if (!mp4 || !webm) return;
  const title = window.BlogMath.text(figure.querySelector("h4")) || video.getAttribute("aria-label");
  const context = window.BlogMath.text(figure.querySelector(".video-caption")).replace(/\s*·?\s*(?:Open|Enlarge) video\s*$/, "").trim() || "";
  const description = window.BlogMath.text(figure.querySelector("figcaption p")) || "Recorded demonstration.";
  const variant = mp4.split("/").pop().replace(/\.mp4$/, "");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "model-output-open";
  button.textContent = "View model outputs";
  button.setAttribute("aria-label", `View model outputs for ${title}`);
  button.setAttribute("aria-controls", "rollout-panel");
  button.setAttribute("aria-expanded", "false");
  button.addEventListener("click", () => {
    video.pause();
    window.dispatchEvent(new CustomEvent("experiment:open-rollouts", {
      detail: {
        tableId: "article-demos", variant, trigger: button, title, showModelOutputs: true,
        group: {
          chartTableId: "article-demos", variant, title, context, description,
          clips: [{ id: variant, label: title, mp4, webm, poster: video.getAttribute("poster") }],
        },
      },
    }));
  });
  figure.append(button);
});

// Demo links, result cells, and chart bars share one sample viewer.
document.querySelectorAll(".demo-link").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    window.dispatchEvent(new CustomEvent("experiment:open-rollouts", {
      detail: {
        tableId: link.dataset.table || "promising-behaviors", variant: link.dataset.demo, trigger: link,
        manifest: link.dataset.manifest || "promising-behaviors.json", title: window.BlogMath.text(link).trim(),
      },
    }));
  });
});

document.querySelectorAll(".performance-cell").forEach((cell) => {
  cell.addEventListener("click", () => {
    const difficulty = cell.dataset.difficulty;
    const sr = cell.querySelector(".performance-sr").textContent;
    const tableId = cell.dataset.table || "main-results";
    const model = cell.dataset.model || "Astra";
    window.dispatchEvent(new CustomEvent("experiment:open-rollouts", {
      detail: {
        tableId, variant: cell.dataset.variant, trigger: cell,
        manifest: cell.dataset.manifest || "main-samples.json", title: `${model} · task ${cell.dataset.task} / ${difficulty}`,
        summary: `${model} aggregate results · SR ${sr} · PS ${cell.dataset.ps} · ${cell.dataset.successes}/${cell.dataset.total} successes.`,
      },
    }));
  });
});

// Compare current table values by task and difficulty, without duplicating data.
(() => {
  const toggle = document.getElementById("molmoact2-compare");
  const control = document.getElementById("molmoact2-compare-control");
  const table = document.getElementById("molmoact2-results");
  const hint = document.getElementById("molmoact2-results-hint");
  if (!toggle || !control || !table || !hint) return;
  const key = (cell) => `${cell.dataset.task}:${cell.dataset.difficulty}`;
  const astra = new Map(Array.from(document.querySelectorAll("#main-results .performance-cell"),
    (cell) => [key(cell), cell]));
  const pairs = Array.from(table.querySelectorAll(".performance-cell"), (cell) => ({ cell, reference: astra.get(key(cell)) }));
  if (!pairs.length || pairs.some(({ cell, reference }) =>
    !reference || [cell, reference].some((item) => !item.dataset.ps?.trim() || !Number.isFinite(Number(item.dataset.ps))))) return;
  const defaultHint = hint.innerHTML;
  const views = pairs.map(({ cell, reference }) => {
    const molmoPS = Number(cell.dataset.ps);
    const astraPS = Number(reference.dataset.ps);
    const difference = Math.sign(molmoPS - astraPS);
    const comparison = document.createElement("span");
    comparison.className = "performance-comparison";
    comparison.hidden = true;
    const descriptions = [];
    [["MolmoAct2", cell.dataset.ps, difference], ["Astra", reference.dataset.ps, -difference]].forEach(([model, value, rank]) => {
      const outcome = rank > 0 ? "higher" : rank < 0 ? "lower" : "equal";
      const row = document.createElement("span");
      row.className = "performance-comparison-row";
      const score = document.createElement("strong");
      score.className = "performance-comparison-score";
      score.dataset.outcome = outcome;
      score.append(value);
      const marker = document.createElement("span");
      marker.className = "performance-comparison-marker";
      marker.setAttribute("aria-hidden", "true");
      marker.textContent = rank > 0 ? "↑" : rank < 0 ? "↓" : "=";
      score.append(marker);
      row.append(score);
      comparison.append(row);
      descriptions.push(`${model} PS ${value}, ${outcome}`);
    });
    cell.append(comparison);
    return { cell, comparison, originalLabel: cell.getAttribute("aria-label"),
      comparisonLabel: `task ${cell.dataset.task} / ${cell.dataset.difficulty}. ${descriptions.join("; ")}. View MolmoAct2 samples.` };
  });
  const modelLabels = Array.from(table.querySelectorAll("tbody th[scope='row']"), (header) => {
    const heading = document.createElement("span");
    heading.className = "performance-task-heading";
    const task = document.createElement("span");
    task.append(...header.childNodes);
    const labels = document.createElement("span");
    labels.className = "performance-comparison-models";
    labels.hidden = true;
    ["MolmoAct2", "Astra"].forEach((model) => {
      const label = document.createElement("span");
      label.textContent = model;
      labels.append(label);
    });
    heading.append(task, labels);
    header.append(heading);
    return labels;
  });
  const update = () => {
    const comparing = toggle.checked;
    table.classList.toggle("is-comparing", comparing);
    modelLabels.forEach((labels) => { labels.hidden = !comparing; });
    views.forEach(({ cell, comparison, originalLabel, comparisonLabel }) => {
      comparison.hidden = !comparing;
      cell.setAttribute("aria-label", comparing ? comparisonLabel : originalLabel);
    });
    hint.innerHTML = comparing
      ? "Models are labeled in the first column. <strong>PS</strong>: green ↑ is higher, red ↓ is lower, and = is tied. Click to watch MolmoAct2 samples."
      : defaultHint;
  };
  toggle.checked = false;
  toggle.addEventListener("change", update);
  update();
  control.hidden = false;
})();

// Tables retain every variant; filters only change the plotted selection.
const experimentRows = (table) => Array.from(table.tBodies)
  .flatMap((body) => Array.from(body.rows))
  .filter((row) => !row.hasAttribute("data-table-group"));
document.querySelectorAll(".experiment-chart[data-table]").forEach((container) => {
  const table = document.getElementById(container.dataset.table);
  if (!table || !table.tBodies.length) return;
  const tableRows = experimentRows(table);
  const parseNumber = (cell, optional = false) => {
    const value = (cell?.dataset.value ?? window.BlogMath.text(cell)).trim();
    if (value === "" || value === "—") return optional ? null : NaN;
    return Number(value.replace(/[,%\s]/g, ""));
  };
  const data = tableRows.map((row, index) => {
    const cells = Array.from(row.cells, (cell) => window.BlogMath.text(cell).trim());
    return {
      index, variant: row.dataset.variant || cells[0], label: row.dataset.label || cells[0],
      sr: parseNumber(row.cells[1]) / 100, ps: parseNumber(row.cells[2]),
      turns: parseNumber(row.cells[3], true), tokens: parseNumber(row.cells[4], true),
      psText: cells[2], turnsText: cells[3] || "—",
      manifest: row.dataset.manifest, rolloutTableId: row.dataset.rolloutTable,
      demo: row.dataset.demo,
      controlMode: row.dataset.controlMode, horizon: Number(row.dataset.horizon),
      executionTime: Number(row.dataset.executionTime || 1),
    };
  });
  if (!data.length || data.some((row) =>
    [row.sr, row.ps].some((value) => !Number.isFinite(value)) ||
    [row.turns, row.tokens].some((value) => value !== null && !Number.isFinite(value)))) return;

  const namespace = "http://www.w3.org/2000/svg";
  const element = (tag, attributes = {}, text = null, parent = null) => {
    const node = document.createElementNS(namespace, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== null) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  };
  const title = container.closest("figure").getAttribute("aria-label");
  const id = table.id;
  const rolloutTableId = container.dataset.rolloutTable || id;
  const relatedVariants = Array.from(new Set(
    Array.from(document.querySelectorAll(".experiment-chart[data-table]"))
      .filter((chart) => (chart.dataset.rolloutTable || chart.dataset.table) === rolloutTableId)
      .flatMap((chart) => experimentRows(document.getElementById(chart.dataset.table))
        .map((row) => row.dataset.variant))
  ));
  const formatter = new Intl.NumberFormat("en-US");
  const tokenLimit = Number(container.dataset.tokensMax);
  const tokensMax = Number.isFinite(tokenLimit) && tokenLimit > 0 ? tokenLimit : 2400000;
  const groupedControl = rolloutTableId === "control-results" && data.every((row) =>
    ["single-target", "multi-target"].includes(row.controlMode) && Number.isFinite(row.horizon));
  const compareExecutionTime = groupedControl && new Set(data.map((row) => row.executionTime)).size > 1;
  const controlModeLines = {
    "single-target": ["Single-target planning", "+ multi-step tracking"],
    "multi-target": ["Multi-target planning", "+ one-step-per-target tracking"],
  };
  const executionTimeLabel = (row) => row.executionTime === 1
    ? "Normal execution time" : `${row.executionTime}× execution time`;
  const labelLines = (row) => {
    if (groupedControl) return [`H=${row.horizon}`];
    if (id === "memory-results") return ["Memory", `${row.variant.split("-")[1]} turns`];
    if (id === "effort-results") return [row.label.replace(/ reasoning effort$/, ""), "reasoning effort"];
    if (id === "collaboration-results") return row.variant === "no-demo" ? ["No", "demo"] : row.label.split(" ");
    return [row.label];
  };

  const filters = document.createElement("fieldset");
  filters.className = "chart-filters";
  const legend = document.createElement("legend");
  legend.textContent = "Show variants";
  const choices = document.createElement("div");
  choices.className = "chart-filter-options";
  choices.classList.toggle("is-grouped", groupedControl);
  const filterModes = new Map(), filterTimings = new Map();
  const filterGroup = (groups, key, title, className, parent) => {
    if (!groups.has(key)) {
      const group = document.createElement("fieldset");
      group.className = className;
      const heading = document.createElement("legend");
      heading.textContent = title;
      const options = document.createElement("div");
      options.className = "chart-filter-group-options";
      group.append(heading, options);
      parent.append(group);
      groups.set(key, options);
    }
    return groups.get(key);
  };
  const inputs = data.map((row) => {
    let options = choices;
    if (groupedControl) {
      options = filterGroup(filterModes, row.controlMode, controlModeLines[row.controlMode].join(" "),
        "chart-filter-mode", choices);
      if (compareExecutionTime) {
        options = filterGroup(filterTimings, `${row.controlMode}-${row.executionTime}`, executionTimeLabel(row),
          "chart-filter-time", options);
      }
    }
    const label = document.createElement("label");
    label.className = "chart-filter-label";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `${id}-filter-${row.variant}`;
    input.dataset.variant = row.variant;
    input.checked = true;
    input.setAttribute("aria-controls", `${id}-plot`);
    const name = document.createElement("span");
    window.BlogMath.setText(name, groupedControl ? `H=${row.horizon}` : row.label);
    label.append(input, name);
    options.append(label);
    input.addEventListener("change", render);
    return input;
  });
  const filterActions = document.createElement("div");
  filterActions.className = "chart-filter-actions";
  [["Select all", true], ["Clear all", false]].forEach(([label, checked]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.dataset.selection = checked ? "all" : "none";
    button.setAttribute("aria-controls", `${id}-plot`);
    button.addEventListener("click", () => {
      inputs.forEach((input) => { input.checked = checked; });
      render();
    });
    filterActions.append(button);
  });
  filters.append(legend, choices, filterActions);

  const plot = document.createElement("div");
  plot.className = "chart-plot";
  const sidebar = document.createElement("div");
  sidebar.className = "chart-sidebar";
  const empty = document.createElement("p");
  empty.className = "chart-empty";
  empty.setAttribute("role", "status");
  empty.textContent = "No variants selected.";
  empty.hidden = true;
  const scroll = document.createElement("div");
  scroll.className = "experiment-chart-scroll";
  scroll.tabIndex = 0;
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", `${title} chart. Scroll horizontally on small screens.`);
  let chartHeight = Math.max(520, 446 + (Math.max(...data.map((row) => labelLines(row).length)) - 1) * 17 + 20);
  const svg = element("svg", {
    id: `${id}-plot`, viewBox: `0 0 600 ${chartHeight}`, role: "group",
    "aria-labelledby": `${id}-chart-title`, "aria-describedby": `${id}-chart-description`,
  });
  element("title", { id: `${id}-chart-title` }, title, svg);
  element("desc", { id: `${id}-chart-description` },
    "Above zero: success rate as solid bars and progress score as a solid line. " +
    "Below zero: episode length (model decision turns per rollout) as diagonally hatched bars and tokens per rollout as a dashed line. " +
    "Costs are positive and increase downward. Scales stay fixed when filtering. " +
    "Use Show variants to select comparisons. Click a variant or press Enter for its rollout videos. " +
    "Hover or focus for exact values, or read the complete table below. " +
    "Missing cost measurements are omitted, not plotted as zero.", svg);
  scroll.append(svg);
  const patterns = element("defs", {}, null, svg);

  const left = 52, right = 548, top = 36, middle = 225, bottom = 414;
  const height = middle - top;
  // Axes stay fixed while filtering; an explicit override can accommodate larger costs.
  const srY = (value) => middle - value * height;
  const turnsY = (value) => middle + value / 200 * height;
  const tokensY = (value) => middle + value / tokensMax * height;
  const text = (px, py, value, className, anchor = "middle", parent = svg) =>
    element("text", { x: px, y: py, class: className, "text-anchor": anchor }, value, parent);
  const line = (x1, y1, x2, y2, className, parent = svg) =>
    element("line", { x1, y1, x2, y2, class: className }, null, parent);

  element("rect", { x: left, y: top, width: right - left, height,
    class: "chart-performance-background" }, null, svg);
  [0.25, 0.5, 0.75, 1].forEach((fraction) => {
    const upper = srY(fraction);
    const lower = middle + fraction * height;
    line(left, upper, right, upper, "chart-grid");
    line(left, lower, right, lower, "chart-grid");
    text(left - 10, upper + 5, `${fraction * 100}%`, "chart-tick", "end");
    text(right + 10, upper + 5, fraction.toFixed(2), "chart-tick", "start");
    text(left - 10, lower + 5, String(fraction * 200), "chart-tick", "end");
    text(right + 10, lower + 5, `${Number((fraction * tokensMax / 1000000).toFixed(2))}M`, "chart-tick", "start");
  });
  line(left, top - 6, left, bottom + 6, "chart-axis");
  line(right, top - 6, right, bottom + 6, "chart-axis chart-axis-secondary");
  text(left - 10, middle + 5, "0", "chart-tick", "end");
  text(right + 10, middle + 5, "0", "chart-tick", "start");

  const bars = element("g", { "aria-hidden": "true" }, null, svg);
  line(left, middle, right, middle, "chart-zero");
  const lines = element("g", { "aria-hidden": "true" }, null, svg);
  const readout = document.createElement("div");
  readout.className = "chart-readout";
  readout.setAttribute("role", "status");
  readout.setAttribute("aria-live", "polite");
  readout.setAttribute("aria-atomic", "true");
  let visible = [...data];
  let selectedIndex = 0;
  const select = (index) => {
    selectedIndex = index;
    controls.forEach((control, i) => {
      control.classList.toggle("is-active", i === index);
      control.setAttribute("aria-pressed", String(i === index));
      tableRows[i].classList.toggle("chart-row-active", i === index);
    });
    readout.replaceChildren();
    if (index === null) return;
    const row = data[index];
    const variant = document.createElement("strong");
    variant.className = "chart-selected-variant";
    if (groupedControl) {
      const mode = document.createElement("span");
      mode.className = "chart-selected-mode";
      mode.textContent = controlModeLines[row.controlMode].join(" ");
      const detail = document.createElement("span");
      detail.className = "chart-selected-detail";
      window.BlogMath.setText(detail, compareExecutionTime
        ? `${executionTimeLabel(row)} · H=${row.horizon}` : `H=${row.horizon}`);
      variant.append(mode, detail);
    } else window.BlogMath.setText(variant, row.label);
    readout.append(variant);
    [
      ["Success rate", `${Math.round(row.sr * 100)}%`, "Upper left axis", "is-bar"],
      ["Progress score", row.psText, "Upper right axis", "is-line"],
      ["Episode length", row.turns === null ? "—" : row.turnsText, "Lower left axis", "is-bar is-hatched"],
      ["Tokens", row.tokens === null ? "—" : formatter.format(row.tokens), "Lower right axis", "is-line is-dashed"],
    ].forEach(([label, value, axisLabel, swatchClass]) => {
      const metric = document.createElement("span");
      metric.className = "chart-metric";
      const swatch = document.createElement("span");
      swatch.className = `chart-legend-swatch ${swatchClass}`;
      swatch.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      const number = document.createElement("strong");
      const axis = document.createElement("small");
      name.className = "chart-metric-name";
      name.textContent = label;
      number.className = "chart-metric-value";
      number.textContent = value;
      axis.className = "chart-metric-axis";
      axis.textContent = axisLabel;
      metric.append(swatch, name, number, axis);
      readout.append(metric);
    });
  };
  // Keep controls connected so an open video panel can restore focus after filtering.
  const controls = data.map((row, index) => {
    const control = element("g", { tabindex: "0", role: "button", class: "chart-variant",
      "aria-label": `${row.label}: SR ${Math.round(row.sr * 100)}%, PS ${row.psText}, ` +
        `episode length ${row.turns === null ? "unavailable" : row.turnsText} and ` +
        `${row.tokens === null ? "unavailable" : formatter.format(row.tokens)} tokens per rollout. View rollout videos.`,
      "aria-controls": "rollout-panel", "aria-expanded": "false",
      "data-variant": row.variant, "data-table": id, "data-filter-control": inputs[index].id,
    }, null, svg);
    control.addEventListener("pointerenter", () => select(index));
    control.addEventListener("focus", () => select(index));
    const openRollouts = () => {
      select(index);
      window.dispatchEvent(new CustomEvent("experiment:open-rollouts", {
        detail: { tableId: row.rolloutTableId || rolloutTableId, variant: row.demo || row.variant,
          manifest: row.manifest, title: row.label, trigger: control },
      }));
    };
    control.addEventListener("click", openRollouts);
    control.addEventListener("keydown", (event) => {
      if (["Enter", " "].includes(event.key)) {
        event.preventDefault();
        openRollouts();
        return;
      }
      const current = visible.indexOf(row);
      let next;
      if (event.key === "ArrowRight") next = (current + 1) % visible.length;
      else if (event.key === "ArrowLeft") next = (current - 1 + visible.length) % visible.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = visible.length - 1;
      else return;
      event.preventDefault();
      const target = controls[visible[next].index];
      target.focus({ preventScroll: true });
      target.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    return control;
  });
  const sharedLabels = element("g", { class: "chart-shared-labels", "aria-hidden": "true" }, null, svg);
  const labelMeasure = document.createElement("canvas").getContext("2d");
  labelMeasure.font = `500 12px ${getComputedStyle(container).fontFamily}`;
  const wrapLabel = (value, width) => {
    const result = [];
    let current = "";
    value.split(" ").forEach((word) => {
      const candidate = current ? `${current} ${word}` : word;
      if (current && labelMeasure.measureText(candidate).width > width) {
        result.push(current);
        current = word;
      } else current = candidate;
    });
    if (current) result.push(current);
    return result;
  };
  function renderControlGroups(band) {
    sharedLabels.replaceChildren();
    if (!groupedControl) return;
    const modes = new Map(), timings = new Map();
    visible.forEach((row, index) => {
      [[modes, row.controlMode], [timings, `${row.controlMode}-${row.executionTime}`]].forEach(([groups, key]) => {
        const group = groups.get(key) || { start: index, end: index, row };
        group.end = index;
        groups.set(key, group);
      });
    });
    const drawGroup = (group, values, y, className) => {
      const start = left + group.start * band + 8;
      const end = left + (group.end + 1) * band - 8;
      const center = (start + end) / 2;
      const node = element("g", { class: className, "data-mode": group.row.controlMode,
        "data-time": group.row.executionTime, "data-left": start, "data-right": end }, null, sharedLabels);
      line(start, y, end, y, "chart-group-bracket", node);
      line(start, y - 3, start, y, "chart-group-bracket", node);
      line(end, y - 3, end, y, "chart-group-bracket", node);
      const labels = values.flatMap((value) => wrapLabel(value, end - start - 8));
      labels.forEach((value, index) => text(center, y + 20 + index * 17, value, "chart-mode-label", "middle", node));
      return y + 20 + (labels.length - 1) * 17;
    };
    let modeY = 460;
    if (compareExecutionTime && visible.length) {
      const bottom = Math.max(...Array.from(timings.values(), (group) => drawGroup(group,
        [executionTimeLabel(group.row)],
        460, "chart-time-group")));
      modeY = bottom + 18;
    }
    const bottoms = Array.from(modes.values(), (group) =>
      drawGroup(group, controlModeLines[group.row.controlMode], modeY, "chart-mode-group"));
    chartHeight = Math.max(540, ...bottoms.map((value) => value + 24));
    svg.setAttribute("viewBox", `0 0 600 ${chartHeight}`);
  }
  function render() {
    visible = data.filter((row) => inputs[row.index].checked);
    const hasSelection = visible.length > 0;
    empty.hidden = hasSelection;
    scroll.hidden = !hasSelection;
    readout.hidden = !hasSelection;
    const band = (right - left) / Math.max(1, visible.length);
    const barWidth = Math.min(28, band * 0.4);
    const x = (index) => left + band * (index + 0.5);
    renderControlGroups(band);
    bars.replaceChildren();
    lines.replaceChildren();
    patterns.replaceChildren();
    visible.forEach((row, index) => {
      const shade = Math.round(230 - relatedVariants.indexOf(row.variant) / Math.max(1, relatedVariants.length - 1) * 108);
      const fill = `rgb(${shade}, ${shade}, ${shade})`;
      element("rect", { x: x(index) - barWidth / 2, y: srY(row.sr),
        width: barWidth, height: middle - srY(row.sr), fill,
        class: "chart-bar", "data-series": "sr", "data-value": row.sr, "data-variant": row.variant }, null, bars);
      if (row.turns !== null) {
        const patternId = `${id}-episode-hatch-${row.index}`;
        const pattern = element("pattern", { id: patternId, width: 6, height: 6,
          patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, null, patterns);
        element("rect", { width: 6, height: 6, fill, "fill-opacity": 0.12 }, null, pattern);
        element("path", { d: "M 3 0 V 6", fill: "none", stroke: fill, "stroke-width": 1.2 }, null, pattern);
        element("rect", { x: x(index) - barWidth / 2, y: middle,
          width: barWidth, height: turnsY(row.turns) - middle, fill: `url(#${patternId})`,
          class: "chart-bar chart-bar-cost", "data-series": "turns", "data-value": row.turns, "data-variant": row.variant }, null, bars);
      }
    });
    if (hasSelection) {
      [
        { key: "ps", y: srY, className: "chart-line" },
        { key: "tokens", y: tokensY, className: "chart-line chart-line-cost" },
      ].forEach((series) => {
        let connected = false;
        const commands = [];
        visible.forEach((row, index) => {
          if (row[series.key] === null) {
            connected = false;
            return;
          }
          commands.push(`${connected ? "L" : "M"} ${x(index)} ${series.y(row[series.key])}`);
          connected = true;
        });
        if (!commands.length) return;
        const d = commands.join(" ");
        element("path", { d, class: `${series.className} chart-line-underlay` }, null, lines);
        element("path", { d, class: series.className, "data-series": series.key }, null, lines);
        visible.forEach((row, index) => {
          if (row[series.key] === null) return;
          element("circle", {
            cx: x(index), cy: series.y(row[series.key]), r: 3,
            class: "chart-dot", "data-series": series.key, "data-value": row[series.key], "data-variant": row.variant,
          }, null, lines);
        });
      });
    }
    controls.forEach((control, originalIndex) => {
      const index = visible.indexOf(data[originalIndex]);
      control.style.display = index < 0 ? "none" : "";
      control.setAttribute("aria-hidden", String(index < 0));
      control.setAttribute("tabindex", index < 0 ? "-1" : "0");
      control.replaceChildren();
      if (index < 0) return;
      element("rect", { x: left + index * band + 3, y: top - 8,
        width: band - 6, height: groupedControl ? 460 - top : chartHeight - top - 8,
        rx: 5, class: "chart-hit" }, null, control);
      if (groupedControl) {
        // HTML inside SVG lets the axis use the same KaTeX as article formulas.
        const box = element("foreignObject", {
          x: x(index) - band / 2, y: 428, width: band, height: 30,
          class: "chart-horizon-label", "aria-hidden": "true", "pointer-events": "none",
        }, null, control);
        const label = document.createElement("div");
        label.className = "chart-horizon-content";
        window.BlogMath.setText(label, `H=${data[originalIndex].horizon}`);
        box.append(label);
      } else {
        labelLines(data[originalIndex]).forEach((label, lineIndex) =>
          text(x(index), 446 + lineIndex * 17, label, "chart-variant-label", "middle", control));
      }
    });
    select(visible.some((row) => row.index === selectedIndex) ? selectedIndex : visible[0]?.index ?? null);
  }

  plot.append(empty, scroll);
  sidebar.append(filters, readout);
  container.replaceChildren(plot, sidebar);
  render();
});
