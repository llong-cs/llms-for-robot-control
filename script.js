"use strict";

// Capture play events so dynamically loaded rollout videos share this rule.
document.addEventListener("play", (event) => {
  if (!(event.target instanceof HTMLVideoElement)) return;
  document.querySelectorAll("video").forEach((other) => {
    if (other !== event.target) other.pause();
  });
}, true);

document.querySelectorAll("#article details").forEach((disclosure) => {
  disclosure.addEventListener("toggle", () => {
    if (!disclosure.open) disclosure.querySelectorAll("video").forEach((video) => video.pause());
  });
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
  const title = figure.querySelector("h4")?.textContent || video.getAttribute("aria-label");
  const context = figure.querySelector(".video-caption")?.textContent.replace(/\s*·?\s*(?:Open|Enlarge) video\s*$/, "").trim() || "";
  const description = figure.querySelector("figcaption p")?.textContent || "Recorded demonstration.";
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
        manifest: link.dataset.manifest || "promising-behaviors.json", title: link.textContent.trim(),
      },
    }));
  });
});

document.querySelectorAll(".performance-cell").forEach((cell) => {
  cell.addEventListener("click", () => {
    const difficulty = { easy: "Easy", medium: "Medium", hard: "Hard", xhard: "Extra hard" }[cell.dataset.difficulty];
    const sr = cell.querySelector(".performance-sr").textContent;
    const tableId = cell.dataset.table || "main-results";
    const model = cell.dataset.model || "Astra";
    window.dispatchEvent(new CustomEvent("experiment:open-rollouts", {
      detail: {
        tableId, variant: cell.dataset.variant, trigger: cell,
        manifest: cell.dataset.manifest || "main-samples.json", title: `${model} · Task ${cell.dataset.task} · ${difficulty}`,
        summary: `${model} aggregate results · SR ${sr} · PS ${cell.dataset.ps} · ${cell.dataset.successes}/${cell.dataset.total} successes.`,
      },
    }));
  });
});

// Tables retain every variant; filters only change the plotted selection.
document.querySelectorAll(".experiment-chart[data-table]").forEach((container) => {
  const table = document.getElementById(container.dataset.table);
  if (!table || !table.tBodies.length) return;
  const tableRows = Array.from(table.tBodies[0].rows);
  const parseNumber = (cell, optional = false) => {
    const value = (cell?.dataset.value ?? cell?.textContent ?? "").trim();
    if (value === "" || value === "—") return optional ? null : NaN;
    return Number(value.replace(/[,%\s]/g, ""));
  };
  const data = tableRows.map((row, index) => {
    const cells = Array.from(row.cells, (cell) => cell.textContent.trim());
    return {
      index, variant: row.dataset.variant || cells[0], label: cells[0],
      sr: parseNumber(row.cells[1]) / 100, ps: parseNumber(row.cells[2]),
      turns: parseNumber(row.cells[3], true), tokens: parseNumber(row.cells[4], true),
      psText: cells[2], turnsText: cells[3] || "—",
      manifest: row.dataset.manifest, rolloutTableId: row.dataset.rolloutTable,
      demo: row.dataset.demo,
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
      .flatMap((chart) => Array.from(document.getElementById(chart.dataset.table).tBodies[0].rows,
        (row) => row.dataset.variant))
  ));
  const formatter = new Intl.NumberFormat("en-US");
  const tokenLimit = Number(container.dataset.tokensMax);
  const tokensMax = Number.isFinite(tokenLimit) && tokenLimit > 0 ? tokenLimit : 2400000;
  const labelLines = (row) => {
    if (rolloutTableId === "control-results") {
      const horizon = row.variant.match(/(?:move-by-|chunk-)(\d+)/)?.[1];
      const mode = row.variant.includes("chunk") ? "multi targets" : "single target";
      return [`H=${horizon}`, ...(data.length > 4 ? mode.split(" ") : [mode]),
        ...(row.variant.endsWith("2xslow") ? ["2× time"] : [])];
    }
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
  const inputs = data.map((row) => {
    const label = document.createElement("label");
    label.className = "chart-filter-label";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `${id}-filter-${row.variant}`;
    input.dataset.variant = row.variant;
    input.checked = true;
    input.setAttribute("aria-controls", `${id}-plot`);
    const name = document.createElement("span");
    name.textContent = row.label;
    label.append(input, name);
    choices.append(label);
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
  const svg = element("svg", {
    id: `${id}-plot`, viewBox: "0 0 600 520", role: "group",
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
    variant.textContent = row.label;
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
  function render() {
    visible = data.filter((row) => inputs[row.index].checked);
    const hasSelection = visible.length > 0;
    empty.hidden = hasSelection;
    scroll.hidden = !hasSelection;
    readout.hidden = !hasSelection;
    const band = (right - left) / Math.max(1, visible.length);
    const barWidth = Math.min(28, band * 0.4);
    const x = (index) => left + band * (index + 0.5);
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
        width: band - 6, height: bottom - top + 92, rx: 5, class: "chart-hit" }, null, control);
      labelLines(data[originalIndex]).forEach((label, lineIndex) =>
        text(x(index), 446 + lineIndex * 17, label, "chart-variant-label", "middle", control));
    });
    select(visible.some((row) => row.index === selectedIndex) ? selectedIndex : visible[0]?.index ?? null);
  }

  plot.append(empty, scroll);
  sidebar.append(filters, readout);
  container.replaceChildren(plot, sidebar);
  render();
});
