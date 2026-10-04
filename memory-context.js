(() => {
  "use strict";

  const root = document.getElementById("memory-context-example");
  if (!root) return;
  const controls = root.querySelector(".memory-context-controls");
  const buttons = [...controls.querySelectorAll("[data-memory-length]")];
  const status = root.querySelector(".memory-context-status");
  const retry = root.querySelector(".memory-context-retry");
  const content = root.querySelector("#memory-context-content");
  const timeline = root.querySelector(".memory-context-timeline");
  const detail = root.querySelector(".memory-context-detail");
  let data;
  let variant;
  let loading = false;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function image(source, turn, thumbnail = false) {
    const node = element("img");
    node.src = source.src;
    node.width = source.width;
    node.height = source.height;
    node.alt = thumbnail ? "" : `Turn ${turn}: ${source.name}`;
    node.decoding = "async";
    node.loading = "lazy";
    return node;
  }

  function recordedDetails(label, value) {
    const block = element("details");
    block.append(element("summary", "", label), element("pre", "", JSON.stringify(value, null, 2)));
    return block;
  }

  function selectTurn(turnNumber) {
    const turn = variant.turns.find((entry) => entry.turn === turnNumber);
    if (!turn) return;
    timeline.querySelectorAll("button").forEach((button) => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.turn) === turnNumber));
    });
    const observation = element("div", "memory-context-observation");
    observation.append(element("h4", "", `Turn ${turn.turn} · ${turn.current ? "Current observation" : "Past observation"}`));
    const cameras = element("div", "memory-context-cameras");
    turn.images.forEach((source) => {
      const figure = element("figure");
      const link = element("a");
      link.href = source.src;
      link.dataset.mediaViewer = "image";
      link.dataset.mediaTitle = `Memory ${variant.memory} · turn ${turn.turn} · ${source.name}`;
      link.setAttribute("aria-label", `Enlarge ${source.name} from turn ${turn.turn}`);
      link.setAttribute("aria-haspopup", "dialog");
      link.setAttribute("aria-controls", "media-viewer");
      link.append(image(source, turn.turn));
      const cameraName = source.name.includes("wrist") ? "Wrist camera" : "External camera";
      figure.append(link, element("figcaption", "", cameraName));
      cameras.append(figure);
    });
    observation.append(cameras, recordedDetails("Robot state", turn.state));

    const actions = element("div", "memory-context-actions");
    actions.append(element("h4", "", turn.current ? "Awaiting Astra's next action" : `Astra · turn ${turn.turn}`));
    if (turn.current) {
      actions.append(element("p", "memory-context-awaiting", "This input ends with the current observation. Turn 12's action has not been generated yet."));
    } else {
      const output = turn.toolCalls.map((call) => `${call.name}(${JSON.stringify(call.arguments, null, 2)})`).join("\n\n");
      const pre = element("pre", "memory-context-output", output);
      pre.tabIndex = 0;
      pre.setAttribute("aria-label", `Recorded action tool calls and notes from turn ${turn.turn}`);
      actions.append(pre);
      if (turn.toolResults?.length) actions.append(recordedDetails("Tool response", turn.toolResults.map((result) => result.output)));
    }
    detail.replaceChildren(observation, actions);
  }

  function centerSelectedTurn() {
    const selected = timeline.querySelector('[aria-pressed="true"]');
    if (selected && root.open) {
      // Scroll only the horizontal strip, never the article or its focus.
      timeline.scrollLeft = selected.offsetLeft - timeline.clientWidth / 2 + selected.offsetWidth / 2;
    }
  }

  function selectMemory(length) {
    variant = data.variants.find((entry) => entry.memory === length);
    if (!variant) return;
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(Number(button.dataset.memoryLength) === length)));
    status.textContent = length ? `${length} past turns + current observation` : "Current observation only";
    const turns = variant.turns.map((turn) => {
      const button = element("button", `memory-context-turn${turn.current ? " is-current" : ""}`);
      button.type = "button";
      button.dataset.turn = turn.turn;
      button.setAttribute("aria-controls", "memory-context-detail");
      button.setAttribute("aria-label", `Turn ${turn.turn}, ${turn.current ? "current observation" : "past observation and action"}`);
      button.append(image(turn.images[0], turn.turn, true));
      button.append(element("strong", "", `Turn ${turn.turn}`));
      button.append(element("span", "", turn.current ? "Current" : "Obs + action"));
      button.addEventListener("click", () => selectTurn(turn.turn));
      return button;
    });
    timeline.replaceChildren(...turns);
    detail.id = "memory-context-detail";
    // Start with the latest past exchange so the recorded action and note are visible.
    const initial = variant.turns.findLast((turn) => !turn.current) || variant.turns[0];
    selectTurn(initial.turn);
    centerSelectedTurn();
  }

  async function load() {
    if (data || loading) return;
    loading = true;
    retry.hidden = true;
    status.textContent = "Loading recorded inputs…";
    try {
      const response = await fetch(root.dataset.contextSrc);
      if (!response.ok) throw new Error("Recorded inputs are unavailable");
      const result = await response.json();
      if (![0, 2, 10].every((length) => result.variants?.some((entry) => entry.memory === length && entry.turns?.length === length + 1))) {
        throw new Error("Incomplete recorded inputs");
      }
      data = result;
      content.hidden = false;
      controls.hidden = false;
      selectMemory(2);
    } catch (_) {
      data = null;
      content.hidden = true;
      controls.hidden = true;
      status.textContent = "The recorded inputs could not be loaded.";
      retry.hidden = false;
    } finally {
      loading = false;
    }
  }

  buttons.forEach((button) => button.addEventListener("click", () => selectMemory(Number(button.dataset.memoryLength))));
  retry.addEventListener("click", load);
  root.addEventListener("toggle", () => {
    if (root.open) {
      load();
      requestAnimationFrame(centerSelectedTurn);
    }
  });
  if (root.open) load();
})();
