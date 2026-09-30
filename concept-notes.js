"use strict";

(() => {
  const article = document.getElementById("article");
  const note = document.getElementById("concept-note");
  const title = document.getElementById("concept-note-title");
  const body = document.getElementById("concept-note-body");
  const closeButton = document.getElementById("concept-note-close");
  const terms = Array.from(document.querySelectorAll(".concept-term"));
  if (!article || !note || !title || !body || !closeButton) return;

  let activeTerm = null;
  let positionFrame = null;

  function positionNote() {
    if (note.hidden || !activeTerm) return;
    const viewportWidth = document.documentElement.clientWidth;
    const viewportHeight = window.innerHeight;
    const availableWidth = article.getBoundingClientRect().left - 36;
    const compact = availableWidth < 220;
    note.classList.toggle("is-compact", compact);

    if (compact) {
      note.style.width = `${Math.min(340, viewportWidth - 24)}px`;
      note.style.left = "12px";
      note.style.top = "auto";
      note.style.bottom = "12px";
      return;
    }

    const width = Math.min(280, availableWidth);
    note.style.width = `${width}px`;
    note.style.left = `${article.getBoundingClientRect().left - width - 20}px`;
    note.style.bottom = "auto";
    const topLimit = Math.max(16, viewportHeight - note.getBoundingClientRect().height - 16);
    note.style.top = `${Math.max(16, Math.min(activeTerm.getBoundingClientRect().top - 12, topLimit))}px`;
  }

  function schedulePosition() {
    if (note.hidden || positionFrame !== null) return;
    positionFrame = requestAnimationFrame(() => {
      positionFrame = null;
      positionNote();
    });
  }

  function closeNote({ restoreFocus = true } = {}) {
    if (note.hidden) return;
    const trigger = activeTerm;
    note.hidden = true;
    activeTerm = null;
    terms.forEach((term) => {
      term.classList.remove("is-open");
      term.setAttribute("aria-expanded", "false");
    });
    if (restoreFocus && trigger?.isConnected) trigger.focus({ preventScroll: true });
  }

  terms.forEach((term) => {
    term.addEventListener("click", () => {
      if (!note.hidden && activeTerm === term) {
        closeNote();
        return;
      }
      const template = document.getElementById(`concept-${term.dataset.concept}`);
      if (!(template instanceof HTMLTemplateElement)) return;
      window.dispatchEvent(new CustomEvent("article:open-concept", {
        detail: { concept: term.dataset.concept, trigger: term },
      }));
      activeTerm = term;
      title.textContent = template.dataset.title || term.textContent;
      body.replaceChildren(template.content.cloneNode(true));
      body.scrollTop = 0;
      terms.forEach((item) => {
        const selected = item === term;
        item.classList.toggle("is-open", selected);
        item.setAttribute("aria-expanded", String(selected));
      });
      note.hidden = false;
      positionNote();
      closeButton.focus({ preventScroll: true });
    });
  });

  closeButton.addEventListener("click", () => closeNote());
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || note.hidden) return;
    event.preventDefault();
    closeNote();
  });
  document.addEventListener("pointerdown", (event) => {
    if (note.hidden || note.contains(event.target) || event.target.closest(".concept-term")) return;
    closeNote({ restoreFocus: false });
  });
  window.addEventListener("experiment:open-rollouts", () => closeNote({ restoreFocus: false }));
  window.addEventListener("article:navigate", () => closeNote({ restoreFocus: false }));
  document.querySelectorAll("details").forEach((disclosure) => {
    disclosure.addEventListener("toggle", () => {
      if (!disclosure.open && activeTerm && disclosure.contains(activeTerm)) closeNote({ restoreFocus: false });
    });
  });
  window.addEventListener("scroll", schedulePosition, { passive: true });
  window.addEventListener("resize", schedulePosition);
})();
