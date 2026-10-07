"use strict";

(() => {
  const article = document.getElementById("article");
  if (!article) return;
  const terms = Array.from(article.querySelectorAll(".concept-term"));
  const notes = new Map();
  const slots = new Map();
  let positionFrame = null;

  // Keep open notes outside disclosures so collapsing a section does not hide them.
  function blockAfter(term) {
    let block = term.closest("p") || term.parentElement;
    for (let parent = term.parentElement; parent && parent !== article; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) block = parent;
    }
    return block;
  }

  function slotFor(term) {
    const anchor = blockAfter(term);
    if (!slots.has(anchor)) {
      const slot = document.createElement("div");
      slot.className = "concept-note-slot";
      slot.hidden = true;
      anchor.after(slot);
      slots.set(anchor, slot);
    }
    return slots.get(anchor);
  }

  function syncVisibility() {
    slots.forEach((slot) => {
      slot.hidden = !Array.from(slot.children).some((note) => !note.hidden);
    });
    terms.forEach((term) => {
      const entry = notes.get(term.dataset.concept);
      if (!entry) return;
      term.setAttribute("aria-expanded", String(!entry.note.hidden));
      term.classList.toggle("is-active", !entry.note.hidden);
    });
    positionNotes();
  }

  function closeNote(key) {
    const entry = notes.get(key);
    entry.note.hidden = true;
    entry.note.classList.remove("is-active");
    syncVisibility();
    visibleAnchor(entry.term).focus({ preventScroll: true });
  }

  terms.forEach((term) => {
    const key = term.dataset.concept;
    const template = document.getElementById(`concept-${key}`);
    if (!(template instanceof HTMLTemplateElement)) return;
    if (!notes.has(key)) {
      const note = document.createElement("aside");
      note.id = `concept-note-${key}`;
      note.className = "concept-note";
      note.hidden = true;
      note.tabIndex = -1;
      note.setAttribute("aria-labelledby", `${note.id}-title`);
      const title = document.createElement("h3");
      title.id = `${note.id}-title`;
      title.className = "concept-note-title";
      const titleText = template.dataset.title || window.BlogMath.text(term);
      window.BlogMath.setText(title, titleText);
      const header = document.createElement("div");
      header.className = "concept-note-header";
      const close = document.createElement("button");
      close.type = "button";
      close.className = "concept-note-close";
      close.setAttribute("aria-label", `Close note: ${titleText}`);
      close.textContent = "×";
      close.addEventListener("click", () => closeNote(key));
      header.append(title, close);
      const body = document.createElement("div");
      body.className = "concept-note-body";
      body.append(template.content.cloneNode(true));
      note.append(header, body);
      slotFor(term).append(note);
      notes.set(key, { note, term });
    }
    term.setAttribute("aria-controls", notes.get(key).note.id);
    term.setAttribute("aria-expanded", "false");
    term.removeAttribute("aria-haspopup");
    term.addEventListener("click", () => {
      const entry = notes.get(key);
      const { note } = entry;
      if (!note.hidden) {
        closeNote(key);
        term.focus({ preventScroll: true });
        return;
      }
      window.dispatchEvent(new CustomEvent("article:open-concept", {
        detail: { concept: key, trigger: term },
      }));
      entry.term = term;
      slotFor(term).append(note);
      note.hidden = false;
      notes.forEach((entry) => entry.note.classList.toggle("is-active", entry.note === note));
      syncVisibility();
      // Preserve the reading position even when a long note extends offscreen.
      note.focus({ preventScroll: true });
    });
  });

  function visibleAnchor(term) {
    let anchor = term;
    for (let parent = term.parentElement; parent && parent !== article; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement && !parent.open) {
        anchor = parent.querySelector("summary") || parent;
      }
    }
    return anchor;
  }

  function positionNotes() {
    const articleBounds = article.getBoundingClientRect();
    const availableWidth = articleBounds.left - 36;
    const marginLayout = availableWidth >= 220;
    article.classList.toggle("has-margin-notes", marginLayout);
    const width = Math.min(280, availableWidth);
    let bottom = 0;
    const openNotes = Array.from(notes.values()).filter(({ note }) => !note.hidden);
    openNotes.sort((a, b) => a.term.compareDocumentPosition(b.term) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);
    openNotes.forEach(({ note, term }) => {
      if (!marginLayout) {
        note.style.removeProperty("width");
        note.style.removeProperty("left");
        note.style.removeProperty("top");
        return;
      }
      note.style.width = `${width}px`;
      note.style.left = `${-width - 20}px`;
      const anchorTop = visibleAnchor(term).getBoundingClientRect().top - article.getBoundingClientRect().top;
      const top = Math.max(anchorTop - 10, bottom);
      note.style.top = `${top}px`;
      bottom = top + note.getBoundingClientRect().height + 18;
    });
  }

  function schedulePosition() {
    if (positionFrame !== null) return;
    positionFrame = requestAnimationFrame(() => {
      positionFrame = null;
      positionNotes();
    });
  }

  window.addEventListener("resize", schedulePosition);
  article.addEventListener("load", schedulePosition, true);
  article.querySelectorAll("details").forEach((details) => {
    details.addEventListener("toggle", schedulePosition);
  });
  const observer = new ResizeObserver(schedulePosition);
  observer.observe(article);
  notes.forEach(({ note }) => observer.observe(note));
  document.fonts.ready.then(schedulePosition);
  positionNotes();
})();
