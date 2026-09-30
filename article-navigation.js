"use strict";

(() => {
  const nav = document.getElementById("article-nav");
  const contents = document.getElementById("article-contents");
  const toggle = document.getElementById("contents-toggle");
  const current = document.getElementById("reading-section");
  if (!nav || !contents || !toggle || !current) return;
  const menu = contents.querySelector(".contents-menu");

  const entries = Array.from(contents.querySelectorAll('a[href^="#"]'))
    .map((link) => ({ link, target: document.getElementById(link.hash.slice(1)) }))
    .filter(({ target }) => target);
  if (!entries.length) return;

  let pendingFrame = null;
  let activeEntry = null;
  entries.forEach(({ target }) => {
    target.classList.add("article-destination");
    if (!target.hasAttribute("tabindex")) target.tabIndex = -1;
  });
  const pageTitle = document.getElementById("blog-title");
  if (pageTitle) {
    pageTitle.classList.add("article-destination");
    pageTitle.tabIndex = -1;
  }

  function positionContents() {
    if (!contents.open || !menu) return;
    const rect = nav.getBoundingClientRect();
    const below = window.innerHeight - rect.bottom - 18;
    const above = rect.top - 18;
    const opensAbove = below < 140 && above > below;
    menu.classList.toggle("is-above", opensAbove);
    menu.style.setProperty("--contents-available-height", `${Math.max(48, opensAbove ? above : below)}px`);
  }

  function updateCurrent() {
    pendingFrame = null;
    positionContents();
    const threshold = nav.getBoundingClientRect().bottom + 40;
    let selected = entries[0];
    entries.forEach((entry) => {
      if (entry.target.getBoundingClientRect().top <= threshold) selected = entry;
    });
    if (selected === activeEntry) return;
    activeEntry = selected;
    current.textContent = selected.link.textContent;
    entries.forEach(({ link }) => {
      if (link === selected.link) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }

  function scheduleUpdate() {
    if (pendingFrame === null) pendingFrame = requestAnimationFrame(updateCurrent);
  }

  function closeContents(restoreFocus = false) {
    if (!contents.open) return;
    contents.open = false;
    if (restoreFocus) toggle.focus({ preventScroll: true });
  }

  nav.addEventListener("click", (event) => {
    const link = event.target.closest('a[href^="#"]');
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    closeContents();
    window.dispatchEvent(new CustomEvent("article:navigate"));
    // Keep native hash navigation and browser history, then focus the destination.
    const target = document.getElementById(link.hash.slice(1));
    if (target) requestAnimationFrame(() => {
      target.focus({ preventScroll: true });
      updateCurrent();
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !contents.open) return;
    event.preventDefault();
    closeContents(true);
  });
  document.addEventListener("pointerdown", (event) => {
    if (!contents.contains(event.target)) closeContents();
  });
  contents.addEventListener("focusout", (event) => {
    if (event.relatedTarget && !contents.contains(event.relatedTarget)) closeContents();
  });
  contents.addEventListener("toggle", positionContents);
  window.addEventListener("article:open-concept", () => closeContents());
  window.addEventListener("experiment:open-rollouts", () => closeContents());
  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate);
  window.addEventListener("hashchange", scheduleUpdate);
  window.addEventListener("load", scheduleUpdate);
  updateCurrent();
})();
