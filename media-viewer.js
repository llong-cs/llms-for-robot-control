"use strict";

(() => {
  const dialog = document.getElementById("media-viewer");
  const title = document.getElementById("media-viewer-title");
  const content = document.getElementById("media-viewer-content");
  const closeButton = document.getElementById("media-viewer-close");
  const zoomButton = document.getElementById("media-viewer-zoom");
  const error = document.getElementById("media-viewer-error");
  if (!dialog || !title || !content || !closeButton || !error) return;

  let active = null;
  let backdropPressed = false;
  // Native dialogs use the viewport; account for the page's stable scrollbar gutter.
  const syncViewport = () => dialog.style.setProperty("--media-viewport-width", `${document.documentElement.getBoundingClientRect().width}px`);
  const playback = (video) => ({ time: video.currentTime, playing: !video.paused && !video.ended });
  const restorePlayback = (video, state) => {
    // Reparenting can pause a media element. Keep its existing decoder and timeline.
    if (video.readyState >= 1 && Math.abs(video.currentTime - state.time) > 0.1) {
      try { video.currentTime = state.time; } catch (_) { /* An unavailable seek range is harmless. */ }
    }
    if (state.playing && video.paused) video.play().catch(() => {});
    else if (!state.playing && !video.paused) video.pause();
  };
  const setZoom = (zoomed) => {
    const image = content.querySelector("img");
    if (image) {
      if (zoomed && image.naturalWidth) {
        const aspect = image.naturalWidth / image.naturalHeight;
        const fittedWidth = Math.min(image.clientWidth, image.clientHeight * aspect);
        image.style.width = `${Math.ceil(Math.max(image.naturalWidth, fittedWidth * 2))}px`;
      } else {
        image.style.removeProperty("width");
      }
    }
    content.classList.toggle("is-zoomed", zoomed);
    if (zoomButton) {
      zoomButton.textContent = zoomed ? "Fit to window" : "Zoom in";
      zoomButton.setAttribute("aria-pressed", String(zoomed));
    }
    content.scrollTo(0, 0);
  };

  function close({ restoreFocus = true, restoreScroll = true } = {}) {
    const view = active;
    if (!view) {
      if (dialog.open) dialog.close();
      return;
    }
    active = null;
    view.listeners.abort();
    if (view.video) {
      const state = playback(view.video);
      if (!view.hadViewerClass) view.video.classList.remove("media-viewer-video");
      if (view.placeholder?.isConnected) {
        view.placeholder.replaceWith(view.video);
        restorePlayback(view.video, state);
      } else if (view.parent?.isConnected) {
        const next = view.nextSibling?.parentNode === view.parent ? view.nextSibling : null;
        view.parent.insertBefore(view.video, next);
        restorePlayback(view.video, state);
      } else {
        view.video.pause();
        if (!view.parent) {
          view.video.removeAttribute("src");
          view.video.load();
        }
      }
    }
    content.replaceChildren();
    setZoom(false);
    if (zoomButton) zoomButton.hidden = true;
    error.hidden = true;
    document.body.classList.remove("media-viewer-open");
    if (dialog.open) dialog.close();
    if (restoreScroll) window.scrollTo({ left: view.scrollX, top: view.scrollY, behavior: "instant" });
    if (restoreFocus && view.trigger.isConnected) view.trigger.focus({ preventScroll: true });
  }

  function open(link) {
    const kind = link.dataset.mediaViewer;
    if (!["image", "video"].includes(kind) || !link.getAttribute("href")) return;
    close({ restoreFocus: false });
    const sourceFigure = link.closest("figure");
    const sourceVideo = kind === "video" ? sourceFigure?.querySelector("video") : null;
    const sourceError = sourceFigure?.querySelector(".media-error, .rollout-video-error");
    const thumbnail = link.querySelector("img");
    const view = {
      trigger: link, kind, listeners: new AbortController(),
      scrollX: window.scrollX, scrollY: window.scrollY,
    };
    active = view;
    const titleText = link.dataset.mediaTitle || window.BlogMath.text(sourceFigure?.querySelector("#rollout-current-title, figcaption strong")) || sourceVideo?.getAttribute("aria-label") ||
      thumbnail?.alt || link.getAttribute("aria-label") || (kind === "image" ? "Image" : "Video");
    window.BlogMath.setText(title, titleText);
    error.hidden = true;
    error.textContent = `The ${kind} could not be loaded. Close this viewer and try again.`;
    setZoom(false);
    if (zoomButton) {
      zoomButton.hidden = kind !== "image";
      zoomButton.disabled = kind === "image";
    }
    document.body.classList.add("media-viewer-open");
    syncViewport();
    try {
      dialog.showModal();
    } catch (_) {
      close();
      return;
    }
    closeButton.focus({ preventScroll: true });
    const listenerOptions = { signal: view.listeners.signal };
    const showError = () => {
      if (active !== view) return;
      error.hidden = false;
      if (zoomButton) zoomButton.disabled = true;
    };

    if (kind === "image") {
      const image = document.createElement("img");
      image.className = "media-viewer-image";
      image.alt = thumbnail?.alt || titleText;
      image.decoding = "async";
      image.addEventListener("load", () => {
        if (active !== view) return;
        error.hidden = true;
        if (zoomButton) zoomButton.disabled = false;
      }, listenerOptions);
      image.addEventListener("error", showError, listenerOptions);
      image.addEventListener("click", () => {
        if (image.naturalWidth) setZoom(!content.classList.contains("is-zoomed"));
      }, listenerOptions);
      content.append(image);
      image.src = link.href;
      return;
    }

    const video = sourceVideo || document.createElement("video");
    view.video = video;
    view.hadViewerClass = video.classList.contains("media-viewer-video");
    const state = playback(video);
    if (sourceVideo) {
      view.parent = video.parentNode;
      view.nextSibling = video.nextSibling;
      const bounds = video.getBoundingClientRect();
      const style = getComputedStyle(video);
      const placeholder = document.createElement("div");
      placeholder.className = "media-viewer-placeholder";
      placeholder.setAttribute("aria-hidden", "true");
      Object.assign(placeholder.style, {
        boxSizing: "border-box", width: `${bounds.width}px`, height: `${bounds.height}px`,
        maxWidth: "100%", margin: style.margin, display: style.display === "inline" ? "inline-block" : style.display,
      });
      video.before(placeholder);
      view.placeholder = placeholder;
    } else {
      video.controls = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.setAttribute("aria-label", titleText);
      video.src = link.href;
    }
    video.addEventListener("error", showError, listenerOptions);
    video.addEventListener("loadeddata", () => {
      if (active === view) error.hidden = true;
    }, listenerOptions);
    const sources = Array.from(video.querySelectorAll("source"));
    const failedSources = new Set();
    sources.forEach((source) => source.addEventListener("error", () => {
      failedSources.add(source);
      if (failedSources.size === sources.length && video.readyState === 0) showError();
    }, listenerOptions));
    video.classList.add("media-viewer-video");
    content.append(video);
    restorePlayback(video, state);
    if (video.error || (video.readyState === 0 && sourceError && !sourceError.hidden)) showError();
  }

  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target instanceof Element ? event.target.closest("a[data-media-viewer]") : null;
    if (!link || link.classList.contains("demo-link")) return;
    event.preventDefault();
    open(link);
  });
  closeButton.addEventListener("click", () => close());
  zoomButton?.addEventListener("click", () => {
    if (active?.kind === "image" && !zoomButton.disabled) setZoom(!content.classList.contains("is-zoomed"));
  });
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    event.stopPropagation();
    close();
  });
  dialog.addEventListener("close", () => {
    if (!dialog.open) close();
  });
  const outside = (event) => {
    const bounds = dialog.getBoundingClientRect();
    return event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom;
  };
  dialog.addEventListener("pointerdown", (event) => { backdropPressed = outside(event); });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog && backdropPressed && outside(event)) close();
    backdropPressed = false;
  });
  document.addEventListener("keydown", (event) => {
    if (!dialog.open) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
    } else if (event.key === "Tab") {
      // Let the native dialog handle focus without the underlying panel's trap.
      event.stopPropagation();
    }
  }, true);
  window.addEventListener("article:navigate", () => close({ restoreFocus: false, restoreScroll: false }), true);
  window.addEventListener("resize", () => { if (dialog.open) syncViewport(); });
  window.MediaViewer = { close, isOpen: () => dialog.open };
})();
