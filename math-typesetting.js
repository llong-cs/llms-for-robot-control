"use strict";

// Only explicit article formulas and opt-in editorial labels are typeset.
// Recorded model outputs, input context, code, and data remain verbatim.
(() => {
  function text(node) {
    if (!node) return "";
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node instanceof Element && node.hasAttribute("data-tex")) {
      return node.dataset.mathText ?? node.textContent;
    }
    return Array.from(node.childNodes, text).join("");
  }

  function typeset(root) {
    if (!window.katex || !root) return;
    const formulas = Array.from(root.querySelectorAll("[data-tex]"));
    if (root instanceof Element && root.matches("[data-tex]")) formulas.unshift(root);
    formulas.forEach((formula) => {
      if (formula.dataset.mathRendered) return;
      const fallback = formula.innerHTML;
      // Preserve one canonical label before KaTeX adds HTML and MathML versions.
      formula.dataset.mathText = text(formula);
      try {
        window.katex.render(formula.dataset.tex, formula, {
          displayMode: formula.classList.contains("math-display"),
          output: "htmlAndMathml",
          throwOnError: true,
          strict: "error",
          trust: false,
        });
        formula.dataset.mathRendered = "true";
      } catch (error) {
        formula.innerHTML = fallback;
        console.warn("Could not typeset an article formula; keeping its readable fallback.", error.message);
      }
    });
  }

  function setText(node, value) {
    const label = String(value ?? "");
    const fragment = document.createDocumentFragment();
    // This small vocabulary is used only in authored chart and dialog labels.
    const pattern = /\bH(?:\s*=\s*\d+(?:\.\d+)?)?\b|π(?:₀\.₅|0\.5)/g;
    let cursor = 0;
    for (const match of label.matchAll(pattern)) {
      fragment.append(document.createTextNode(label.slice(cursor, match.index)));
      const formula = document.createElement("span");
      formula.className = "math";
      formula.dataset.tex = match[0].startsWith("π")
        ? "\\pi_{0.5}" : match[0].replace(/\s*=\s*/, " = ");
      formula.textContent = match[0];
      fragment.append(formula);
      cursor = match.index + match[0].length;
    }
    fragment.append(document.createTextNode(label.slice(cursor)));
    node.replaceChildren(fragment);
    typeset(node);
  }

  window.BlogMath = Object.freeze({ typeset, text, setText });
  typeset(document);
  // Render templates before concept-notes.js clones them into the comment panels.
  document.querySelectorAll("template").forEach((template) => typeset(template.content));
})();
