"use strict";

// Only explicitly marked article formulas are typeset; model outputs stay verbatim.
(() => {
  if (!window.katex) return;

  function typeset(root) {
    root.querySelectorAll("[data-tex]").forEach((formula) => {
      if (formula.dataset.mathRendered) return;
      const fallback = formula.innerHTML;
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

  typeset(document);
  // Render templates before concept-notes.js clones them into the comment panels.
  document.querySelectorAll("template").forEach((template) => typeset(template.content));
})();
