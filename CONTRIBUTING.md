# Contributing Guide

> This document defines the quality standard for all content in this repository.
> Every section must read like a chapter from a graduate-level textbook written
> by a mathematician who deeply understands modern ML systems.

---

## File Structure — Non-Negotiable

Each section directory must contain exactly three files:

```text
XX-Chapter-Name/
  YY-Section-Name/
    notes.md          # Primary reference — 2000+ lines
    theory.ipynb      # Interactive derivations — 50+ cells
    exercises.ipynb   # Graded problems — 8+ exercises, 3 cells each
```

No other files. No `.py` scripts, no data files, no scratch notebooks.

---

## notes.md Standard

### Required Sections (in order)

1. Navigation header: `[← Back](../README.md) | [Next →](../YY/notes.md)`
2. Title (`# Title`) + epigraph quote
3. `## Overview` — 2–3 paragraphs: what, why, AI connection
4. `## Prerequisites` — bulleted list with section references
5. `## Companion Notebooks` — table linking theory.ipynb and exercises.ipynb
6. `## Learning Objectives` — 8–12 measurable bullet points starting with a verb
7. `## Table of Contents` — complete anchor-linked TOC
8. All numbered sections from the approved TOC
9. `## Common Mistakes` — table with 8+ entries (Mistake / Why Wrong / Fix)
10. `## Exercises` — 8+ problems with difficulty stars
11. `## Why This Matters for AI` — table (Concept / AI Impact)
12. `## Conceptual Bridge` — backward + forward connections + ASCII diagram

### Quality Bar

- **Minimum 2000 lines.** Short sections are incomplete sections.
- **Every definition** must be followed by 3+ examples and 2+ non-examples.
- **Every major theorem** must include a proof sketch or intuition paragraph.
- **AI connections** must be specific: name the paper, model, or technique.
  "used in deep learning" is not acceptable — "used in LoRA (Hu et al., 2022)" is.
- **Notation** must conform to `docs/NOTATION_GUIDE.md` without exception.
- **No emojis** in section content (titles, body text, tables).

---

## theory.ipynb Standard

- **Minimum 50 cells.**
- Built via Python builder script in `/tmp/` — never write notebook JSON directly.
- First code cell must be the exact setup block from `docs/VISUALIZATION_GUIDE.md`.
- Every section in the approved TOC gets at least one code cell.
- Every code cell must `print()` its results — no silent cells.
- All random operations use `np.random.seed(42)` set in the setup cell.
- Numerical results include PASS/FAIL verification: `print(f"{'PASS' if ok else 'FAIL'} — description")`.
- All plots follow `docs/VISUALIZATION_GUIDE.md` (palette, labels, title, tight_layout).
- No external data files — generate everything synthetically.

---

## exercises.ipynb Standard

- **8–12 exercises.** Each exercise = 3 cells: problem (markdown), scaffold (code), solution (code).
- Difficulty: `★` exercises 1–3 (mechanics), `★★` 4–6 (theory), `★★★` 7–8+ (AI applications).
- Every solution cell must include:
  - `header("Exercise N: Title")` call
  - At least one `check_close()` or `check_true()` assertion
  - `print("\nTakeaway: ...")` as the final line
- Scaffold cells must be runnable (print `None` or placeholder) — no syntax errors.
- Solution cells are self-contained — redefine any functions from the scaffold.

---

## Commit Standards

- Stage only project content files: `notes.md`, `*.ipynb`, `docs/*.md`, `README.md`.
- Never stage `.claude/`, `.venv/`, `__pycache__/`, `/tmp/` scripts, or `.DS_Store`.
- Commit message format: imperative mood, present tense, ≤ 72 characters subject line.
  - Good: `Add Eigenvalues section to Advanced Linear Algebra`
  - Bad: `added some stuff`, `WIP`, `fixes`

---

## Review Checklist

Before opening a PR, verify every item:

- [ ] `notes.md` ≥ 2000 lines with all 12 required sections
- [ ] All notation follows `docs/NOTATION_GUIDE.md`
- [ ] All plots follow `docs/VISUALIZATION_GUIDE.md`
- [ ] `theory.ipynb` ≥ 50 cells, valid JSON, runs top-to-bottom without errors
- [ ] `exercises.ipynb` has ≥ 8 exercises with scaffold + solution cells
- [ ] Every solution cell has `header()`, at least one `check_*()`, and `Takeaway:`
- [ ] Common Mistakes table has ≥ 8 entries
- [ ] AI connections cite specific papers or systems
- [ ] No `.claude/` files staged
- [ ] Commit message is clean (no Co-Authored-By lines)
