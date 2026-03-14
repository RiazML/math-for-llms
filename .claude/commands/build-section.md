# Build Math-for-LLMs Section

You are building content for a world-class mathematics curriculum for AI/ML/LLMs. The user will provide:

1. **Section path** — the target directory (e.g., `03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors`)
2. **Full content list** — a detailed table of contents with all topics, subtopics, and bullet points to implement (or just a topic name)

Your job is to produce three files in that directory: `notes.md`, `theory.ipynb`, and `exercises.ipynb`.

> **MANDATORY WORKFLOW**: Always follow this exact order:
> 1. Research (Phase 1) → 2. Generate TOC (Phase 0) → 3. Get user approval → 4. Write files (Phases 3-5)
>
> Do NOT write a single line of `notes.md`, `theory.ipynb`, or `exercises.ipynb` until the user has explicitly approved the TOC.

---

## Phase 1: Research

Before writing anything, perform deep research:

### 1a. Read existing project standards
- Read `docs/NOTATION_GUIDE.md` — mathematical symbol conventions you MUST follow
- Read `docs/VISUALIZATION_GUIDE.md` — matplotlib/seaborn style rules for all plots
- Read `docs/CHEATSHEET.md` — formula reference to ensure consistency
- Read `docs/ML_MATH_MAP.md` — which math connects to which ML concepts
- Read `CONTRIBUTING.md` — file structure and writing style requirements

### 1b. Study completed sections for style calibration
Read ALL THREE files (notes.md, theory.ipynb, exercises.ipynb) from at least 2 completed sections:
- `01-Mathematical-Foundations/01-Number-Systems/` — baseline style reference
- `02-Linear-Algebra-Basics/05-Matrix-Rank/` — demonstrates theory depth
- `02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/` — most comprehensive example

Pay attention to: line count (~2000-3000+ for notes.md), section depth, ASCII art format, AI connection density, proof style, exercise structure.

### 1c. Understand prerequisite chain
- Read the **previous** section's `notes.md` to know what the student already knows
- Read the **next** section's `notes.md` to know where this section leads
- Read the **chapter README.md** (`../README.md`) to understand section ordering

### 1d. External research
- Search GitHub for high-quality open-source implementations (university courses, textbook repos)
- Search the web for university lecture notes, syllabi, and reference material on this exact topic

---

## Phase 0: Generate and Approve the Table of Contents

**This phase is mandatory regardless of whether the user provides a content list.**

### Step 1: Build the TOC

If the user provided only a topic name, first generate a **Full Content List** with these standard sections:

1. **Intuition** — what is it, why does it matter, why it matters for AI/deep learning, historical timeline
2. **Formal Definitions** — rigorous axioms/definitions, immediate consequences, standard examples, non-examples, edge cases
3. **Core Theory** — 3-8 subsections covering the main theorems, proofs, properties, and computational methods (topic-specific)
4. **Advanced Topics** — deeper theory, connections to functional analysis, infinite-dimensional cases where relevant
5. **Applications in Machine Learning** — PCA, LoRA, attention mechanisms, gradient methods, interpretability, training dynamics (whatever is relevant to THIS topic)
6. **Common Mistakes** — table of frequent errors with explanations and fixes (8-12 entries)
7. **Exercises** — 8 graded exercises (* / ** / ***) covering axiom verification through AI applications
8. **Why This Matters for AI (2026 Perspective)** — table connecting each concept to concrete AI/LLM usage
9. **Conceptual Bridge** — how this topic connects backward (prerequisites) and forward (what it enables)

Each subsection should have 4-8 detailed bullet points. Every bullet should specify WHAT to explain and connect it to AI where relevant.

### Step 2: Present the TOC

From the content list, produce a numbered Table of Contents showing:
- All top-level sections numbered (1. Intuition, 2. Formal Definitions, …)
- All subsections numbered (3.1, 3.2, …)
- A one-sentence description of what each subsection covers
- `[code]` tag next to subsections that will have code cells in `theory.ipynb`
- `[viz]` tag next to subsections that will have visualizations

### Step 3: Wait for approval

Present the TOC to the user and **STOP**. Wait for explicit approval before proceeding.

If the user requests changes, revise and present again. Repeat until approved.

### Step 4: Lock the structure

Once approved, the TOC becomes the canonical blueprint. Every section in `notes.md`, every code cell in `theory.ipynb`, and every exercise in `exercises.ipynb` must implement exactly what the approved TOC specifies — no additions, no omissions.

---

## Phase 3: Write `notes.md`

Write the full `notes.md` implementing every topic from the approved TOC.

**Target depth: 2000-3000+ lines.** Match the depth of existing completed sections.

### Format and structure (match exactly):

```markdown
[← Back to {Chapter Name}](../README.md) | [Next: {Next Section} →](../XX-Next-Section/notes.md)

---

# {Section Title}

> _"A memorable, relevant quote about this topic."_

## Overview
{2-3 paragraphs: what this section covers, why it matters, how it connects to AI}

## Prerequisites
- {Bullet list of specific prior knowledge needed, referencing exact prior sections}

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | {One-line description} |
| [exercises.ipynb](exercises.ipynb) | {One-line description} |

## Learning Objectives

After completing this section, you will:
- {8-12 specific, measurable objectives}

---

## Table of Contents
- [1. Intuition](#1-intuition)
  - [1.1 Subsection Title](#11-subsection-title)
  - [1.2 Subsection Title](#12-subsection-title)
- [2. Formal Definitions](#2-formal-definitions)
{...complete anchor-linked TOC matching ALL sections below}

---

## 1. Intuition
### 1.1 {Subsection}
...

## 2. Formal Definitions
...

{Continue through ALL sections from the approved TOC}

## N. Common Mistakes
| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | ... | ... | ... |
{8-12 entries minimum}

## N+1. Exercises
{Numbered exercise descriptions with difficulty stars and parts (a)-(e)}

## N+2. Why This Matters for AI (2026 Perspective)
| Aspect | Impact |
|---|---|
| ... | ... |

## Conceptual Bridge
{3-4 paragraphs connecting backward and forward}

{ASCII diagram showing position in curriculum — use ═══ bordered box format}
```

### Writing rules for notes.md:

- **Mathematically rigorous but accessible**: define every term before using it; prove key results; provide intuition before formalism
- **AI connections everywhere**: every major concept should have a "**For AI:**" bullet or paragraph explaining how it appears in transformers, LoRA, training, interpretability, etc.
- **Rich examples and non-examples**: for every definition, give 3+ standard examples AND 2+ non-examples showing what fails
- **Proofs where they build understanding**: include proofs that illuminate the concept; skip proofs that are purely mechanical
- **Consistent notation**: use the notation from `docs/NOTATION_GUIDE.md` — vectors in **bold lowercase** ($\mathbf{x}$), matrices in **uppercase** ($A$), etc.
- **No shallow sections**: every section should be substantive. If a subsection would be less than 3 paragraphs, merge it with a neighboring one
- **Historical context**: include timeline showing who discovered what and when, connecting to modern AI usage
- **2026 perspective**: reference current methods (LoRA, DoRA, MLA, FlashAttention, mechanistic interpretability, RoPE) where relevant
- **ASCII art diagrams**: use the ═══ bordered box format used throughout the repo:

```
CONCEPT TITLE
════════════════════════════════════════════════════════════════════════

  Content here — structured visualization, relationship diagrams,
  comparison tables, or process flows

════════════════════════════════════════════════════════════════════════
```

---

## Phase 4: Write `theory.ipynb`

### CRITICAL: How to write Jupyter notebooks

**NEVER write notebook JSON directly via the Write tool.** The LaTeX escaping (`\sum` → `\\sum`, `\frac` → `\\frac`) and string quoting in JSON make direct writing extremely fragile.

**Instead, use the Python builder script pattern:**

1. Write a Python script to `/tmp/build_theory.py` using the Write tool
2. Run it with `python3 /tmp/build_theory.py`
3. Validate with `python3 -c "import json; nb = json.load(open('path/to/theory.ipynb')); print(f'{len(nb[\"cells\"])} cells, valid JSON')"`

#### Builder script template:

```python
import json

path = '/Users/prime/CODE/math_for_llms/{chapter}/{section}/theory.ipynb'

def md(src):
    """Create a markdown cell. src is a plain Python string with \n for newlines."""
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    """Create a code cell. src is a plain Python string with \n for newlines."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

cells = []

# Cell 0: Title
cells.append(md("# Section Title\n\n> *\"Epigraph matching notes.md\"*\n\nInteractive theory notebook..."))

# Cell 1: Setup
cells.append(code(
    "import numpy as np\n"
    "import scipy.linalg as la\n"
    "from scipy import stats\n"
    "\n"
    "try:\n"
    "    import matplotlib.pyplot as plt\n"
    "    import matplotlib\n"
    "    plt.style.use('seaborn-v0_8-whitegrid')\n"
    "    plt.rcParams['figure.figsize'] = [10, 6]\n"
    "    plt.rcParams['font.size'] = 12\n"
    "    HAS_MPL = True\n"
    "except ImportError:\n"
    "    HAS_MPL = False\n"
    "\n"
    "try:\n"
    "    import seaborn as sns\n"
    "    HAS_SNS = True\n"
    "except ImportError:\n"
    "    HAS_SNS = False\n"
    "\n"
    "np.set_printoptions(precision=6, suppress=True)\n"
    "np.random.seed(42)\n"
    "\n"
    "print('Setup complete.')"
))

# Cell 2: Section markdown
cells.append(md("---\n\n## 1. Intuition\n\nBrief mathematical context..."))

# Cell 3: Section code
cells.append(code(
    "# === 1.1 Subsection Title ===\n"
    "\n"
    "# Code here using implicit string concatenation\n"
    "# for multi-line source\n"
    "result = np.array([1, 2, 3])\n"
    "print(f'Result: {result}')\n"
    "\n"
    "# Verify\n"
    "assert np.allclose(result, [1, 2, 3])\n"
    "print('PASS - result verified')"
))

# ... continue for ALL sections ...

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(path, 'w') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(cells)} cells to {path}")
```

**Key rules for the builder script:**
- Use Python implicit string concatenation (`"line1\n" "line2\n"`) for multi-line cell source — this avoids all escaping issues
- Each cell's `source` must be a list containing ONE string: `[src]`
- LaTeX in markdown cells needs NO special escaping — Python raw strings handle it naturally
- For very large notebooks (60+ cells), split into two scripts: one for the first half, one that appends

#### Appending cells to an existing notebook:

```python
import json

path = '/Users/prime/CODE/math_for_llms/{chapter}/{section}/theory.ipynb'
nb = json.load(open(path))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

new_cells = []
new_cells.append(md("---\n\n## 7. New Section\n\nContent..."))
new_cells.append(code("# === 7.1 Title ===\n" "code here\n"))

nb['cells'].extend(new_cells)

with open(path, 'w') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Appended {len(new_cells)} cells. Total: {len(nb['cells'])} cells")
```

### Code cell rules:

- **Libraries**: `numpy`, `scipy`, `matplotlib`, `seaborn`, `sympy`, `scikit-learn` (all in `requirements.txt`). Optional: `torch` (guard with `try/except` and `HAS_TORCH` flag)
- **Setup cell** (first code cell): imports, print options, random seed, matplotlib/seaborn guards (see template above)
- **Every code cell must print results**: no silent cells. Use `print()` with clear labels
- **Use f-strings** for formatted output
- **Guard matplotlib** with `if HAS_MPL:` blocks
- **Guard seaborn** with `if HAS_SNS:` blocks
- **Comment header format**: `# === {section_number}.{subsection} {Title} ===`
- **No external data**: generate all data synthetically with fixed random seeds
- **Show the math working**: print intermediate steps, not just final answers
- **Verify results**: include numerical checks with printed PASS/FAIL:
  ```python
  ok = np.allclose(computed, expected)
  print(f"{'PASS' if ok else 'FAIL'} — description of what was verified")
  ```
- **Matplotlib style**: follow `docs/VISUALIZATION_GUIDE.md` — use colorblind-friendly palettes (`viridis`, `plasma`), label all axes, add titles

### Content coverage:

- Every numbered section in the approved TOC gets at least one code cell
- Visualizations for geometric concepts (subspaces, projections, etc.)
- Simulations for AI connections (gradient subspaces, attention rank, LoRA)
- The notebook must be runnable top-to-bottom without errors
- Typical size: 50-70+ cells

---

## Phase 5: Write `exercises.ipynb`

Write a Jupyter notebook with 8-12 graded exercises, using the **same Python builder script pattern** as theory.ipynb.

### Structure per exercise (3 cells each):

1. **Markdown cell**: problem statement with mathematical context, difficulty stars, parts (a)-(e), LaTeX equations
2. **Code cell (scaffold)**: `# YOUR CODE HERE` placeholders with pre-defined matrices/vectors — must be runnable without errors (prints `None` or placeholder values)
3. **Code cell (solution)**: complete working solution with PASS/FAIL checks and Takeaway line

### Exercise notebook structure:

**Cell 0: Header markdown** — must include:
```markdown
# {Section Title} — Exercises

8 exercises covering {topic summary}.

| Format | Description |
|---|---|
| **Problem** | Markdown cell with task description |
| **Your Solution** | Code cell with scaffolding |
| **Solution** | Code cell with reference solution and checks |

### Difficulty Levels

| Level | Exercises | Focus |
|---|---|---|
| ★ | 1-3 | Core mechanics |
| ★★ | 4-6 | Deeper theory |
| ★★★ | 7-8 | AI / ML applications |

### Topic Map

| Topic | Exercise |
|---|---|
| {topic1} | 1, 2 |
| {topic2} | 3, 4 |
| ... | ... |
```

**Cell 1: Setup cell** — must include these EXACT helper functions:
```python
import numpy as np
import numpy.linalg as la

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

def header(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def check_close(name, got, expected, tol=1e-8):
    ok = np.allclose(got, expected, atol=tol, rtol=tol)
    print(f"{'PASS' if ok else 'FAIL'} — {name}")
    if not ok:
        print("  expected:", expected)
        print("  got     :", got)
    return ok

def check_true(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} — {name}")
    return cond

# Plus any shared utility functions needed across exercises
# e.g., rref(), nullspace_basis(), gram_schmidt()
```

**Exercise cells pattern:**
```python
# Scaffold cell:
# Exercise N: Your Solution
def function_name(params):
    # YOUR CODE HERE
    pass

# Given data
A = np.array([[1, 2], [3, 4]])
result = function_name(A)
print(result)

# Solution cell:
# Exercise N: Solution
def function_name(params):
    # full working implementation
    ...

# Same given data
A = np.array([[1, 2], [3, 4]])
result = function_name(A)

header("Exercise N: Title")
print(f"Result: {result}")
check_close("result matches expected", result, expected_value)
check_true("property holds", some_boolean_condition)
print("\nTakeaway: {one-line insight connecting to AI/ML}")
```

**Final cell: Closing markdown**
```markdown
---

## What to Review After Finishing

- [ ] Checkpoint 1 — description
- [ ] Checkpoint 2 — description
- [ ] ...

## References
1. Reference 1
2. Reference 2
```

### Exercise rules:
- **Difficulty levels**: `★` (core mechanics, exercises 1-3), `★★` (deeper theory, 4-6), `★★★` (AI applications, 7-8+)
- **Every solution cell** MUST include: `header()` call, at least one `check_close()` or `check_true()`, and end with `print("\nTakeaway: ...")`
- **Scaffold cells** must be syntactically valid Python — runnable even if they print `None`
- **Solution cells** must be self-contained — re-define any functions from the scaffold so both cells work independently

---

## Execution Order

1. Read existing files in the target directory
2. Read `docs/NOTATION_GUIDE.md`, `docs/VISUALIZATION_GUIDE.md`, and `CONTRIBUTING.md`
3. Read 2-3 completed sections for style calibration (all 3 files each)
4. Read prerequisite chain (previous/next section notes.md, chapter README.md)
5. Research online (GitHub repos, university courses)
6. Generate the full Table of Contents (Phase 0) — **present to user and STOP. Wait for approval.**
7. Write `notes.md` based on the approved TOC (full content, all sections, 2000+ lines)
8. Write `theory.ipynb` via Python builder script based on the approved TOC
9. Write `exercises.ipynb` via Python builder script based on the approved TOC
10. Validate: run `python3 -c "import json; json.load(open('theory.ipynb'))"` and same for exercises.ipynb
11. Verify: check that every approved TOC entry has corresponding content in all three files

---

## Quality Checklist

Before delivering each file, verify:

- [ ] Every section in the approved TOC is implemented (no skipped topics)
- [ ] No shallow sections (every subsection has 3+ substantive paragraphs in notes.md)
- [ ] AI connections are specific (name the method: LoRA, MLA, RLHF, FlashAttention, RoPE, etc.) not generic
- [ ] Mathematical notation follows `docs/NOTATION_GUIDE.md` consistently
- [ ] Navigation links in notes.md point to correct relative paths
- [ ] Notebook JSON is valid (verified with `python3 -c "import json; json.load(open(...))"`)
- [ ] Code cells use only allowed libraries (numpy, scipy, matplotlib, seaborn, sympy, scikit-learn, optional torch)
- [ ] All random seeds are fixed (`np.random.seed(42)`) for reproducibility
- [ ] Exercises have BOTH scaffold AND solution cells (3 cells per exercise)
- [ ] Every solution cell has `header()`, `check_close()`/`check_true()`, and `Takeaway:` line
- [ ] Common Mistakes table has 8+ entries
- [ ] notes.md has proper anchor-linked Table of Contents
- [ ] notes.md is 2000+ lines with substantive content throughout
- [ ] Plots follow `docs/VISUALIZATION_GUIDE.md` (colorblind-friendly, labeled axes, titles)
- [ ] theory.ipynb has 50+ cells covering all TOC sections

---

## Chunked Delivery

These files are large. Deliver in chunks, pausing for user review after each:

1. **Chunk 1**: `notes.md` — write full file via Write tool
2. **Chunk 2**: `theory.ipynb` first half — write builder script to `/tmp/build_theory.py`, run it
3. **Chunk 3**: `theory.ipynb` second half — write append script to `/tmp/append_theory.py`, run it
4. **Chunk 4**: `exercises.ipynb` — write builder script to `/tmp/build_exercises.py`, run it

After each chunk, pause for user review before continuing. After chunks 2-4, always validate JSON.

---

## Input format

The user will invoke this command as:

```
/build-section <section-path> <content-list or topic-name>
```

Examples:
```
/build-section 03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors {paste content list}
/build-section 08-Optimization/03-Gradient-Descent "Gradient Descent and Variants"
```

If only a topic name is given (no detailed content list), generate the full content list first as part of Phase 0 (TOC generation).

$ARGUMENTS
