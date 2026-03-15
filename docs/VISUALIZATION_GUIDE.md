# Visualization Guide

> **Authority:** Every plot produced in this repository must satisfy the rules in
> this document. Consistency across 25 chapters depends on strict adherence.

---

## 1. Mandatory Setup Block

Copy this verbatim into every notebook that produces plots.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="colorblind")
    HAS_SNS = True
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")
    HAS_SNS = False

mpl.rcParams.update({
    "figure.figsize":    (10, 6),
    "figure.dpi":         120,
    "font.size":           13,
    "axes.titlesize":      15,
    "axes.labelsize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     11,
    "legend.framealpha":   0.85,
    "lines.linewidth":      2.0,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "savefig.bbox":       "tight",
    "savefig.dpi":         150,
})
np.random.seed(42)
print("Plot setup complete.")
```

---

## 2. Color Palette

### 2.1 Primary Colors (colorblind-safe, Wong 2011)

| Role | Hex | When to use |
| --- | --- | --- |
| Primary | `#0077BB` | Main curve, first series |
| Secondary | `#EE7733` | Second curve, contrast |
| Tertiary | `#009988` | Third series |
| Error / negative | `#CC3311` | Errors, warnings, losses |
| Neutral | `#555555` | Annotations, reference lines |
| Highlight | `#EE3377` | Special points, emphasis |

```python
COLORS = {
    "primary":   "#0077BB",
    "secondary": "#EE7733",
    "tertiary":  "#009988",
    "error":     "#CC3311",
    "neutral":   "#555555",
    "highlight": "#EE3377",
}
```

### 2.2 Colormaps by Purpose

| Purpose | Colormap | Usage |
| --- | --- | --- |
| Heatmaps, attention | `"viridis"` | Attention weight matrices |
| Diverging (signed) | `"RdBu_r"` | Weight matrices, gradient sign |
| Probability / density | `"plasma"` | Loss landscapes |
| Categorical (≤ 10 classes) | `"tab10"` | Class labels |

> **Forbidden colormaps:** `"jet"`, `"rainbow"`, `"hot"` — they distort perception
> and fail colorblind users. Never encode binary information with red vs. green alone.

---

## 3. Required Plot Elements

Every figure must include all of the following:

```python
fig, ax = plt.subplots()

ax.plot(x, y, color=COLORS["primary"], label="curve label")

# 1. Title — concise, sentence case
ax.set_title("Eigenvalue decay of a random Gaussian matrix")

# 2. Axis labels with units where applicable
ax.set_xlabel("Index $i$")
ax.set_ylabel("Eigenvalue $\\lambda_i$")

# 3. Legend (when 2+ series)
ax.legend(loc="best")

# 4. Tight layout
fig.tight_layout()
plt.show()
```

**Checklist for every figure:**

- [ ] Title set with `ax.set_title()`
- [ ] Both axes labelled with `ax.set_xlabel()` / `ax.set_ylabel()`
- [ ] LaTeX math in labels: `"$\\lambda_i$"` (double backslash in Python strings)
- [ ] Legend present when 2+ series
- [ ] `fig.tight_layout()` called before `plt.show()`
- [ ] All colors from `COLORS` dict or approved colormaps
- [ ] No bare `plt.plot()` without assigning to `ax`

---

## 4. Plot Templates by Type

### 4.1 Line Plot (curves, training loss, convergence)

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, train_loss, color=COLORS["primary"],   label="Train loss")
ax.plot(steps, val_loss,   color=COLORS["secondary"], label="Val loss", linestyle="--")
ax.set_title("Training and validation loss")
ax.set_xlabel("Step")
ax.set_ylabel("Cross-entropy loss")
ax.legend()
fig.tight_layout()
plt.show()
```

### 4.2 Heatmap (matrices, attention, correlation)

```python
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(matrix, cmap="viridis", aspect="auto")
fig.colorbar(im, ax=ax, label="Value")
ax.set_title("Attention weight matrix")
ax.set_xlabel("Key position")
ax.set_ylabel("Query position")
fig.tight_layout()
plt.show()
```

For seaborn:
```python
if HAS_SNS:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_matrix, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Pearson $r$"})
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    plt.show()
```

### 4.3 Scatter Plot (embeddings, data distributions)

```python
fig, ax = plt.subplots(figsize=(8, 8))
for cls_idx, cls_name in enumerate(class_names):
    mask = labels == cls_idx
    ax.scatter(X[mask, 0], X[mask, 1],
               label=cls_name, alpha=0.7, s=40,
               color=plt.cm.tab10(cls_idx / 10))
ax.set_title("2D PCA projection of embeddings")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(markerscale=1.5)
fig.tight_layout()
plt.show()
```

### 4.4 Bar Chart (comparison, ablation results)

```python
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, values, color=COLORS["primary"], alpha=0.85, edgecolor="white")
ax.set_title("Method comparison on benchmark")
ax.set_xlabel("Method")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
# Value annotations
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
plt.show()
```

### 4.5 Multi-Panel Figure

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("SVD approximation at ranks 1, 5, 20", fontsize=15)

for ax, k in zip(axes, [1, 5, 20]):
    approx = low_rank_approx(A, k)
    ax.imshow(approx, cmap="viridis")
    ax.set_title(f"Rank-{k} approximation")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.tight_layout()
plt.show()
```

---

## 5. Mathematical Concept Conventions

### 5.1 Vector / Subspace Geometry

- Draw vectors as arrows with `ax.annotate("", xy=tip, xytext=origin, arrowprops=dict(arrowstyle="->", color=..., lw=2))`
- Show angles with `matplotlib.patches.Arc`
- Label vectors with LaTeX: `ax.text(x, y, r"$\mathbf{v}_1$", fontsize=13)`
- Use `ax.set_aspect("equal")` for any geometric figure — distorted axes are errors

### 5.2 Loss Landscapes

- Use `ax.contourf()` for filled contours with `cmap="plasma"`, `levels=40`
- Overlay gradient descent path with `ax.plot()` in `COLORS["error"]`
- Mark critical points (minima, saddles) with `ax.scatter()`, `marker="*"`, `s=200`

### 5.3 Probability Distributions

- PDF curves: filled with `ax.fill_between(..., alpha=0.15)` + solid line
- Histograms: `ax.hist(..., density=True, bins=40, alpha=0.7)` with overlaid PDF
- Always label x-axis as the random variable: `"$x$"`, `"$z$"`, etc.

### 5.4 Training Dynamics

- Use log-scale y-axis for loss: `ax.set_yscale("log")`
- Mark key events (warmup end, LR drop) with vertical dashed lines:
  `ax.axvline(step, linestyle="--", color=COLORS["neutral"], label="LR drop")`

---

## 6. Text and Annotations

```python
# Equation annotation
ax.annotate(r"$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$",
            xy=(x_pos, y_pos), xytext=(x_pos + dx, y_pos + dy),
            fontsize=12,
            arrowprops=dict(arrowstyle="->", color=COLORS["neutral"]))

# Horizontal / vertical reference line
ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
ax.axvline(threshold, color=COLORS["error"], linewidth=1.2,
           linestyle=":", label="Decision boundary")

# Shaded region
ax.axvspan(x_start, x_end, alpha=0.12, color=COLORS["tertiary"],
           label="Warmup region")
```

---

## 7. What Not to Do

| Violation | Correct approach |
| --- | --- |
| `plt.plot(x, y)` without `ax` | Always use `fig, ax = plt.subplots()` |
| No axis labels | Every axis needs a label, always |
| No title | Every figure needs a title |
| Default `C0`, `C1` colors | Use `COLORS` dict |
| `cmap="jet"` | Use `"viridis"` or `"RdBu_r"` |
| `plt.show()` before `tight_layout` | Call `fig.tight_layout()` first |
| Aspect ratio distortion in geometry | `ax.set_aspect("equal")` for geometric plots |
| Font size below 11 | Set `fontsize` ≥ 11 for all text |
| Legend outside figure bounds | Use `ax.legend(loc="best")` or `bbox_to_anchor` with `bbox_inches="tight"` |

---

*Visualization conventions follow the scientific visualization standards of
Rougier, Droettboom & Bourne (2014) "Ten Simple Rules for Better Figures"
and the colorblind palette of Wong (2011) Nature Methods.*
