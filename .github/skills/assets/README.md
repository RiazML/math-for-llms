# Assets Directory

This directory is reserved for template files and resources that can be copied into generated modules.

## Purpose

Assets are files that are NOT loaded into the context window but are used in the output. Examples include:

- **Template notebooks** with pre-configured cells
- **Sample datasets** for examples
- **Visualization templates** for matplotlib/plotly
- **LaTeX templates** for mathematical notation
- **Configuration files** for Jupyter notebooks

## Current Assets

Currently empty. Add assets as needed for common module generation tasks.

## Usage in Skill

When generating modules, Claude can copy assets from this directory into the module's `assets/` folder.

Example:
```python
# Copy template notebook
shutil.copy(
    "assets/notebook_template.ipynb",
    f"{module_path}/assets/notebook_template.ipynb"
)
```

## Suggested Assets to Add

Consider adding:
- `notebook_template.ipynb` - Pre-configured Jupyter notebook with common imports
- `matplotlib_style.mplstyle` - Consistent visualization styling
- `sample_data.csv` - Small dataset for examples
- `latex_macros.txt` - Common LaTeX macros for mathematical notation
