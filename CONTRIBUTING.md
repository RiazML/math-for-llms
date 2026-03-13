# Contributing to Mathematics for AI/ML/LLM

Thank you for your interest in contributing! This project thrives on community input.

## How to Contribute

### Report Issues
- Found a typo or incorrect formula? Open an issue
- Spotted a broken notebook? Let us know
- Have a suggestion? We'd love to hear it

### Submit Changes

1. **Fork** the repository
2. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-improvement
   ```
3. **Make your changes** following the guidelines below
4. **Test** your changes (run any modified notebooks)
5. **Commit** with a clear message
   ```bash
   git commit -m "Add: description of your change"
   ```
6. **Push** and open a Pull Request
   ```bash
   git push origin feature/your-improvement
   ```

## What Can You Contribute?

### Fix Errors
- Typos in notes or notebooks
- Incorrect formulas or derivations
- Broken code cells

### Add Content
- New exercises with worked solutions
- Better visualizations and interactive plots
- Additional examples linking math to ML applications

### Improve Explanations
- Clearer intuition for difficult concepts
- More step-by-step derivations
- Better transitions between topics

## Content Guidelines

### File Structure
Every topic folder should follow this structure:
```
XX-Topic-Name/
├── notes.md           # Concepts, intuition, key formulas
├── theory.ipynb       # Interactive demonstrations
└── exercises.ipynb    # Practice problems with solutions
```

### Writing Style
- **Be concise** — explain clearly without unnecessary jargon
- **Show the "why"** — connect every concept to its ML/AI application
- **Use visuals** — a good plot is worth a thousand words
- **Include examples** — concrete beats abstract every time

### Notebook Guidelines
- All cells should run top-to-bottom without errors
- Include clear markdown headers and explanations between code cells
- Use `matplotlib` / `seaborn` / `plotly` for visualizations
- Keep outputs in the notebook so readers can preview without running

### Commit Messages
- `Add:` for new content
- `Fix:` for corrections
- `Update:` for improvements to existing content
- `Docs:` for documentation changes

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to learn.

## Questions?

Open an issue with the `question` label — we're happy to help!
