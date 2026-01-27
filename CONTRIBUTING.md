# Contributing to Mathematics for AI/ML

Thank you for your interest in contributing! This document provides guidelines for contributing to this repository.

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix errors in code, formulas, or explanations
2. **New Content**: Add new examples, exercises, or topics
3. **Improvements**: Enhance existing explanations or visualizations
4. **Documentation**: Improve README files or add comments

### Contribution Process

1. **Fork the Repository**

   ```bash
   git clone https://github.com/yourusername/math_for_ai.git
   cd math_for_ai
   ```

2. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the coding standards below
   - Test your changes
   - Update documentation as needed

4. **Commit Changes**

   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Coding Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write NumPy-style docstrings
- Maximum line length: 88 characters (Black formatter)

```python
def example_function(x: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
    """
    Brief description of the function.

    Parameters
    ----------
    x : np.ndarray
        Description of x
    learning_rate : float, optional
        Description of learning_rate, by default 0.01

    Returns
    -------
    np.ndarray
        Description of return value

    Examples
    --------
    >>> example_function(np.array([1, 2, 3]))
    array([...])
    """
    pass
```

### README.md Style

- Use LaTeX for all formulas: `$inline$` or `$$block$$`
- Include ASCII diagrams where helpful
- Always explain intuition before formal definitions
- Link to ML applications

### Examples and Exercises

- Every example should be runnable independently
- Include visualization where possible
- Add detailed comments explaining each step
- Connect to real ML use cases

## File Structure for New Topics

When adding a new topic, create:

```
XX-Topic-Name/
├── README.md      # Theory and explanations
├── examples.py    # Working implementations
└── exercises.py   # Practice problems
```

## Review Process

All contributions will be reviewed for:

1. **Correctness**: Mathematical accuracy
2. **Clarity**: Clear explanations
3. **Code Quality**: Following style guidelines
4. **Completeness**: All required files present

## Questions?

Feel free to open an issue if you have questions about contributing!
