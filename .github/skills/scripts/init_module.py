#!/usr/bin/env python3
"""
Initialize AI/ML Mathematics Module Structure

Creates a template directory structure for a new mathematics module
with README.md, Jupyter notebooks, and assets folder.
"""

import os
import sys
import argparse
from pathlib import Path

README_TEMPLATE = """# {topic_title}

## Overview

[TODO: Add 1-2 paragraph introduction to {topic_name}]

## Mathematical Foundation

### Formal Definition

[TODO: Formal mathematical definition with proper notation]

### Intuitive Explanation

[TODO: 3+ analogies explaining the concept]

#### ASCII Art Visualization

```
[TODO: Add ASCII art diagram]
```

## Theory

### Complete Mathematical Derivation

[TODO: Step-by-step derivation]

### Geometric Interpretation

[TODO: Visual/geometric understanding]

## Computational Methods

[TODO: Algorithms and implementation approaches]

## Worked Examples

### Example 1: Basic Application

[TODO: Simple example]

### Example 2: Intermediate Application

[TODO: More complex example]

[Continue with 6+ more examples of progressive difficulty]

## ML/AI Applications

### Application 1: [Specific ML Use Case]

[TODO: Real-world ML application]

[Continue with 4+ more ML applications]

## Common Mistakes & Pitfalls

[TODO: List common errors and misconceptions]

## Historical Context

[TODO: Development of this concept]

## Advanced Topics

[TODO: Extensions and cutting-edge developments]

## Prerequisites

- [TODO: List prerequisites]

## Further Reading

[TODO: 15+ curated references including papers, textbooks, courses]

---

**Module Status:**
- [ ] README.md complete (6,000+ words)
- [ ] examples.ipynb created
- [ ] exercises.ipynb created
- [ ] ASCII art included for all major concepts
- [ ] ML applications documented
- [ ] Validated and tested
"""

EXAMPLES_NOTEBOOK_TEMPLATE = """{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {topic_title} - Interactive Examples\\n",
    "\\n",
    "This notebook contains 20+ interactive examples demonstrating {topic_name}.\\n",
    "\\n",
    "**Learning Objectives:**\\n",
    "1. TODO: Objective 1\\n",
    "2. TODO: Objective 2\\n",
    "3. TODO: Objective 3\\n",
    "\\n",
    "**Prerequisites:** TODO: List prerequisites"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Import required libraries\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "from sklearn import datasets\\n",
    "\\n",
    "# Set random seed for reproducibility\\n",
    "np.random.seed(42)"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Example 1: Basic Concept\\n",
    "\\n",
    "TODO: Explanation with ASCII art\\n",
    "\\n",
    "```\\n",
    "[ASCII art diagram]\\n",
    "```"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# TODO: Implementation code for Example 1\\n",
    "pass"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Example 2: Intermediate Application\\n",
    "\\n",
    "TODO: Continue with more examples"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
"""

EXERCISES_NOTEBOOK_TEMPLATE = """{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {topic_title} - Exercises\\n",
    "\\n",
    "This notebook contains 15+ exercises with complete solutions.\\n",
    "\\n",
    "**Difficulty Levels:**\\n",
    "- 🟢 Easy (1-5)\\n",
    "- 🟡 Medium (6-10)\\n",
    "- 🔴 Hard (11-13)\\n",
    "- ⚫ Challenge (14-15)"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Setup\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Exercise 1: 🟢 Basic Concept (Easy)\\n",
    "\\n",
    "**Problem:**\\n",
    "TODO: Clear problem statement\\n",
    "\\n",
    "**Hints:**\\n",
    "<details>\\n",
    "<summary>Click for hint 1</summary>\\n",
    "TODO: Helpful hint\\n",
    "</details>\\n",
    "\\n",
    "<details>\\n",
    "<summary>Click for hint 2</summary>\\n",
    "TODO: Another hint\\n",
    "</details>"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Your solution here\\n",
    "pass"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "**Solution:**\\n",
    "<details>\\n",
    "<summary>Click to reveal solution</summary>\\n",
    "\\n",
    "TODO: Complete solution with explanation\\n",
    "\\n",
    "**Why this works:**\\n",
    "TODO: Detailed reasoning\\n",
    "\\n",
    "**Common mistakes:**\\n",
    "TODO: Pitfalls to avoid\\n",
    "\\n",
    "</details>"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
"""


def create_module(topic_name: str, output_dir: Path):
    """Create module directory structure with templates."""
    
    # Create topic directory
    topic_dir = output_dir / topic_name
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create assets directory
    (topic_dir / "assets").mkdir(exist_ok=True)
    
    # Format topic title (capitalize and replace hyphens/underscores)
    topic_title = topic_name.replace("-", " ").replace("_", " ").title()
    
    # Create README.md
    readme_path = topic_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(README_TEMPLATE.format(
            topic_title=topic_title,
            topic_name=topic_name
        ))
    
    # Create examples.ipynb
    examples_path = topic_dir / "examples.ipynb"
    with open(examples_path, "w") as f:
        f.write(EXAMPLES_NOTEBOOK_TEMPLATE.format(
            topic_title=topic_title,
            topic_name=topic_name
        ))
    
    # Create exercises.ipynb
    exercises_path = topic_dir / "exercises.ipynb"
    with open(exercises_path, "w") as f:
        f.write(EXERCISES_NOTEBOOK_TEMPLATE.format(
            topic_title=topic_title,
            topic_name=topic_name
        ))
    
    print(f"✅ Module '{topic_name}' created successfully at {topic_dir}")
    print(f"\nCreated files:")
    print(f"  - {readme_path}")
    print(f"  - {examples_path}")
    print(f"  - {exercises_path}")
    print(f"  - {topic_dir / 'assets'}/ (empty directory)")
    print(f"\nNext steps:")
    print(f"  1. Fill in README.md TODOs")
    print(f"  2. Add interactive examples to examples.ipynb")
    print(f"  3. Create exercises in exercises.ipynb")
    print(f"  4. Validate with: python scripts/validate_module.py {topic_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize AI/ML Mathematics Module Structure"
    )
    parser.add_argument(
        "topic_name",
        help="Name of the topic (e.g., 'eigenvalues-eigenvectors')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    create_module(args.topic_name, args.output_dir)


if __name__ == "__main__":
    main()
