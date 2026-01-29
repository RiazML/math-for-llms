#!/usr/bin/env python3
"""
Validate AI/ML Mathematics Module Completeness

Checks if a module meets quality standards:
- README.md word count (minimum 6,000)
- Notebook cell counts
- ASCII art presence
- Required sections
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def check_readme(readme_path: Path) -> Tuple[bool, Dict]:
    """Validate README.md file."""
    results = {
        "exists": False,
        "word_count": 0,
        "has_ascii_art": False,
        "required_sections": [],
        "missing_sections": [],
        "warnings": []
    }
    
    if not readme_path.exists():
        results["warnings"].append("README.md not found")
        return False, results
    
    results["exists"] = True
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Count words
    results["word_count"] = count_words(content)
    
    # Check for ASCII art (look for code blocks with box drawing characters)
    ascii_art_patterns = [
        r'```[\s\S]*?[┌┐└┘├┤┬┴┼─│╔╗╚╝╠╣╦╩╬═║][\s\S]*?```',
        r'[┌┐└┘├┤┬┴┼─│╔╗╚╝╠╣╦╩╬═║]'
    ]
    for pattern in ascii_art_patterns:
        if re.search(pattern, content):
            results["has_ascii_art"] = True
            break
    
    # Check for required sections
    required_sections = [
        "Overview",
        "Mathematical Foundation",
        "Theory",
        "Worked Examples",
        "ML/AI Applications",
        "Common Mistakes",
        "Prerequisites",
        "Further Reading"
    ]
    
    for section in required_sections:
        # Look for section headers (## or ###)
        pattern = rf'###+\s*{re.escape(section)}'
        if re.search(pattern, content, re.IGNORECASE):
            results["required_sections"].append(section)
        else:
            results["missing_sections"].append(section)
    
    # Add warnings
    if results["word_count"] < 6000:
        results["warnings"].append(
            f"README.md is only {results['word_count']} words "
            f"(minimum 6,000 required)"
        )
    
    if not results["has_ascii_art"]:
        results["warnings"].append("No ASCII art detected in README.md")
    
    if results["missing_sections"]:
        results["warnings"].append(
            f"Missing sections: {', '.join(results['missing_sections'])}"
        )
    
    # Pass if word count >= 6000 and has ASCII art
    passed = results["word_count"] >= 6000 and results["has_ascii_art"]
    
    return passed, results


def check_notebook(notebook_path: Path, min_cells: int = 10) -> Tuple[bool, Dict]:
    """Validate Jupyter notebook."""
    results = {
        "exists": False,
        "cell_count": 0,
        "markdown_cells": 0,
        "code_cells": 0,
        "has_ascii_art": False,
        "warnings": []
    }
    
    if not notebook_path.exists():
        results["warnings"].append(f"{notebook_path.name} not found")
        return False, results
    
    results["exists"] = True
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        cells = notebook.get("cells", [])
        results["cell_count"] = len(cells)
        
        for cell in cells:
            cell_type = cell.get("cell_type", "")
            if cell_type == "markdown":
                results["markdown_cells"] += 1
                # Check for ASCII art in markdown cells
                source = "".join(cell.get("source", []))
                if re.search(r'[┌┐└┘├┤┬┴┼─│╔╗╚╝╠╣╦╩╬═║]', source):
                    results["has_ascii_art"] = True
            elif cell_type == "code":
                results["code_cells"] += 1
        
        if results["cell_count"] < min_cells:
            results["warnings"].append(
                f"{notebook_path.name} has only {results['cell_count']} cells "
                f"(minimum {min_cells} recommended)"
            )
        
        if results["code_cells"] == 0:
            results["warnings"].append(
                f"{notebook_path.name} has no code cells"
            )
        
        passed = results["cell_count"] >= min_cells and results["code_cells"] > 0
        
    except json.JSONDecodeError:
        results["warnings"].append(f"{notebook_path.name} is not valid JSON")
        passed = False
    except Exception as e:
        results["warnings"].append(f"Error reading {notebook_path.name}: {str(e)}")
        passed = False
    
    return passed, results


def validate_module(module_path: Path) -> Dict:
    """Validate entire module."""
    results = {
        "module_path": str(module_path),
        "readme": {},
        "examples": {},
        "exercises": {},
        "overall_pass": False,
        "summary": []
    }
    
    # Check README.md
    readme_pass, readme_results = check_readme(module_path / "README.md")
    results["readme"] = readme_results
    
    # Check examples.ipynb
    examples_pass, examples_results = check_notebook(
        module_path / "examples.ipynb",
        min_cells=20
    )
    results["examples"] = examples_results
    
    # Check exercises.ipynb
    exercises_pass, exercises_results = check_notebook(
        module_path / "exercises.ipynb",
        min_cells=15
    )
    results["exercises"] = exercises_results
    
    # Overall assessment
    results["overall_pass"] = readme_pass and examples_pass and exercises_pass
    
    # Generate summary
    results["summary"] = []
    
    if readme_pass:
        results["summary"].append("✅ README.md meets standards")
    else:
        results["summary"].append("❌ README.md needs improvement")
    
    if examples_pass:
        results["summary"].append("✅ examples.ipynb meets standards")
    else:
        results["summary"].append("❌ examples.ipynb needs improvement")
    
    if exercises_pass:
        results["summary"].append("✅ exercises.ipynb meets standards")
    else:
        results["summary"].append("❌ exercises.ipynb needs improvement")
    
    return results


def print_results(results: Dict):
    """Print validation results in readable format."""
    print("\n" + "="*70)
    print(f"MODULE VALIDATION REPORT: {Path(results['module_path']).name}")
    print("="*70)
    
    # README.md results
    print("\n📄 README.md:")
    readme = results["readme"]
    if readme["exists"]:
        print(f"  Word count: {readme['word_count']} "
              f"{'✅' if readme['word_count'] >= 6000 else '❌'}")
        print(f"  ASCII art: {'✅ Found' if readme['has_ascii_art'] else '❌ Missing'}")
        print(f"  Sections: {len(readme['required_sections'])}/{len(readme['required_sections']) + len(readme['missing_sections'])}")
        if readme["warnings"]:
            print("  Warnings:")
            for warning in readme["warnings"]:
                print(f"    ⚠️  {warning}")
    else:
        print("  ❌ Not found")
    
    # examples.ipynb results
    print("\n📓 examples.ipynb:")
    examples = results["examples"]
    if examples["exists"]:
        print(f"  Total cells: {examples['cell_count']}")
        print(f"  Markdown cells: {examples['markdown_cells']}")
        print(f"  Code cells: {examples['code_cells']}")
        print(f"  ASCII art: {'✅ Found' if examples['has_ascii_art'] else '⚠️ Missing'}")
        if examples["warnings"]:
            print("  Warnings:")
            for warning in examples["warnings"]:
                print(f"    ⚠️  {warning}")
    else:
        print("  ❌ Not found")
    
    # exercises.ipynb results
    print("\n📝 exercises.ipynb:")
    exercises = results["exercises"]
    if exercises["exists"]:
        print(f"  Total cells: {exercises['cell_count']}")
        print(f"  Markdown cells: {exercises['markdown_cells']}")
        print(f"  Code cells: {exercises['code_cells']}")
        if exercises["warnings"]:
            print("  Warnings:")
            for warning in exercises["warnings"]:
                print(f"    ⚠️  {warning}")
    else:
        print("  ❌ Not found")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    for item in results["summary"]:
        print(f"  {item}")
    
    print("\n" + "="*70)
    if results["overall_pass"]:
        print("🎉 MODULE PASSES ALL QUALITY STANDARDS!")
    else:
        print("📋 MODULE NEEDS ADDITIONAL WORK")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate AI/ML Mathematics Module Completeness"
    )
    parser.add_argument(
        "module_path",
        type=Path,
        help="Path to the module directory"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.module_path.is_dir():
        print(f"Error: {args.module_path} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    results = validate_module(args.module_path)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)
    
    # Exit with code 0 if passed, 1 if failed
    sys.exit(0 if results["overall_pass"] else 1)


if __name__ == "__main__":
    main()
