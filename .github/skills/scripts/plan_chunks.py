#!/usr/bin/env python3
"""
Generate Chunk Delivery Plan for Module Content

Analyzes module content and creates an optimal plan for delivering
content in chunks that respect GitHub Copilot's 2,000 token limit.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Formula: words × 1.33 × 1.10 (overhead for formatting)
    """
    word_count = len(text.split())
    base_tokens = word_count * 1.33
    with_overhead = base_tokens * 1.10
    return int(with_overhead)


def extract_sections(content: str) -> List[Dict[str, any]]:
    """Extract sections from markdown content."""
    sections = []
    
    # Split by headings (## or ###)
    pattern = r'^(#{2,3})\s+(.+)$'
    lines = content.split('\n')
    
    current_section = None
    current_content = []
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Save previous section
            if current_section:
                sections.append({
                    "level": current_section["level"],
                    "title": current_section["title"],
                    "content": '\n'.join(current_content),
                    "tokens": estimate_tokens('\n'.join(current_content))
                })
            
            # Start new section
            current_section = {
                "level": len(match.group(1)),
                "title": match.group(2).strip()
            }
            current_content = [line]
        else:
            if current_section:
                current_content.append(line)
    
    # Save last section
    if current_section:
        sections.append({
            "level": current_section["level"],
            "title": current_section["title"],
            "content": '\n'.join(current_content),
            "tokens": estimate_tokens('\n'.join(current_content))
        })
    
    return sections


def create_chunks(sections: List[Dict], target_tokens: int = 1500) -> List[List[Dict]]:
    """
    Group sections into chunks that stay under target token count.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for section in sections:
        section_tokens = section["tokens"]
        
        # If adding this section would exceed target, start new chunk
        if current_tokens + section_tokens > target_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        # If single section exceeds target, it gets its own chunk
        if section_tokens > target_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            chunks.append([section])
            continue
        
        current_chunk.append(section)
        current_tokens += section_tokens
    
    # Add remaining sections
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def analyze_readme(readme_path: Path) -> Dict:
    """Analyze README.md and create chunking plan."""
    with open(readme_path, 'r') as f:
        content = f.read()
    
    total_words = len(content.split())
    total_tokens = estimate_tokens(content)
    
    sections = extract_sections(content)
    chunks = create_chunks(sections)
    
    return {
        "total_words": total_words,
        "total_tokens": total_tokens,
        "num_chunks": len(chunks),
        "chunks": chunks
    }


def print_plan(analysis: Dict, module_name: str):
    """Print the chunking plan."""
    print("\n" + "="*70)
    print(f"📊 CHUNKING PLAN FOR: {module_name}")
    print("="*70)
    
    print(f"\n📄 README.md Analysis:")
    print(f"  Total words: {analysis['total_words']:,}")
    print(f"  Estimated tokens: {analysis['total_tokens']:,}")
    print(f"  GitHub Copilot limit: 2,000 tokens per response")
    print(f"  Safe chunk target: 1,500 tokens")
    print(f"  Number of chunks: {analysis['num_chunks']}")
    
    print("\n" + "="*70)
    print("CHUNK BREAKDOWN:")
    print("="*70)
    
    for i, chunk in enumerate(analysis['chunks'], 1):
        chunk_tokens = sum(section['tokens'] for section in chunk)
        print(f"\n📦 Chunk {i}/{analysis['num_chunks']} (~{chunk_tokens} tokens):")
        for section in chunk:
            indent = "  " * (section['level'] - 1)
            print(f"  {indent}• {section['title']} (~{section['tokens']} tokens)")
    
    print("\n" + "="*70)
    print("DELIVERY INSTRUCTIONS:")
    print("="*70)
    print(f"""
When generating this module:

1. FIRST, show this plan to the user and get confirmation
2. Generate each chunk one at a time
3. Wait for user to type "continue" before next chunk
4. Each chunk should include:
   - Chunk indicator: "README - CHUNK X/{analysis['num_chunks']}"
   - Previous context (for chunks 2+)
   - Full content for sections in that chunk
   - Completion marker: "✅ CHUNK X COMPLETE"
   - Token count and progress
   - Preview of next chunk

Example first message to user:

```
📊 CONTENT ANALYSIS FOR GITHUB COPILOT

Topic: {module_name}
Estimated Length: ~{analysis['total_words']:,} words ({analysis['total_tokens']:,} tokens)

⚠️ GitHub Copilot Limit: 2,000 tokens per response
📦 Delivery Strategy: {analysis['num_chunks']} chunks

BREAKDOWN:""")
    
    for i, chunk in enumerate(analysis['chunks'], 1):
        chunk_tokens = sum(section['tokens'] for section in chunk)
        section_names = [s['title'] for s in chunk]
        print(f"Chunk {i} (~{chunk_tokens} tokens): {', '.join(section_names)}")
    
    print(f"""
Total Delivery: {analysis['num_chunks']} chunks, ~{analysis['total_tokens']:,} tokens

Ready to begin with Chunk 1?
(Reply: "yes" / "start" / "continue")
```
""")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chunk Delivery Plan for Module Content"
    )
    parser.add_argument(
        "module_path",
        type=Path,
        help="Path to the module directory"
    )
    
    args = parser.parse_args()
    
    readme_path = args.module_path / "README.md"
    
    if not readme_path.exists():
        print(f"Error: README.md not found in {args.module_path}")
        return 1
    
    module_name = args.module_path.name
    analysis = analyze_readme(readme_path)
    print_plan(analysis, module_name)
    
    return 0


if __name__ == "__main__":
    exit(main())
