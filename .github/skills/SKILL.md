---
name: aiml-math-module-generator
description: Generate comprehensive, university-level AI/ML mathematics learning modules with README.md files (6,000+ words delivered in chunks), interactive Jupyter notebooks, ASCII art visualizations, and complete exercise sets. Use when creating or modernizing mathematics educational content for AI/ML topics, converting .py to .ipynb, or generating detailed tutorials on mathematical concepts like linear algebra, calculus, probability, optimization, or machine learning theory. Designed for GitHub Copilot's 2,000 token limit with chunk-based delivery.
---

# AI/ML Mathematics Module Generator

Generate world-class, university-level mathematics learning modules optimized for AI/ML education. This skill creates comprehensive content delivered in manageable chunks for GitHub Copilot's token constraints.

## When to Use This Skill

**Required triggers:**
- "Create a learning module on [AI/ML math topic]"
- "Generate comprehensive documentation for [math concept]"
- "Convert .py to .ipynb for [topic]"
- "Modernize mathematics module for [topic]"
- "Create README.md for [math topic]"
- "Generate tutorial on [AI/ML mathematics]"

**Keywords that activate this skill:**
- learning module, comprehensive, detailed, tutorial, complete, in-depth
- README.md, examples.ipynb, exercises.ipynb
- AI/ML mathematics, linear algebra, calculus, probability, optimization
- university-level, MIT/Stanford quality

## Core Workflow

### Phase 1: Content Planning (Always Start Here)

**Step 1.1: Calculate Token Requirements**

```python
# Token estimation formula
total_words = sum_of_all_sections_word_count
total_tokens = total_words × 1.33 × 1.10  # Base tokens + 10% overhead
number_of_chunks = ceiling(total_tokens / 1500)  # Safe chunk size
```

**Step 1.2: Create Delivery Plan**

Always communicate the plan to the user BEFORE generating content:

```
📊 CONTENT ANALYSIS FOR GITHUB COPILOT

Topic: [Topic Name]
Estimated Length: ~[X] words ([Y] tokens)

⚠️ GitHub Copilot Limit: 2,000 tokens per response
📦 Delivery Strategy: [N] chunks

BREAKDOWN:
Chunk 1 (~1,500 tokens): [sections]
Chunk 2 (~1,500 tokens): [sections]
Chunk 3 (~1,500 tokens): [sections]
...

Total Delivery: [N] chunks, ~[Y] tokens

Ready to begin with Chunk 1?
(Reply: "yes" / "start" / "continue")
```

**CRITICAL: Do NOT generate content until user confirms**

### Phase 2: README.md Generation (Chunked Delivery)

**CRITICAL REQUIREMENT: README.md files MUST be delivered in complete chunks. The full content is split across multiple chunks, but each chunk must be complete without truncation.**

**Target Specifications:**
- **Length:** 6,000-10,000 words minimum
- **Depth:** University graduate level (MIT OCW / Stanford CS229 equivalent)
- **Structure:** Delivered in 1,500-token chunks
- **ASCII Art:** Include for every major concept
- **Quality:** Complete enough that no external resources needed

**Required Sections:**
1. Formal Definition & Mathematical Notation
2. Intuitive Explanations (3+ analogies with ASCII art)
3. Complete Mathematical Theory with Derivations
4. Geometric/Visual Interpretations (ASCII art)
5. Computational Methods & Algorithms
6. 8+ Worked Examples (progressive difficulty)
7. ML/AI Applications (5+ real examples)
8. Common Mistakes & Pitfalls
9. Historical Context & Development
10. Advanced Topics & Extensions
11. Prerequisites & Further Reading (15+ references)

**Chunk Structure Template:**

```markdown
# [Topic] README - CHUNK [X]/[N]

📍 **Previously:** [If X > 1, one-sentence summary of previous chunk]

---

## [Section Title]

[Content with ASCII art visualizations]

### ASCII Art Example:
```
[Use templates from references/ascii_art_templates.md]
```

[Continue with theory, examples, derivations]

---

✅ CHUNK [X] COMPLETE

📊 Metrics:
- Tokens: ~[X]
- Progress: [X]/[N] complete
- Sections: [list]

📋 NEXT (Chunk [X+1]):
- [Section names]

👉 Type "continue" for next chunk
```

### Phase 3: Jupyter Notebook Generation

**examples.ipynb Requirements:**
- 20+ interactive examples with visualizations
- Rich markdown explanations between cells
- LaTeX math in markdown cells
- ASCII art for quick concept reference
- Real datasets (sklearn, etc.)
- Progressive difficulty
- Performance analysis

**exercises.ipynb Requirements:**
- 15+ exercises (Easy → Medium → Hard → Challenge)
- Complete solutions with reasoning
- Hints system (progressive disclosure)
- Multiple solution approaches
- Common mistakes section
- Extensions for advanced learners

**Notebook Chunk Structure:**

Each notebook cell should be delivered in logical groups, never split mid-function or mid-explanation.

### Phase 4: Quality Assurance

**Module Completeness Checklist:**
- [ ] README.md: 6,000+ words, university-level depth
- [ ] README.md: Delivered in complete chunks (no truncation)
- [ ] ASCII art: Every major concept visualized
- [ ] examples.ipynb: 20+ interactive examples
- [ ] exercises.ipynb: 15+ exercises with solutions
- [ ] Integration: Seamless theory → examples → exercises flow
- [ ] ML applications: 5+ real-world connections
- [ ] Quality: MIT/Stanford graduate course equivalent

## Token Management Rules

**ABSOLUTE LIMITS:**
- Maximum per response: 1,800 tokens (safety margin)
- Target per chunk: 1,500 tokens
- NEVER generate complete long content in single response
- ALWAYS assess token requirements BEFORE generating

**Token Reference:**
- 500 tokens ≈ 375 words (short section, 2-3 paragraphs)
- 1,000 tokens ≈ 750 words (medium section, 4-6 paragraphs)
- 1,500 tokens ≈ 1,125 words (SAFE TARGET per chunk)
- 1,800 tokens ≈ 1,350 words (maximum safe chunk)

**Estimation Quick Guide:**
- Short paragraph (50 words) = ~75 tokens
- Medium paragraph (100 words) = ~135 tokens
- Code block (10 lines) = ~150 tokens
- ASCII art diagram = 100-200 tokens
- Bullet list (5 items, 10 words each) = ~75 tokens

## ASCII Art Integration

**Required for:**
- Matrix structures and operations
- Vector operations
- Neural network architectures
- Computational graphs
- Coordinate systems and transformations
- Probability distributions
- Optimization landscapes
- Algorithm flowcharts

**Guidelines:**
- Use templates from `references/ascii_art_templates.md`
- Include immediately after concept introduction
- Place before detailed mathematical formulas
- Reserve 100-200 tokens per major concept
- Follow consistent character conventions
- Test for accessibility (screen readers)

See `references/ascii_art_templates.md` for complete template library.

## Script Usage

**Initialize Module Structure:**
```bash
python scripts/init_module.py <topic-name> --output-dir <path>
```

Creates:
```
<topic-name>/
├── README.md (template with TODOs)
├── examples.ipynb (starter notebook)
├── exercises.ipynb (starter notebook)
└── assets/ (for images, data)
```

**Validate Module Completeness:**
```bash
python scripts/validate_module.py <module-path>
```

Checks:
- README.md word count (minimum 6,000)
- Notebook cell counts
- ASCII art presence
- Section completeness
- Quality standards

**Generate Chunk Plan:**
```bash
python scripts/plan_chunks.py <module-path>
```

Analyzes content and creates optimal chunk delivery plan.

## Content Quality Standards

**Theory Sections:**
- Formal mathematical definitions with proper notation
- Step-by-step derivations (no skipped steps)
- Multiple proof approaches when applicable
- Intuitive explanations alongside formalism

**Examples:**
- Start simple, build to complex
- Include both toy and real-world examples
- Show common mistakes and debugging
- Connect to practical ML applications

**Exercises:**
- Progressive difficulty curve
- Include both theory and implementation
- Provide complete solutions, not just answers
- Explain WHY solutions work
- Add extensions for advanced students

## Integration with Supporting References

This skill works with bundled reference files:

**ASCII Art Templates** (`references/ascii_art_templates.md`):
- 50+ ready-to-use visualization templates
- Character conventions and style guide
- Examples for matrices, vectors, neural nets, graphs

**Content Structure Guide** (`references/content_structure.md`):
- Detailed section templates for README.md
- Jupyter notebook cell organization
- Exercise design patterns

**Quality Checklist** (`references/quality_standards.md`):
- University-level benchmarks
- Completeness criteria
- Common pitfalls to avoid

## Example Usage Patterns

**Pattern 1: Single Topic Module**
```
User: "Create a comprehensive module on eigenvalues and eigenvectors"

Claude:
1. Reads this SKILL.md
2. Plans content (~8,000 words = ~11,000 tokens = ~8 chunks)
3. Communicates plan to user
4. Generates README.md chunk by chunk
5. Creates examples.ipynb
6. Creates exercises.ipynb
7. Validates completeness
```

**Pattern 2: Repository Modernization**
```
User: "Modernize my linear algebra repository - convert .py to .ipynb"

Claude:
1. Analyzes existing structure
2. Plans conversion for each topic
3. Generates comprehensive README.md for each
4. Converts to rich Jupyter notebooks
5. Adds ASCII art visualizations
6. Creates exercise notebooks with solutions
```

**Pattern 3: Incremental Expansion**
```
User: "Add backpropagation module to my deep learning repo"

Claude:
1. Checks prerequisites (chain rule, gradients exist?)
2. Plans 70,000-token comprehensive module
3. Delivers in ~48 chunks
4. Includes theory, visualizations, implementations
5. Integrates with existing repository structure
```

## Success Criteria

**A module is complete when:**
✅ Complete beginner can follow intuitive explanations (with ASCII art)
✅ Intermediate student can verify all mathematics
✅ Advanced student can implement from provided code
✅ Practitioner can apply to real ML problems
✅ Researcher can extend to novel applications
✅ All content delivered in complete chunks (no truncation)
✅ README.md is 6,000+ words minimum
✅ No need to consult external resources for this topic

## Critical Reminders

1. **ALWAYS plan chunks before generating content**
2. **NEVER exceed 1,800 tokens per response**
3. **ALWAYS include ASCII art for major concepts**
4. **README.md MUST be 6,000+ words delivered in chunks**
5. **Quality over speed - better 50 excellent chunks than 10 mediocre**
6. **Complete content - include ALL examples, exercises, explanations**
7. **University-level rigor - don't dumb down mathematics**
8. **Chunk at logical boundaries - never mid-sentence or mid-thought**
9. **Each chunk is complete - no truncation within chunks**
