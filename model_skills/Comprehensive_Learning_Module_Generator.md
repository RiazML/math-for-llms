# Token-Aware Content Generation Skill for GitHub Copilot
## Optimized for Claude Opus 4.5 via GitHub Copilot Subscription

**Version:** 1.1
**Platform:** GitHub Copilot with Claude Opus 4.5
**Critical Constraint:** 2,000 token maximum output per response
**Purpose:** Generate complete learning modules without hitting token limits
**Integration:** Works with `Mathematical_ASCII_Visualization_System.md` and `AI_ML_Mathematics_Modernization_Framework.md`

---

## ABSOLUTE RULES - NEVER VIOLATE

1. **MAXIMUM OUTPUT PER RESPONSE: 1,800 tokens (safety margin)**
2. **NEVER generate complete long-form content in single response**
3. **ALWAYS assess token requirements BEFORE generating content**
4. **ALWAYS split content into chunks when total exceeds 1,500 tokens**
5. **ALWAYS communicate chunking plan to user before starting**

---

## TOKEN REFERENCE GUIDE

### Conversion Table
| Tokens | Words | Characters | Typical Content |
|--------|-------|------------|-----------------|
| 500    | 375   | 1,500      | Short section, 2-3 paragraphs |
| 1,000  | 750   | 3,000      | Medium section, 4-6 paragraphs |
| 1,500  | 1,125 | 4,500      | Full chunk (SAFE TARGET) |
| 1,800  | 1,350 | 5,400      | Maximum safe chunk |
| 2,000  | 1,500 | 6,000      | HARD LIMIT (DO NOT EXCEED) |

### Token Estimation Formula
````
estimated_tokens = (word_count × 1.33) × 1.10
````
- Multiply word count by 1.33 to get base tokens
- Add 10% overhead for formatting/markdown
- Round up to nearest 100

### Quick Estimation
- Short paragraph (50 words) = ~75 tokens
- Medium paragraph (100 words) = ~135 tokens
- Long paragraph (150 words) = ~200 tokens
- Code block (10 lines) = ~150 tokens
- Bullet list (5 items, 10 words each) = ~75 tokens

---

## SKILL ACTIVATION TRIGGERS

### When to Use This Skill

This skill MUST be used when user requests:
- "Create a learning module on [topic]"
- "Write a comprehensive guide about [topic]"
- "Generate documentation for [topic]"
- "Create a tutorial on [topic]"
- "Write detailed content about [topic]"
- "Make a training module for [topic]"
- "Develop a course on [topic]"

### Detection Keywords
- "learning module"
- "comprehensive"
- "detailed"
- "tutorial"
- "guide"
- "documentation"
- "complete"
- "full"
- "in-depth"
- "step-by-step"

**For AI/ML Mathematics Modules:** Also activate when:
- "Create examples.ipynb"
- "Convert .py to .ipynb"
- "Generate README.md"
- "Mathematics module"
- "AI/ML tutorial"

If ANY of these keywords appear AND estimated content > 1,000 words, activate this skill.

---

## INTEGRATION WITH SUPPORTING SKILLS

### Required Integration: ASCII Art Visualization Guide

**MANDATORY:** For all AI/ML mathematics modules, integrate ASCII art visualizations:

**In README.md Generation:**
- Include ASCII art immediately after each major concept introduction
- Use ASCII art as quick reference in examples and derivations
- Place ASCII art before detailed mathematical formulas
- Follow `Mathematical_ASCII_Visualization_System.md` character conventions

**In Jupyter Notebook Generation:**
- Use ASCII art in markdown cells for concept visualization
- Include ASCII art in code comments for algorithm explanation
- Reference ASCII art guide for consistent styling

**ASCII Art Token Budget:** Reserve 100-200 tokens per major concept for ASCII visualizations.

### Connection to AI_ML_Mathematics_Modernization_Framework.md

**Repository Modernization Context:**
- This skill implements the content generation portion of `AI_ML_Mathematics_Modernization_Framework.md`
- Target quality: University-level depth (6,000+ words for README.md)
- Required components: Interactive examples, complete solutions, ASCII art integration
- Success criteria: Equivalent to MIT OCW or Stanford CS229 quality

**Module Completeness Checklist Integration:**
- [ ] **README.md:** 6,000+ words, university-level depth
- [ ] **examples.ipynb:** 20+ interactive examples
- [ ] **exercises.ipynb:** 15+ complete solutions
- [ ] **ASCII Art:** Every major concept has immediate ASCII visualization
- [ ] **Integration:** Seamless flow between ASCII art, theory, examples, exercises

---

## MANDATORY WORKFLOW - FOLLOW EXACTLY

### PHASE 1: ASSESSMENT (ALWAYS DO THIS FIRST)

When user requests learning module or long content:

**Step 1.1: Calculate Requirements**
````
1. Identify all sections needed
2. Estimate words per section
3. Calculate total words
4. Convert to tokens: total_words × 1.33 × 1.10
5. Determine number of chunks: CEILING(total_tokens / 1500)
````

**Step 1.2: Create Chunk Plan**
````
1. Divide content into logical sections
2. Group sections into chunks (1,500 tokens each)
3. Identify break points between chunks
4. Assign section names to each chunk
````

**Step 1.3: Communicate Plan (MANDATORY RESPONSE)**
````
📊 CONTENT ANALYSIS FOR GITHUB COPILOT

Topic: [topic name]
Estimated Length: ~[X] words ([Y] tokens)

⚠️ GitHub Copilot Limit: 2,000 tokens per response
📦 Delivery Strategy: [N] parts

BREAKDOWN:
Part 1 (~1,500 tokens): [section names]
Part 2 (~1,500 tokens): [section names]
Part 3 (~1,500 tokens): [section names]
[add more as needed]

Total Delivery: [N] parts, ~[X] tokens

Ready to begin with Part 1?
(Reply: "yes" / "start" / "continue")
````

**CRITICAL: Do NOT generate content until user confirms**

---

### PHASE 2: CONTENT GENERATION (CHUNK BY CHUNK)

#### Chunk Structure Rules

**Every chunk MUST include:**
1. Part indicator: `PART X/N`
2. Progress context: What was covered previously
3. Main content: Staying within 1,500 token budget
4. Completion marker: `✅ PART X COMPLETE`
5. Token count: Approximate tokens used
6. Progress tracker: `X/N parts complete`
7. Next preview: What Part X+1 will cover
8. Continuation prompt: Clear instruction to user

#### Chunk Template (FOLLOW EXACTLY)
````markdown
# [Module Title] - PART [X]/[N]

[If X > 1, include previous context:]
📍 **Previously in Part [X-1]:** [1-sentence summary]

---

## [Section 1 Title]

[Content for section 1]

## [Section 2 Title]

[Content for section 2]

[Continue sections as token budget allows]

---

✅ PART [X] COMPLETE

📊 **Metrics:**
- Tokens used: ~[X]
- Progress: [X]/[N] parts complete
- Sections covered: [list section names]

📋 **NEXT in Part [X+1]:**
- [Section name 1]
- [Section name 2]

👉 **Type "continue", "next", or "yes" for Part [X+1]**
````

#### First Chunk Template (PART 1)
````markdown
# [Module Title] - PART 1/[N]

## 📊 Module Overview
[2-3 sentences: What this module teaches and why it matters]

**Content Structure:**
This module is delivered in [N] parts:
- Part 1: [topics] ← YOU ARE HERE
- Part 2: [topics]
- Part 3: [topics]
[list all parts]

---

## 📋 Learning Objectives
By completing all [N] parts of this module, you will be able to:
1. [Objective 1 - specific, measurable, action verb]
2. [Objective 2 - specific, measurable, action verb]
3. [Objective 3 - specific, measurable, action verb]
4. [Objective 4 - specific, measurable, action verb]

---

## ✅ Prerequisites
Before starting, you should:
- [Prerequisite 1 with brief explanation]
- [Prerequisite 2 with brief explanation]
- [Prerequisite 3 with brief explanation]

**Estimated Time:** [X] minutes for all parts

---

## 📖 Introduction

### What is [Topic]?
[2-3 paragraphs explaining the concept clearly]

### Why Learn [Topic]?
[2-3 paragraphs explaining real-world relevance and benefits]

### How This Module Works
[1-2 paragraphs explaining learning approach]

---

## 🎯 Core Concept 1: [Concept Name]

### Definition
[Clear, concise definition in 1-2 sentences]

### Explanation
[3-4 paragraphs explaining the concept thoroughly]

### Key Characteristics
1. **[Characteristic 1]:** [Brief explanation]
2. **[Characteristic 2]:** [Brief explanation]
3. **[Characteristic 3]:** [Brief explanation]

### Simple Example
**Scenario:** [Setup context]

**Application:** [Show concept in action]

**Explanation:** [Why this example demonstrates the concept]

---

✅ PART 1 COMPLETE

📊 **Metrics:**
- Tokens used: ~1,500
- Progress: 1/[N] parts complete
- Sections covered: Overview, Objectives, Prerequisites, Introduction, Core Concept 1

📋 **NEXT in Part 2:**
- Core Concept 2: [Name]
- Core Concept 3: [Name]
- Practical Examples

👉 **Type "continue", "next", or "yes" for Part 2**
````

#### Middle Chunk Template (PART 2 to N-1)
````markdown
# [Module Title] - PART [X]/[N]

📍 **Previously in Part [X-1]:** [1-sentence summary of what was covered]

---

## 🎯 Core Concept [X]: [Concept Name]

### Definition
[Clear definition]

### How It Works
[Detailed explanation with 3-4 paragraphs]

### Connection to Previous Concepts
[How this builds on what was learned in previous parts]

### Example [X]
**Scenario:** [Context]

**Step-by-step:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Result:** [Outcome and explanation]

---

## 🎯 Core Concept [X+1]: [Concept Name]

### Definition
[Clear definition]

### Detailed Explanation
[3-4 paragraphs]

### Example [X+1]
[Concrete example with explanation]

---

## 💡 Practical Applications

### Application 1: [Scenario Name]
[Real-world use case explanation]

### Application 2: [Scenario Name]
[Real-world use case explanation]

---

✅ PART [X] COMPLETE

📊 **Metrics:**
- Tokens used: ~1,500
- Progress: [X]/[N] parts complete
- Sections covered: Core Concept [X], Core Concept [X+1], Practical Applications

📋 **NEXT in Part [X+1]:**
- [Preview of next sections]

👉 **Type "continue", "next", or "yes" for Part [X+1]**
````

#### Final Chunk Template (PART N)
````markdown
# [Module Title] - PART [N]/[N] (FINAL)

📍 **Previously:** Covered [brief list of all previous major topics]

---

## 💪 Practice Exercises

### Exercise 1: [Name]
**Difficulty:** [Beginner/Intermediate/Advanced]

**Scenario:** [Setup the problem]

**Task:** [What the learner should do]

**Hints:**
- [Hint 1]
- [Hint 2]

**Solution:**
[Detailed solution with explanation]

---

### Exercise 2: [Name]
[Follow same structure as Exercise 1]

---

### Exercise 3: [Name]
[Follow same structure as Exercise 1]

---

## ✍️ Knowledge Check Assessment

### Question 1
[Question text]

**Options:**
a) [Option A]
b) [Option B]
c) [Option C]
d) [Option D]

**Correct Answer:** [Letter]
**Explanation:** [Why this is correct and others are wrong]

---

### Question 2
[Follow same structure]

---

### Question 3
[Follow same structure]

---

## ⚠️ Common Mistakes & How to Avoid Them

### Mistake 1: [Description]
**Why it happens:** [Explanation]
**How to avoid:** [Prevention strategy]

### Mistake 2: [Description]
**Why it happens:** [Explanation]
**How to avoid:** [Prevention strategy]

### Mistake 3: [Description]
**Why it happens:** [Explanation]
**How to avoid:** [Prevention strategy]

---

## 📝 Module Summary

### Key Takeaways
1. **[Concept 1]:** [One-sentence summary]
2. **[Concept 2]:** [One-sentence summary]
3. **[Concept 3]:** [One-sentence summary]
4. **[Concept 4]:** [One-sentence summary]

### You Now Can:
✓ [Achievement 1 - ties to learning objective]
✓ [Achievement 2 - ties to learning objective]
✓ [Achievement 3 - ties to learning objective]

---

## 📚 Further Learning Resources

### Recommended Reading
1. **[Resource 1]:** [Brief description and why it's useful]
2. **[Resource 2]:** [Brief description and why it's useful]
3. **[Resource 3]:** [Brief description and why it's useful]

### Practice Resources
- [Practice resource 1 with link or description]
- [Practice resource 2 with link or description]

### Next Steps
[Suggest what the learner should do next to continue their learning journey]

---

✅ MODULE COMPLETE! 🎉

📊 **Final Metrics:**
- Total tokens: ~[X] tokens across [N] parts
- Total sections: [Y]
- All learning objectives: ✓ Achieved

📦 **Complete Module Breakdown:**
- Part 1: [sections covered]
- Part 2: [sections covered]
- Part 3: [sections covered]
[list all parts]

🎯 **Mission Accomplished!**

Would you like me to:
- Create additional practice exercises?
- Explain any concept in more depth?
- Create a quick reference guide?
- Generate flashcards for review?

Type your request or "done" if complete.
````

---

## CONTENT ALLOCATION GUIDE

### Section Token Budgets

Use these allocations within each 1,500-token chunk:

| Section Type | Token Budget | Word Equivalent | Usage Notes |
|--------------|-------------|-----------------|-------------|
| Module Overview | 100-150 | 75-115 words | Brief, essential only |
| Learning Objectives | 150-200 | 115-150 words | 3-5 objectives max |
| Prerequisites | 100-150 | 75-115 words | 3-4 items max |
| Introduction | 300-400 | 225-300 words | Context + motivation |
| Core Concept (simple) | 400-500 | 300-375 words | Definition + example |
| Core Concept (complex) | 600-800 | 450-600 words | Deep explanation |
| Example (brief) | 150-200 | 115-150 words | Simple demonstration |
| Example (detailed) | 250-350 | 190-265 words | Step-by-step walkthrough |
| Practice Exercise | 200-300 | 150-225 words | Task + solution |
| Assessment Question | 100-150 | 75-115 words | Question + explanation |
| Summary | 150-200 | 115-150 words | Key takeaways only |
| Resources | 100-150 | 75-115 words | 3-5 links/references |

### Chunk Composition Examples

**Example 1: Concept-Heavy Chunk (1,500 tokens)**
- Core Concept 1: 700 tokens
- Core Concept 2: 700 tokens
- Transition text: 100 tokens

**Example 2: Example-Heavy Chunk (1,500 tokens)**
- Brief concept review: 200 tokens
- Example 1 (detailed): 400 tokens
- Example 2 (detailed): 400 tokens
- Example 3 (detailed): 400 tokens
- Transition text: 100 tokens

**Example 3: Practice Chunk (1,500 tokens)**
- Exercise 1: 300 tokens
- Exercise 2: 300 tokens
- Exercise 3: 300 tokens
- Exercise 4: 300 tokens
- Assessment questions (3): 300 tokens

**Example 4: Mixed Chunk (1,500 tokens)**
- Core Concept: 600 tokens
- Detailed Example: 400 tokens
- Practice Exercise: 300 tokens
- Transition text: 200 tokens

---

## CHUNK BOUNDARY RULES

### Good Break Points (✓ Split Here)
- Between major sections (Introduction → Core Concepts)
- Between different concepts (Concept 1 → Concept 2)
- Between theory and practice (Concepts → Examples)
- Between examples and exercises (Examples → Practice)
- Between practice and assessment (Practice → Knowledge Check)
- After complete subtopics
- After a full example with explanation

### Bad Break Points (✗ Never Split Here)
- Middle of concept explanation
- Middle of example demonstration
- Middle of step-by-step procedure
- Middle of exercise solution
- Between question and answer
- Middle of related list items
- Middle of comparison or contrast

### Transition Guidelines
- Always end chunks with natural conclusion
- Always start new chunks with brief context
- Always reference what was previously covered
- Always preview what's coming next
- Use visual separators (---, horizontal rules)

---

## QUALITY ASSURANCE CHECKLIST

### Before Sending ANY Chunk

**Pre-Generation Checklist:**
- [ ] Token count estimated and under 1,800
- [ ] Chunk ends at logical section boundary
- [ ] Content is complete (no mid-thought cutoff)
- [ ] Part indicator included (PART X/N)
- [ ] Previous content referenced (if Part 2+)
- [ ] Progress metrics included
- [ ] Next part preview included
- [ ] Clear continuation prompt included
- [ ] No orphaned headings or incomplete sections

### After Generating Each Chunk

**Post-Generation Verification:**
- [ ] User received content without errors
- [ ] No truncation occurred
- [ ] User understands what to do next
- [ ] User confirms ready for next part
- [ ] Adjust remaining chunks if user requests changes

---

## TOKEN MONITORING SYSTEM

### Real-Time Monitoring (Conceptual)

While generating content, mentally track:
````
INITIALIZE:
  chunk_token_budget = 1,500
  current_token_count = 0
  
WHILE generating_content:
  FOR EACH section:
    section_tokens = ESTIMATE(section_content)
    
    IF (current_token_count + section_tokens) > 1,500:
      STOP at current section
      ADD completion markers
      ADD continuation prompt
      BREAK
    ELSE:
      INCLUDE section
      current_token_count += section_tokens
      
  IF current_token_count > 1,800:
    WARNING: Approaching limit
    PREPARE to end chunk
````

### Warning Signs (Stop Immediately If Detected)
- Content becoming too detailed/long
- Examples multiplying beyond plan
- Explanations expanding unexpectedly
- Lists growing longer than intended
- Approaching 1,500 words in current chunk

### Emergency Protocol

If approaching 1,800 tokens mid-generation:

1. **Stop at next logical boundary**
2. **Add emergency marker:** `⚠️ Chunk size optimized for GitHub Copilot`
3. **Mark exact stopping point:** `📍 Stopped after: [section name]`
4. **List what remains:** `Still to cover: [list]`
5. **Revise chunk plan:** `Adjusted plan: Will now need [N+1] parts`
6. **Prompt continuation:** `Ready for Part [X]?`

---

## CHUNKING STRATEGIES BY MODULE TYPE

### Strategy 1: Concept-Focused Module (3-5 concepts)

**Characteristics:**
- Heavy on explanations
- Multiple related concepts
- Moderate examples

**Chunking Pattern:**
- **Part 1:** Intro + Objectives + Concept 1
- **Part 2:** Concepts 2-3
- **Part 3:** Concepts 4-5 + Examples
- **Part 4:** Practice + Assessment + Summary

**Token Distribution:**
- Part 1: 1,500 tokens
- Part 2: 1,500 tokens
- Part 3: 1,500 tokens
- Part 4: 1,500 tokens
- **Total: 6,000 tokens (4 parts)**

---

### Strategy 2: Example-Heavy Module (Learning by doing)

**Characteristics:**
- Many practical examples
- Lighter on theory
- Step-by-step demonstrations

**Chunking Pattern:**
- **Part 1:** Intro + Objectives + Core Concepts (brief)
- **Part 2:** Examples 1-3 (detailed)
- **Part 3:** Examples 4-6 (detailed)
- **Part 4:** Practice + Assessment + Summary

**Token Distribution:**
- Part 1: 1,500 tokens
- Part 2: 1,500 tokens
- Part 3: 1,500 tokens
- Part 4: 1,500 tokens
- **Total: 6,000 tokens (4 parts)**

---

### Strategy 3: Practice-Intensive Module (Skill building)

**Characteristics:**
- Focused on application
- Many exercises
- Immediate feedback

**Chunking Pattern:**
- **Part 1:** Intro + Objectives + Core Concepts
- **Part 2:** Detailed Examples
- **Part 3:** Practice Exercises 1-5
- **Part 4:** Practice Exercises 6-10 + Assessment
- **Part 5:** Advanced Challenges + Summary

**Token Distribution:**
- Part 1: 1,500 tokens
- Part 2: 1,500 tokens
- Part 3: 1,500 tokens
- Part 4: 1,500 tokens
- Part 5: 1,500 tokens
- **Total: 7,500 tokens (5 parts)**

---

### Strategy 4: Comprehensive Deep-Dive Module (Graduate level)

**Characteristics:**
- Very detailed explanations
- Complex concepts
- Multiple layers of understanding

**Chunking Pattern:**
- **Part 1:** Intro + Objectives + Prerequisites + Background
- **Part 2:** Core Concept 1 (deep dive)
- **Part 3:** Core Concept 2 (deep dive)
- **Part 4:** Core Concept 3 (deep dive)
- **Part 5:** Integration + Applications
- **Part 6:** Advanced Examples
- **Part 7:** Practice + Assessment + Summary

**Token Distribution:**
- 7 parts × 1,500 tokens each
- **Total: 10,500 tokens (7 parts)**

---

### Strategy 6: Mathematics Module (AI/ML Focus)

**Characteristics:**
- Complex mathematical concepts
- ASCII art visualizations required
- Interactive Jupyter examples
- University-level depth

**Chunking Pattern:**
- **Part 1:** Intro + Objectives + Prerequisites + ASCII Art Guide
- **Part 2:** Core Concept 1 + ASCII Art + Examples
- **Part 3:** Core Concept 2 + ASCII Art + Examples
- **Part 4:** Core Concept 3 + ASCII Art + Applications
- **Part 5:** Practice Exercises + Solutions + Assessment
- **Part 6:** Advanced Topics + Summary + Resources

**Token Distribution:**
- Part 1: 1,500 tokens (ASCII art: 200 tokens)
- Part 2: 1,500 tokens (ASCII art: 200 tokens)
- Part 3: 1,500 tokens (ASCII art: 200 tokens)
- Part 4: 1,500 tokens (ASCII art: 200 tokens)
- Part 5: 1,500 tokens (ASCII art: 100 tokens)
- Part 6: 1,500 tokens (ASCII art: 100 tokens)
- **Total: 9,000 tokens (6 parts)**

**ASCII Art Integration:**
- Reserve 100-200 tokens per major concept
- Place ASCII art immediately after concept introductions
- Use ASCII art in code comments and markdown cells
- Follow `Mathematical_ASCII_Visualization_System.md` conventions

---

## ERROR HANDLING & RECOVERY

### Error Type 1: Token Limit Hit Unexpectedly

**Symptoms:**
- Response truncated mid-sentence
- User reports incomplete delivery
- Content cuts off abruptly

**Recovery Steps:**
````markdown
⚠️ TOKEN LIMIT RECOVERY

I hit GitHub Copilot's token limit faster than estimated.

📋 **What Was Delivered:**
✓ [list completed sections]

📋 **What Remains:**
⏳ [list remaining sections]

🔄 **Revised Delivery Plan:**
I'll now deliver the remaining content in [N] additional parts:
- Part [X]: [sections]
- Part [X+1]: [sections]

This brings the total to [new total] parts instead of [original total].

Ready to continue with Part [X]?
(Reply: "yes" or "continue")
````

---

### Error Type 2: User Requests Changes Mid-Module

**Scenario:** User asks to modify approach after Part 1

**Response Template:**
````markdown
✓ Understood - I'll adjust the approach.

📋 **Original Plan:**
- [original breakdown]

🔄 **Revised Plan:**
- [new breakdown incorporating feedback]

This changes our delivery from [X] parts to [Y] parts.

Ready to proceed with revised Part 2?
````

---

### Error Type 3: User Wants Everything at Once

**Scenario:** User requests "don't split it, give me everything"

**Response (Firm but helpful):**
````markdown
I understand you'd prefer the complete module at once, but here's the constraint:

⚠️ **GitHub Copilot's Hard Limit:** 2,000 tokens per response

📊 **This Module's Size:** ~[X] tokens

If I attempt to deliver this in one response:
❌ Content will be truncated mid-sentence
❌ You'll lose the ending sections
❌ You'll have to ask me to continue anyway
❌ The break will happen at a random, illogical point

✅ **Better Approach (What I'm Proposing):**
- [N] clean, complete parts
- Each part ends at a logical stopping point
- You get ALL content, nothing lost
- Takes [N] "continue" commands from you
- Total time: Same as single truncated delivery + recovery

**Your choice:**
A) Proceed with [N] planned parts (recommended)
B) Attempt single delivery (will truncate, then need recovery)

Which would you prefer?
````

---

### Error Type 4: User Lost Track Between Parts

**Scenario:** User returns after interruption, unsure where they are

**Response:**
````markdown
No problem! Let me get you oriented.

📍 **Where We Are:**
- Module: [Module Title]
- Completed: Parts 1-[X] of [N]
- Current: About to start Part [X+1]

📝 **What You've Received So Far:**
✓ Part 1: [sections covered]
✓ Part 2: [sections covered]
✓ Part [X]: [sections covered]

📋 **What Remains:**
⏳ Part [X+1]: [sections coming]
⏳ Part [X+2]: [sections coming]

Would you like to:
A) Continue with Part [X+1] as planned
B) Quick recap/summary of Parts 1-[X] first
C) Start over from Part 1

Reply with: "continue" / "recap" / "restart"
````

---

## USER COMMUNICATION PROTOCOLS

### Initial Contact (User requests module)

**Response Pattern:**
````markdown
I'll create a [comprehensive/detailed/etc.] learning module on [topic] for you.

⏱️ **Quick Assessment:**
Analyzing requirements... [1 second pause conceptually]

📊 **Delivery Plan:**
Given GitHub Copilot's token limits, optimal delivery is [N] parts.

[Show breakdown table]

This ensures you receive complete, high-quality content without truncation.

Ready to begin? Reply "yes" or "start"
````

---

### Between Parts (Continuation prompt)

**Response Pattern:**
````markdown
[Previous chunk content]

---

✅ PART [X] COMPLETE

📊 Delivered: ~[X] tokens
📍 Progress: [X]/[N] parts ([percentage]%)
✓ Covered: [brief section list]

⏭️ **NEXT:** Part [X+1] covers [preview]

👉 Type "continue", "next", or "yes" when ready
````

---

### Module Completion

**Response Pattern:**
````markdown
[Final chunk content]

---

🎉 MODULE COMPLETE! 🎉

📊 **Final Statistics:**
- Total parts delivered: [N]
- Total tokens: ~[X]
- Total sections: [Y]
- Learning objectives: All ✓ achieved

📚 **Complete Module Contents:**
✓ Part 1: [sections]
✓ Part 2: [sections]
✓ Part 3: [sections]
[list all]

🎯 **You've Successfully Learned:**
[Restate learning objectives as achievements]

💡 **What's Next?**
- Need more practice? Ask for additional exercises
- Want deeper dive? Request advanced topics
- Ready to apply? Start building!
- Need reference? Ask for cheat sheet

How can I help you further? Or type "done" if complete.
````

---

## SPECIAL CASES

### Case 1: Very Short Module (Single Concept, <1,000 words)

**Decision:** Deliver in single chunk

**Template:**
````markdown
# [Module Title] - Complete Module

## 📋 Learning Objectives
[objectives]

## 📖 Introduction
[introduction]

## 🎯 Core Concept
[complete explanation with examples]

## 💪 Practice
[exercises]

## 📝 Summary
[summary]

---

✅ MODULE COMPLETE (Single Part)

📊 Tokens used: ~[X] (under GitHub Copilot limit)

This focused module was delivered in one part as total content fit comfortably within token limits.
````

---

### Case 2: User Requests Specific Part Count

**Scenario:** "Create this in exactly 3 parts"

**Response:**
````markdown
I'll structure this module into exactly 3 parts as requested.

📊 **3-Part Breakdown:**

**Part 1** (~1,500 tokens):
- [sections]

**Part 2** (~1,500 tokens):
- [sections]

**Part 3** (~1,500 tokens):
- [sections]

Note: Estimated total is [X] tokens. Distributing evenly across 3 parts means ~[Y] tokens per part.

Ready for Part 1?
````

---

### Case 3: User Wants PDF/Downloadable Version

**Response:**
````markdown
I'll create the complete module across [N] parts. After delivery, you can:

📄 **Creating a PDF:**
1. Copy all parts into a document
2. Use a markdown-to-PDF converter
3. Or copy into Google Docs/Word and export

📋 **Recommended Approach:**
1. I'll deliver all [N] parts
2. You paste each part into a document sequentially
3. All parts include proper markdown formatting
4. Final document will be properly structured

Shall I proceed with Part 1?
````

---

### Case 4: Intermediate-Level Adjustment Request

**Scenario:** After Part 1, user says "this is too basic/advanced"

**Response:**
````markdown
✓ Understood - I'll adjust the difficulty level.

🔄 **Adjustment:**
- Current level: [level from Part 1]
- Requested level: [new level]
- Parts remaining: [N-1]

**Changes for remaining parts:**
- [Specific adjustment 1]
- [Specific adjustment 2]
- [Specific adjustment 3]

Part 1 remains as-is (already delivered), but Parts 2-[N] will match your preferred level.

Ready for adjusted Part 2?
````

---

## CONTENT QUALITY STANDARDS

### Writing Style Requirements

**Clarity:**
- Use simple, direct language
- Define technical terms on first use
- Use active voice (not passive)
- Keep sentences under 25 words
- One idea per paragraph

**Structure:**
- Clear headings for every section
- Logical flow between sections
- Smooth transitions between chunks
- Consistent formatting throughout
- Visual hierarchy with markdown

**Engagement:**
- Use real-world examples
- Include "why this matters" explanations
- Address reader directly ("you will learn")
- Anticipate and answer questions
- Provide actionable takeaways

**Accuracy:**
- Fact-check all claims
- Use current information
- Cite sources when needed
- Mark opinions as such
- Update outdated content

**ASCII Art Integration (Mathematics Modules):**
- Include ASCII art immediately after concept introductions
- Use ASCII art for immediate intuition before detailed explanations
- Follow character conventions from `Mathematical_ASCII_Visualization_System.md`
- Test ASCII art in GitHub markdown preview
- Ensure accessibility (screen reader compatible)
- Reserve 100-200 tokens per major concept for ASCII visualizations

---

### Example Quality Standards

**Every example must include:**
1. **Context:** Setup the scenario
2. **Application:** Show concept in action
3. **Explanation:** Why/how it works
4. **Takeaway:** What to remember

**Bad Example:**
````markdown
Example: Variables store data.
````

**Good Example:**
````markdown
### Example: Storing User Information

**Context:** You're building a login system and need to remember who's logged in.

**Application:**
```python
username = "alice"
login_time = "2025-01-28 14:30"
```

**Explanation:** The variables `username` and `login_time` store information about the current user. Throughout your program, you can reference `username` to check who's logged in without asking them to enter it repeatedly.

**Takeaway:** Variables let programs remember information for later use, making code more efficient and user-friendly.
````

---

### Practice Exercise Standards

**Every exercise must include:**
1. **Difficulty indicator:** Beginner/Intermediate/Advanced
2. **Clear task:** Exactly what to do
3. **Context/scenario:** Why you're doing it
4. **Hints:** 2-3 helpful tips (optional)
5. **Complete solution:** Not just answer, but explanation
6. **Common mistakes:** What to avoid

**Exercise Template:**
````markdown
### Exercise [N]: [Descriptive Name]

**Difficulty:** [Level]

**Scenario:**
[Real-world context that makes the exercise meaningful]

**Your Task:**
[Clear, specific instructions - numbered steps if multi-part]

**Hints** (try without looking first):
<details>
<summary>Click for hints</summary>

- Hint 1: [Helpful pointer]
- Hint 2: [Another clue]
- Hint 3: [If needed]

</details>

**Solution:**
<details>
<summary>Click to reveal solution</summary>

[Complete solution with code/answer]

**Explanation:**
[Why this solution works]

**Common Mistakes:**
- [Mistake 1]: [How to avoid]
- [Mistake 2]: [How to avoid]

</details>
````

---

## MARKDOWN FORMATTING RULES

### Use Consistent Markdown

**Headings:**
````markdown
# H1: Module Title only
## H2: Major sections (Introduction, Core Concepts, Practice)
### H3: Subsections (What It Is, How It Works)
#### H4: Rarely used, only for deep nesting
````

**Emphasis:**
````markdown
**Bold** for key terms, definitions, important points
*Italic* for emphasis within sentences
`Code` for inline code, commands, variables
````

**Lists:**
````markdown
- Unordered list for features, characteristics
1. Ordered list for steps, procedures, rankings
````

**Code Blocks:**
````markdown
```python
# Use language identifier
def example():
    return "formatted code"
```
````

**Blockquotes:**
````markdown
> Use for important callouts
> But sparingly
````

**Separators:**
````markdown
---
Use horizontal rules between major sections
---
````

**Emojis (Minimal Use):**
````markdown
📋 for objectives/plans
✓ for completed items
⚠️ for warnings
💡 for tips
🎯 for goals
📊 for metrics/data
👉 for calls-to-action
````

---

## TOKEN BUDGET EXAMPLES

### Example 1: Standard 3-Part Module on "Python Functions"

**Total Estimated: 4,500 tokens (3 parts × 1,500)**

**Part 1 Breakdown (1,500 tokens):**
- Module overview: 100 tokens
- Learning objectives: 150 tokens
- Prerequisites: 100 tokens
- Introduction (what/why/how): 400 tokens
- Core Concept 1 (Function Basics): 600 tokens
- Transition text: 150 tokens

**Part 2 Breakdown (1,500 tokens):**
- Previous context: 50 tokens
- Core Concept 2 (Parameters): 700 tokens
- Core Concept 3 (Return Values): 650 tokens
- Transition text: 100 tokens

**Part 3 Breakdown (1,500 tokens):**
- Previous context: 50 tokens
- Examples (3 detailed): 600 tokens
- Practice exercises (3): 450 tokens
- Assessment (3 questions): 250 tokens
- Summary: 100 tokens
- Resources: 50 tokens

---

### Example 2: Deep-Dive 5-Part Module on "Machine Learning Algorithms"

**Total Estimated: 7,500 tokens (5 parts × 1,500)**

**Part 1 (1,500 tokens):**
- Overview: 150 tokens
- Objectives: 200 tokens
- Prerequisites: 200 tokens
- Introduction to ML: 500 tokens
- Algorithm Type 1 (Supervised): 450 tokens

**Part 2 (1,500 tokens):**
- Context: 50 tokens
- Algorithm Type 2 (Unsupervised): 700 tokens
- Algorithm Type 3 (Reinforcement): 700 tokens
- Comparison table: 50 tokens

**Part 3 (1,500 tokens):**
- Context: 50 tokens
- Detailed Example 1 (Linear Regression): 700 tokens
- Detailed Example 2 (K-Means): 700 tokens
- Transition: 50 tokens

**Part 4 (1,500 tokens):**
- Context: 50 tokens
- Detailed Example 3 (Neural Networks): 700 tokens
- When to use which algorithm: 400 tokens
- Common pitfalls: 350 tokens

**Part 5 (1,500 tokens):**
- Context: 50 tokens
- Practice exercises (4): 800 tokens
- Assessment (5 questions): 400 tokens
- Summary: 150 tokens
- Resources: 100 tokens

---

### Example 3: Quick Tutorial on "Git Commit Messages" (Single Part)

**Total: 1,200 tokens (1 part)**

**Single Part Breakdown:**
- Objectives: 100 tokens
- Introduction: 200 tokens
- Core Concept (Good commit messages): 500 tokens
- Examples (5 good vs bad): 250 tokens
- Quick practice: 100 tokens
- Summary: 50 tokens

---

## FINAL CHECKLIST FOR EVERY RESPONSE

### Before Clicking "Send" on ANY Chunk

**Verify:**
- [ ] Token count estimated (under 1,800)
- [ ] Part indicator present (PART X/N)
- [ ] Previous context included (if Part 2+)
- [ ] Content is logically complete
- [ ] No mid-sentence/mid-thought endings
- [ ] Section headings properly formatted
- [ ] Code blocks properly formatted
- [ ] Lists properly formatted
- [ ] ASCII art included for mathematics concepts (if applicable)
- [ ] ASCII art follows `Mathematical_ASCII_Visualization_System.md` conventions
- [ ] ASCII art tested for accessibility
- [ ] Completion marker present (✅ PART X COMPLETE)
- [ ] Token usage noted
- [ ] Progress tracker shown (X/N)
- [ ] Next part preview included
- [ ] Continuation prompt clear and actionable
- [ ] All emojis used correctly
- [ ] No orphaned headings
- [ ] No incomplete examples
- [ ] Markdown syntax correct
- [ ] Consistent formatting throughout

**If ANY checkbox is unchecked, revise before sending.**

---

## SKILL MAINTENANCE

### Version History
- **v1.1 (2026-01-28):** Added ASCII art integration, AI_ML_Mathematics_Modernization_Framework.md connection, enhanced quality standards for mathematics modules
- **v1.0 (2025-01-28):** Initial creation optimized for GitHub Copilot

### Update Protocol
This skill should be reviewed and updated:
- When GitHub Copilot changes token limits
- When better chunking strategies are discovered
- When user feedback suggests improvements
- Quarterly for general maintenance

### Feedback Integration
Track what works:
- Which chunk sizes users prefer
- Which section breakpoints feel most natural
- Which templates get best results
- Which communication patterns are clearest

---

## EMERGENCY OVERRIDE

### When to Ignore This Skill

**Ignore chunking requirements ONLY if:**
1. User explicitly requests single chunk AND
2. Estimated content under 1,200 tokens AND
3. You confirm it will fit safely

**In all other cases, follow this skill strictly.**

---

## SKILL SUCCESS METRICS

### This Skill is Working If:

✅ **No truncated responses:** Content never cuts off mid-sentence
✅ **User clarity:** Users understand the chunking plan
✅ **Smooth delivery:** Clean breaks between parts
✅ **Complete content:** All promised sections delivered
✅ **User satisfaction:** Users report receiving full module
✅ **No complaints:** No user frustration about "hitting token limits"

### This Skill is Failing If:

❌ **Frequent truncation:** Responses often cut off
❌ **User confusion:** Users don't understand why content is split
❌ **Awkward breaks:** Parts end at illogical points
❌ **Missing content:** Sections promised but not delivered
❌ **User frustration:** Complaints about incomplete delivery

---

## CONCLUSION

This skill ensures reliable, complete learning module delivery within GitHub Copilot's strict 2,000-token limit.

### Core Philosophy:
**Better to deliver complete content in planned chunks than incomplete content in failed single attempts.**

### Key Principles:
1. Always assess before generating
2. Always plan chunk strategy
3. Always communicate plan to user
4. Always stay within 1,500-token safety margin
5. Always end chunks at logical boundaries
6. Always mark progress clearly
7. Always provide continuation prompts
8. **For Mathematics Modules:** Always integrate ASCII art visualizations

### Integration Requirements:
- **ASCII Art Guide:** Required for all AI/ML mathematics content
- **AI_ML_Mathematics_Modernization_Framework.md:** Implements repository modernization plan
- **Quality Standards:** University-level depth with immediate intuition
- **Completeness:** No external resources needed for topic mastery

### Expected Outcome:
Users receive complete, high-quality learning modules delivered smoothly across multiple parts, never experiencing truncation or incomplete content. Mathematics modules include ASCII art for immediate conceptual understanding alongside detailed explanations.

---

**SKILL READY FOR USE**

Save this file as: `token_aware_content_generation_skill.md`

Place in: `/mnt/skills/user/` directory

Activate by: Reading this file before generating any learning module

---

**END OF SKILL DOCUMENT**