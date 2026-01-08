# Classification Rubric for Reddit Posts

## Allowed Labels

The `category` field must be exactly one of: `listicle`, `value-add`, `playbook`, `rant`, `other`, or `null`.

## Category Definitions

### Listicle

**Definition (use this):** A post whose primary payload is a finite set of tools or products ONLY where the reader's value comes from curation/selection, not a process or argument. 

**Must have:**
- 2+ distinct items that could be separated into bullets/numbered entries
- Each item is a thing (tool/product/books), not a step or idea

**Must not be:**
- A step-by-step method (that's Playbook)
- Mainly an explanation/model (that's Value-add)
- Mainly emotional complaint (that's Rant)

**Quick test:** If you remove the list, the post collapses.

### Value-add

**Definition (use this):** A post whose primary payload is a conceptual explanation or insight that improves understanding ("how/why X works"), where the reader's value comes from a model, clarification, or reframing rather than a list of options or a procedure.

**Must have:**
- A central claim/explanation ("the real reason is…", "here's the mental model…", "what's actually happening is…")
- Supporting reasoning and examples (even brief)

**Must not be:**
- A checklist/SOP someone can follow verbatim (that's Playbook)
- A collection of items (that's Listicle)
- Mostly venting/complaining (that's Rant)

**Quick test:** If you summarize it, it becomes "X works because Y."

### Playbook

**Definition (use this):** A post whose primary payload is a reproducible sequence of actions to achieve an outcome, where the reader's value comes from steps, ordering, and execution details.

**Must have:**
- An explicit goal/outcome ("to get X…")
- Steps (numbered or clearly sequential), plus at least one of: prerequisites, tools, checkpoints, pitfalls

**Must not be:**
- Mostly explanation without steps (that's Value-add)
- Mostly item recommendations without a sequence (that's Listicle)
- Mostly emotional grievance (that's Rant)

**Quick test:** A reader could follow it tomorrow and know what to do first/next.

### Rant

**Definition (use this):** A post whose primary payload is emotional evaluation + norm enforcement (venting, calling out, warning), where the reader's value comes from resonance/validation and a shared "this is broken/unfair/stupid" stance, not from actionable steps or neutral explanation.

**Must have:**
- Clear emotional stance (anger/frustration/disgust/betrayal) or moral judgment
- Focus on what's wrong (often with an anecdote or grievance)

**Must not be:**
- A practical how-to guide (that's Playbook)
- A neutral explanation/model (that's Value-add)
- A curated set of options (that's Listicle)

**Quick test:** If you strip the emotion/judgment, there's little left.

### Other

Use this category when:
- The post doesn't fit any of the above categories
- The post is too short or lacks substance
- The post is primarily a link drop, announcement, or meta-discussion
- The post is a simple question without substantial content

### Null

Use `null` only when:
- Classification failed due to technical errors
- The post content is completely unreadable or corrupted

## Decision Rules (Priority Order)

1. **Step-by-step method/framework/template/checklist** → `playbook`
2. **Primarily enumerated list of tips/tools/resources** → `listicle`
3. **Primarily venting/complaint/critique with little structure** → `rant`
4. **Educational/lessons learned/guide without strict steps** → `value-add`
5. **None fit / too little substance** → `other`

## Tie-breaker Rule (for ambiguous posts)

Classify by **dominant unit of value**:
- **Items dominate** → `listicle`
- **Explanation/model dominates** → `value-add`
- **Steps/procedure dominate** → `playbook`
- **Emotion/judgment dominates** → `rant`

If two are present, choose the one that would **break the post** if removed (list vs model vs steps vs emotion).

## Edge Cases

### Hybrids
- Choose the **dominant intent** based on the tie-breaker rule
- **Lower confidence** if ambiguous (set `category_confidence` < 0.7)

### Very Short Selftext or Link-Only
- Classify as `other` with **low confidence** (`category_confidence` < 0.5)
- If the title alone suggests a category but selftext is missing, use the title but note low confidence

### Posts with Multiple Categories
- Use the tie-breaker rule to select the primary category
- The category that would "break" the post if removed is the primary one

## Output Contract (JSON-only)

**CRITICAL:** The model response must be **JSON-only** (no markdown, no prose, no code blocks).

The output object must contain **exactly** these three fields:

```json
{
  "category": "listicle",
  "category_confidence": 0.92,
  "category_rationale": "Post contains 10 distinct tools/resources in a numbered list format."
}
```

### Field Specifications:
- `category`: One of the allowed values (`listicle`, `value-add`, `playbook`, `rant`, `other`, or `null`)
- `category_confidence`: Float between 0.0 and 1.0 (higher = more confident)
- `category_rationale`: One sentence maximum explaining the classification (no long quotes, be concise)

## Examples

### Example 1: Listicle
**Input:** "Here are 5 tools I use for productivity: 1) Notion, 2) Todoist, 3) RescueTime..."

**Expected Output:**
```json
{
  "category": "listicle",
  "category_confidence": 0.95,
  "category_rationale": "Numbered list of distinct productivity tools."
}
```

### Example 2: Value-add
**Input:** "The real reason most startups fail isn't funding—it's premature scaling. Here's the mental model..."

**Expected Output:**
```json
{
  "category": "value-add",
  "category_confidence": 0.88,
  "category_rationale": "Explains a mental model about startup failure with reasoning."
}
```

### Example 3: Playbook
**Input:** "Step 1: Set up your domain. Step 2: Configure DNS. Step 3: Deploy..."

**Expected Output:**
```json
{
  "category": "playbook",
  "category_confidence": 0.93,
  "category_rationale": "Sequential steps with clear goal and execution details."
}
```

### Example 4: Rant
**Input:** "I'm so frustrated with how broken the VC system is. They don't care about founders..."

**Expected Output:**
```json
{
  "category": "rant",
  "category_confidence": 0.90,
  "category_rationale": "Emotional complaint about VC system with moral judgment."
}
```

### Example 5: Other
**Input:** "Anyone know a good CRM?"

**Expected Output:**
```json
{
  "category": "other",
  "category_confidence": 0.60,
  "category_rationale": "Simple question without substantial content."
}
```

