
# PDF Report Design — LLM Instructions

**Goal**: Design a professional and creative PDF layout for a dataset. You will act as a Document Architect, selecting and sequencing high-level UI blocks.

## Core Rules
1. **Full Coverage**: Every sample provided must be rendered exactly once.
3. **Pure JSON**: Return only valid JSON. No markdown fences, no preamble.
4. **Professionalism**: Invent a vivid, contextually relevant title based on the `THEME_HINT`.

## Output Format
```json
{
  "title": "Creative Report Title",
  "blocks": [
    "function_name(arg1='value', arg2=123)",
    "another_function()"
  ]
}

```
