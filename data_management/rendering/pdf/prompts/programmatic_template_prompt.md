# Programmatic PDF Template Generation — LLM Instructions

**Goal:** Design a creative, free‑form PDF layout for a dataset of samples. You will provide the *code blocks* and *logic* to render it. Your response must enable a Python runtime to render a complete PDF using a **mandatory tracking system** to ensure 100% data coverage.

### Data Coverage Requirement

Every provided sample **must** be displayed at least once. The system uses a `tracker` object to monitor which samples are rendered. You are responsible for querying the data and "consuming" it through the tracker's retrieval method.

---

## The SamplesTracker API

Instead of a raw list, your functions receive a `tracker` object. You must use its methods to interact with the data:

| Method | Description |
| --- | --- |
| **`get_total_count()`** | Returns the total number of samples (e.g., 15). |
| **`get_all_indices()`** | Returns a list of all integer indices `[0, 1, 2, ...]`. |
| **`get_indices_by_label(label)`** | Returns indices where the sample label matches (0 or 1). |
| **`get_unused_indices()`** | Returns indices not yet retrieved for rendering. |
| **`get_sample_metadata(index)`** | Returns a sample dict `{label, instance_text}` for logic **without** marking it as rendered. |
| **`get_samples_for_rendering(indices)`** | **MANDATORY**: Returns the sample data and marks these indices as "Tracked/Rendered." |

---

## Output Format (JSON only)

Return **pure JSON** matching this schema:

```json
{
  "title": "<creative title string>",
  "blocks": ["<block-call-1>", "<block-call-2>", "..."],
  "functions": "<python source defining custom block functions>"
}

```

* **`blocks`**: A list of call strings (e.g., `"title_block()"`, `"render_category(label=1)"`).
* **`functions`**: Python source defining the functions referenced in `blocks`.

---

## Execution Model & Custom Blocks

Every block function you define **must** accept this standard signature:

```python
def my_block(*, theme, tracker, story, styles, llm, **kwargs):
    # 1. Query indices (e.g. tracker.get_indices_by_label(1))
    # 2. RETRIEVE data (e.g. tracker.get_samples_for_rendering(indices))
    # 3. Append ReportLab flowables to story

```

### Constraints & Content Rules

* **Retrieval**: You MUST call `tracker.get_samples_for_rendering(indices)` to get the text for the PDF. If you access data via other means, the coverage validation will fail.
* **Per‑sample content**: Only use **`instance_text`** and **`label`**. Never mention scores or assignments.
* **Available Helpers**:
* `generate_text_paragraph(prompt)`: Returns a string of synthesized text.
* `Paragraph`, `Spacer`, `Table`, `TableStyle`, `colors`: Standard ReportLab primitives.



---

## Example (Sophisticated Tracker Usage)

```json
{
  "title": "Geological Assay Report: Mineral Density Analysis",
  "blocks": [
    "header_block()",
    "render_verified_samples()",
    "render_rejected_samples()",
    "render_catchall()"
  ],
  "functions": "def header_block(*, theme, tracker, story, styles, llm, **kwargs):\n    story.append(Paragraph(\"GEOLOGICAL ASSAY REPORT: SECTOR 7G\", styles['Title']))\n    story.append(Spacer(1, 12))\n    \n    # Removed dynamic text generation to prevent AttributeErrors. Using static thematic text.\n    intro_text = \"The following report details the lithographic analysis of core samples extracted from the northern ridge. Samples have been categorized by mineral density and purity levels. High-yield samples are prioritized for processing.\"\n    story.append(Paragraph(intro_text, styles['Italic']))\n    story.append(Spacer(1, 24))\n\ndef render_verified_samples(*, theme, tracker, story, styles, llm, **kwargs):\n    # Target Label 1 (High Value/Verified)\n    indices = tracker.get_indices_by_label(1)\n    if not indices:\n        return\n\n    # Retrieve data\n    samples = tracker.get_samples_for_rendering(indices)\n    \n    story.append(Paragraph(\"VERIFIED HIGH-YIELD SAMPLES\", styles['Heading2']))\n    story.append(Spacer(1, 6))\n    \n    # Create a nice table for these high-value items\n    table_data = [[Paragraph(\"<b>ID</b>\", styles['Normal']), Paragraph(\"<b>Assay Description</b>\", styles['Normal'])]]\n    \n    for idx, sample in zip(indices, samples):\n        row = [\n            Paragraph(str(idx), styles['Normal']),\n            Paragraph(sample['instance_text'], styles['BodyText'])\n        ]\n        table_data.append(row)\n        \n    t = Table(table_data, colWidths=[50, 400])\n    t.setStyle(TableStyle([\n        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#8B4513')), # SaddleBrown for header\n        ('TEXTCOLOR', (0,0), (-1,0), colors.white),\n        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),\n        ('VALIGN', (0,0), (-1,-1), 'TOP'),\n        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FAF0E6'), colors.white]) # Linen alternate\n    ]))\n    story.append(t)\n    story.append(Spacer(1, 20))\n\ndef render_rejected_samples(*, theme, tracker, story, styles, llm, **kwargs):\n    # Target Label 0 (Low Value/Rejected)\n    # We check for remaining indices that match Label 0 just in case logic overlaps, \n    # but strictly getting by label is cleaner.\n    indices = tracker.get_indices_by_label(0)\n    if not indices:\n        return\n\n    # Retrieve data\n    samples = tracker.get_samples_for_rendering(indices)\n    \n    story.append(Paragraph(\"LOW-DENSITY / REJECTED SAMPLES\", styles['Heading2']))\n    story.append(Spacer(1, 6))\n    \n    for idx, sample in zip(indices, samples):\n        # Render as a log entry\n        text = f\"<font color='#A52A2A'><b>[SAMPLE-{idx}]</b></font>: {sample['instance_text']}\"\n        story.append(Paragraph(text, styles['Normal']))\n        story.append(Spacer(1, 4))\n    \n    story.append(Spacer(1, 20))\n\ndef render_catchall(*, theme, tracker, story, styles, llm, **kwargs):\n    # Safety net to satisfy 100% coverage requirement\n    remaining = tracker.get_unused_indices()\n    if not remaining:\n        return\n\n    story.append(Paragraph(\"UNCATEGORIZED DATA FRAGMENTS\", styles['Heading3']))\n    samples = tracker.get_samples_for_rendering(remaining)\n    for idx, s in zip(remaining, samples):\n        story.append(Paragraph(f\"Fragment {idx}: {s['instance_text']}\", styles['BodyText']))\n        story.append(Spacer(1, 6))"
}

```

---

## Creative Layout Ideas (Visual & Structural)

* **Label-driven Bifurcation**: Use a `Table` to create two columns; use `tracker.get_indices_by_label(0)` for the left and `(1)` for the right.
* **High-Density Grids**: For large datasets, use small fonts and multi-column tables to list hundreds of samples efficiently.
* **Feature Panels**: Pick a specific index, retrieve it via `get_samples_for_rendering([index])`, and wrap it in a `Table` with a background color for emphasis.
* **Thematic Sections**: Query indices matching specific keywords in the `instance_text` using `tracker.get_sample_metadata(i)` before rendering them as a group.

---

## Summary

1. Use **Tracker Methods** to find indices.
2. Use **`get_samples_for_rendering(indices)`** to get the actual data.
3. Ensure **all indices** are passed to the rendering method at least once.
4. Return **Pure JSON** with no markdown fences.

---