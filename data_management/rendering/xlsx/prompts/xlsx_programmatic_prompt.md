# Structured XLSX Layout Design — LLM Instructions

**Goal:** Design a sophisticated, multi-section Excel report for a dataset. You will define the visual structure, partition samples into meaningful groups, and provide contextual commentary.

### Core Requirements:
1.  **Mandatory Data**: Every `table_block` automatically displays `sample_id`, `text`, and `label`. You do not need to define these; they are always the first three columns.
2.  **Full Coverage**: Every single sample provided in the context must appear at least once in a `table_block`. Use the keyword `"all_remaining"` for the final data section to ensure no data is lost.
3.  **Visual Structure**: Use `header_block` and `text_block` to provide a title, introduction, and section descriptions to make the report readable and professional.

---

## Output Format (JSON only)

Return **pure JSON** (no markdown fences, no backticks) matching this structure:

{
  "sheet_name": "<short_tab_name>",
  "blocks": [
    {
      "type": "header_block",
      "text": "Report Title or Major Section Heading"
    },
    {
      "type": "text_block",
      "text": "Descriptive paragraph explaining the purpose of the following data."
    },
    {
      "type": "table_block",
      "title": "Specific Cluster Name",
      "sample_indices": [0, 1, 2],
      "extra_columns": [
        {"header": "Analysis", "value": "Sample {index} shows {label_name} patterns."},
        {"header": "Meta", "value": "ID: {sample_id}"}
      ]
    },
    {
      "type": "table_block",
      "title": "General Data Pool",
      "sample_indices": "all_remaining",
      "extra_columns": []
    }
  ]
}

### Key Definitions:
- **`type`**: Must be one of:
    - `"header_block"`: Renders a bolded, large-font row.
    - `"text_block"`: Renders a standard text row (ideal for intros/outros).
    - `"table_block"`: Renders a table of data.
- **`text`**: Used in `header_block` and `text_block` for the string content.
- **`sample_indices`**: (For `table_block` only) A list of integers from the preview OR the string `"all_remaining"`.
- **`extra_columns`**: A list of objects with a `header` and a template `value`.
  - Allowed placeholders: `{sample_id}`, `{instance_text}`, `{label_name}`, `{label_details}`, `{index}`.

---

## Constraints & Rules

1.  **No Code Generation**: Do NOT provide Python functions or logic. Provide only valid JSON data.
2.  **No Data Loss**: Ensure every sample index from the preview is accounted for in a `table_block`.
3.  **Clean Placeholders**: Only use the specific placeholders listed above. Do not attempt to use Python methods like `.split()` or `.strip()` inside the curly braces.
4.  **Formatting**: The engine will handle the column widths and bolding. Your job is to provide the **content** and the **partition**.

---

## Example (Professional Audit Layout)

{
  "sheet_name": "Workout Audit Log",
  "blocks": [
    {
      "type": "header_block",
      "text": "Comprehensive Gym Session Analysis"
    },
    {
      "type": "text_block",
      "text": "This report categorizes workout logs to identify high-intensity intervals and recovery patterns across the dataset."
    },
    {
      "type": "table_block",
      "title": "High-Intensity Focus (Failure Sets)",
      "sample_indices": [0, 5, 8],
      "extra_columns": [
        {
          "header": "Intensity Note", 
          "value": "Calculated as {label_name} effort."
        }
      ]
    },
    {
      "type": "header_block",
      "text": "Baseline Data"
    },
    {
      "type": "table_block",
      "title": "General Session Logs",
      "sample_indices": "all_remaining",
      "extra_columns": []
    }
  ]
}

---

## Strategy Ideas

-   **Narrative Flow**: Start with a `header_block` for the title, followed by a `text_block` introduction.
-   **Categorical Grouping**: Group samples with similar labels into their own `table_block` with a descriptive title.
-   **Insightful Metadata**: Use `extra_columns` to synthesize the "story" behind a sample, for example: `"The user performed {instance_text} which is a {label_name} stage."`