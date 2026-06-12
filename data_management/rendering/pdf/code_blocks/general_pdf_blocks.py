from __future__ import annotations

from data_management.rendering.pdf.code_blocks.block_registry import register_block
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from itertools import zip_longest
from reportlab.lib import colors
from typing import Any, Dict, Optional


def _safe_instance_text(sample: Dict[str, Any]) -> str:
    text = sample.get("instance_text") or ""
    return text.strip() or "(no text provided)"


def _extract_property_strings(sample: Dict[str, Any]) -> list[str]:
    property_entries: list[str] = []
    label_meta = sample.get("instance_label_text") or {}
    property_meta = {}
    if isinstance(label_meta, dict):
        property_meta = label_meta.get("properties") or {}
        if not isinstance(property_meta, dict):
            property_meta = {}

    prop_labels = sample.get("property_labels") or {}
    for name, raw_value in prop_labels.items():
        display = property_meta.get(name)
        if display is None:
            display = "N/A" if raw_value is None else str(raw_value)
        property_entries.append(f"{name}: {display}")

    return property_entries


def _format_label_display(
    sample: Dict[str, Any],
    label_mapping: Optional[Dict[int, str]] = None,
) -> str:
    label_idx = sample.get("label")
    label_text = None

    if label_mapping and label_idx in label_mapping:
        label_text = label_mapping[label_idx]

    label_meta = sample.get("instance_label_text")
    if not label_text and isinstance(label_meta, dict):
        label_text = label_meta.get("label") or label_meta.get("name")

    label_display = "Unknown"
    if label_idx is not None:
        label_display = str(label_idx)
    if label_text:
        label_display = f"{label_display} ({label_text})"

    property_entries = _extract_property_strings(sample)
    if property_entries:
        label_display = f"{label_display} | " + "; ".join(property_entries)

    return label_display


def _validate_label(sample: Dict[str, Any], expected_label: Optional[int]) -> None:
    if expected_label is None:
        return
    actual = sample.get("label")
    if actual != expected_label:
        raise ValueError(
            f"Sample label mismatch: expected {expected_label}, found {actual}."
        )


@register_block()
def generic_header(*, story, styles, title="Report", subtitle=None, **kwargs):
    """
    Renders a major report title and an optional italicized subtitle or abstract.
    Use this at the very beginning of the document to set the context.
    """
    story.append(Paragraph(title.upper(), styles['Title']))
    if subtitle:
        story.append(Spacer(1, 6))
        story.append(Paragraph(subtitle, styles['Italic']))
    story.append(Spacer(1, 24))


@register_block()
def render_table_by_label(*, tracker, story, styles, label=1,
                          header_color='#333333', row_color='#F0F0F0', **kwargs):
    """
    Filters samples by label and renders them in a professional
    grid with custom header colors and alternating row backgrounds.
    """
    indices = tracker.get_indices_by_label(label)
    if not indices:
        return

    samples = tracker.get_samples_for_rendering(indices)

    label_mapping = kwargs.get('label_mapping', {})
    if(label in label_mapping):
        title = f"{label_mapping[label]}"
    else:
        title = f"Samples with Label {label}"

    story.append(Paragraph(title, styles['Heading2']))
    story.append(Spacer(1, 8))

    table_data = [
        [
            Paragraph("<b>ID</b>", styles['Normal']),
            Paragraph("<b>Label & Properties</b>", styles['Normal']),
            Paragraph("<b>Content Description</b>", styles['Normal']),
        ]
    ]

    for idx, sample in zip(indices, samples):
        _validate_label(sample, label)
        label_display = _format_label_display(sample, label_mapping)
        table_data.append([
            Paragraph(str(idx), styles['Normal']),
            Paragraph(label_display, styles['BodyText']),
            Paragraph(_safe_instance_text(sample), styles['BodyText'])
        ])

    t = Table(table_data, colWidths=[45, 150, 305])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor(row_color), colors.white])
    ]))
    story.append(t)
    story.append(Spacer(1, 24))


@register_block()
def render_list_by_label(*, tracker, story, styles, label=0,
                         text_color='#000000', tag_name="ITEM", **kwargs):
    """
    Renders a simple vertical list of samples for a specific label.
    Ideal for secondary data, rejected items, or simple logs.
    """
    indices = tracker.get_indices_by_label(label)
    if not indices:
        return

    samples = tracker.get_samples_for_rendering(indices)

    label_mapping = kwargs.get('label_mapping', {})
    if(label in label_mapping):
        title = f"{label_mapping[label]}"
    else:
        title = f"Samples with Label {label}"

    story.append(Paragraph(title, styles['Heading2']))
    story.append(Spacer(1, 8))

    for idx, sample in zip(indices, samples):
        _validate_label(sample, label)
        label_display = _format_label_display(sample, label_mapping)
        text = (
            f"<font color='{text_color}'><b>[{tag_name}-{idx}]</b></font> "
            f"<br/><b>Label:</b> {label_display}<br/>{_safe_instance_text(sample)}"
        )
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))


@register_block()
def render_spotlight(*, tracker, story, styles, index=0, title="Key Highlight", **kwargs):
    """
    Renders a single specific sample (by index) inside a prominent feature box
    with a gold border. Use this to emphasize specific critical cases.
    """
    unused = tracker.get_unused_indices()
    if index not in unused:
        return

    # Tracking happens here
    sample = tracker.get_samples_for_rendering([index])[0]

    story.append(Paragraph(title, styles['Heading3']))
    story.append(Spacer(1, 6))

    label_display = _format_label_display(sample, kwargs.get("label_mapping"))
    p = Paragraph(
        f"<b>Label:</b> {label_display}<br/><i>\"{_safe_instance_text(sample)}\"</i>",
        styles['BodyText'],
    )
    t = Table([[p]], colWidths=[450])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
        ('BOX', (0, 0), (-1, -1), 1, colors.gold),
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(t)
    story.append(Spacer(1, 24))


@register_block()
def generate_text_paragraph(*, story, styles, llm_instruction, title=None, **kwargs):
    """
    Calls an LLM to generate content based on a specific instruction.
    The llm_instruction should only specify what text to generate directly.
    This is useful for adding AI-generated summaries, interpretations,
    or custom headers to the report dynamically.
    """
    # 1. Extract the LLM from kwargs
    llm = kwargs.get('llm')

    if not llm:
        # Gracefully handle missing LLM tools
        story.append(Paragraph("[Error: LLM tool not provided for text generation]", styles['Italic']))
        return

    # 2. Generate the text using the instruction
    # We use the prompt provided in 'llm_instruction'
    prompt_header = "Please generate a short paragraph based on the following instruction, and only respond with the paragraph text:\n"
    raw_content = llm.make_call(prompt_header + llm_instruction).strip()

    # 3. Optional: Render a title for the section if provided
    if title:
        story.append(Paragraph(title, styles['Heading3']))
        story.append(Spacer(1, 6))

    # 4. Render the generated paragraph
    # We use Paragraph to ensure text wrapping and ReportLab styling
    story.append(Paragraph(raw_content, styles['BodyText']))

    # 5. Add standard spacing
    story.append(Spacer(1, 24))


@register_block()
def render_filtered_range(*, tracker, story, styles, start_sample_range, end_sample_range, target_labels,
                          title="Filtered Selection", **kwargs):
    """
    Renders a subset of samples that fall within a specific index range
    AND match a specific set of labels.
    """
    # 1. Get all indices currently available in the tracker
    # We assume 'start' and 'end' refer to the actual sample indices
    all_indices = tracker.get_unused_indices()

    # 2. Filter by range and label membership
    selected_indices = [
        idx for idx in all_indices
        if start_sample_range <= idx <= end_sample_range
           and tracker.get_label_for_index(idx) in target_labels
    ]

    if not selected_indices:
        return

    # 3. Retrieve the actual sample data
    samples = tracker.get_samples_for_rendering(selected_indices)

    # 4. Render header
    story.append(Paragraph(title, styles['Heading2']))
    story.append(Spacer(1, 8))

    # 5. Render as a list (standardized with your other list functions)
    for idx, sample in zip(selected_indices, samples):
        label_val = tracker.get_label_for_index(idx)
        text = (
            f"<b>[ID-{idx}]</b> (Label: {_format_label_display(sample, kwargs.get('label_mapping'))})"
            f"<br/>{_safe_instance_text(sample)}"
        )
        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 18))


@register_block()
def render_double_label_table(*, tracker, story, styles, label_a, label_b,
                              header_color='#2D5986', **kwargs):
    """
    Renders a two-column table comparing samples from two different labels side-by-side.
    Ideal for comparing 'Positive' vs 'Negative' or 'Original' vs 'Edited' data.
    """
    # 1. Fetch indices and samples for both labels
    indices_a = tracker.get_indices_by_label(label_a)
    indices_b = tracker.get_indices_by_label(label_b)

    if not indices_a and not indices_b:
        return

    samples_a = tracker.get_samples_for_rendering(indices_a)
    samples_b = tracker.get_samples_for_rendering(indices_b)

    # 2. Determine Column Headers from mapping
    mapping = kwargs.get('label_mapping', {})
    title_a = mapping.get(label_a, f"Label {label_a}")
    title_b = mapping.get(label_b, f"Label {label_b}")

    # 3. Build Table Data
    # Header Row
    table_data = [[
        Paragraph(f"<b>{title_a}</b>", styles['Normal']),
        Paragraph(f"<b>{title_b}</b>", styles['Normal'])
    ]]

    # Zip the samples together, filling empty slots with an empty string
    for item_a, item_b in zip_longest(samples_a, samples_b, fillvalue=None):
        row = []

        # Format Column A
        if item_a:
            _validate_label(item_a, label_a)
            row.append(Paragraph(
                f"<b>{_format_label_display(item_a, mapping)}</b><br/>{_safe_instance_text(item_a)}",
                styles['BodyText'])
            )
        else:
            row.append(Paragraph("", styles['BodyText']))

        # Format Column B
        if item_b:
            _validate_label(item_b, label_b)
            row.append(Paragraph(
                f"<b>{_format_label_display(item_b, mapping)}</b><br/>{_safe_instance_text(item_b)}",
                styles['BodyText'])
            )
        else:
            row.append(Paragraph("", styles['BodyText']))

        table_data.append(row)

    # 4. Create and Style Table
    # 540 is standard width for A4 with margins; split 50/50
    col_widths = [240, 240]
    t = Table(table_data, colWidths=col_widths)

    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke])
    ]))

    story.append(t)
    story.append(Spacer(1, 24))
