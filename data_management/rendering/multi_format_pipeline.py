from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from openpyxl import Workbook, load_workbook
from pypdf import PdfReader, PdfWriter  # type: ignore
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from data_management.rendering.xlsx.code_blocks.general_xlsx_blocks import (
    render_cover_sheet,
    render_raw_samples_sheet,
    render_label_ribbon_matrix,
    render_dual_column_landscape,
)

IndexedSample = Tuple[int, Dict[str, Any]]

__all__ = [
    "render_instance_with_intermediate_artifacts",
    "convert_xlsx_to_pdf",
    "convert_ppt_to_pdf",
]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unnamed"


def _load_instance_data(
    dataset_root: str, instance_name: str, level: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    instances_root = os.path.join(dataset_root, "instances")
    for root, _, files in os.walk(instances_root):
        for filename in files:
            if filename in (f"{instance_name}.json", f"{_slug(instance_name)}.json"):
                path = os.path.join(root, filename)
                with open(path, "r", encoding="utf-8") as f:
                    return path, json.load(f)
    raise FileNotFoundError(f"Instance '{instance_name}' not found in {instances_root}")


def _label_mapping(instance_data: Dict[str, Any]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    text_meta = instance_data.get("text_mapping") or {}
    labels = text_meta.get("output_labels") or {}
    if isinstance(labels, dict):
        for key, value in labels.items():
            try:
                mapping[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    return mapping


def _split_indices(total: int) -> Tuple[List[int], List[int]]:
    if total <= 1:
        return [0], [0]
    even = list(range(0, total, 2))
    odd = [idx for idx in range(total) if idx not in even]
    if not even:
        even = [0]
    if not odd:
        odd = [even[-1]]
    return even, odd


def _build_indexed_subset(samples: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[IndexedSample]:
    return [(idx, samples[idx]) for idx in indices if 0 <= idx < len(samples)]


def _build_workbook(
    *,
    title: str,
    subset: Sequence[IndexedSample],
    total_samples: int,
    label_mapping: Dict[int, str],
) -> Tuple[Workbook, Set[int]]:
    wb = Workbook()
    overview = wb.active
    overview.title = "Overview"
    render_cover_sheet(
        worksheet=overview,
        title=f"{title} Spreadsheet Summary",
        subtitle="Interim XLSX asset used for downstream PDF export",
        summary_items={
            "Total samples in dataset": str(total_samples),
            "Samples covered in this workbook": str(len(subset)),
            "Generated at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    used: Set[int] = set()
    raw_sheet = wb.create_sheet("Raw Data")
    used |= render_raw_samples_sheet(
        worksheet=raw_sheet,
        indexed_samples=subset,
        label_mapping=label_mapping,
    )

    ribbon_sheet = wb.create_sheet("Label Matrix")
    used |= render_label_ribbon_matrix(
        worksheet=ribbon_sheet,
        indexed_samples=subset,
    )

    cards_sheet = wb.create_sheet("Story Cards")
    used |= render_dual_column_landscape(
        worksheet=cards_sheet,
        indexed_samples=subset,
        columns=2,
    )

    return wb, used


def _build_presentation(
    *,
    title: str,
    subset: Sequence[IndexedSample],
) -> Tuple[Presentation, Set[int]]:
    presentation = Presentation()
    render_cover_slide(presentation=presentation, title=f"{title} Slide Narrative", subtitle="Interim PPT asset")

    used: Set[int] = set()
    used |= render_sample_spotlight_slide(presentation=presentation, indexed_samples=subset[:4])
    used |= render_storyboard_slide(presentation=presentation, indexed_samples=subset[:3])
    used |= render_multi_column_showcase_slide(presentation=presentation, indexed_samples=subset, columns=3)
    used |= render_rotated_label_band_slide(presentation=presentation, indexed_samples=subset)

    return presentation, used


def convert_xlsx_to_pdf(xlsx_path: str, pdf_path: str, rotate: bool = False) -> None:
    workbook = load_workbook(xlsx_path)
    styles = getSampleStyleSheet()
    story = []

    page_width, page_height = landscape(A4)
    left_margin = 36
    right_margin = 36
    top_margin = 40
    bottom_margin = 36
    usable_width = page_width - left_margin - right_margin

    body_style = styles["BodyText"].clone("XLSXBody")
    body_style.wordWrap = "CJK"
    header_style = styles["Heading5"].clone("XLSXHeader")
    header_style.wordWrap = "CJK"

    MAX_CELL_CHARS = 600

    def _expand_row_cells(row: List[str]) -> List[List[str]]:
        normalized = [str(cell or "") for cell in row]
        expanded: List[List[str]] = [normalized.copy()]
        for col_idx, value in enumerate(normalized):
            text = value or ""
            if len(text) <= MAX_CELL_CHARS:
                continue
            chunks = [text[i:i + MAX_CELL_CHARS] for i in range(0, len(text), MAX_CELL_CHARS)]
            expanded[0][col_idx] = chunks[0]
            for chunk in chunks[1:]:
                continuation = [""] * len(normalized)
                continuation[col_idx] = chunk
                expanded.append(continuation)
        return expanded

    for sheet_index, sheet in enumerate(workbook.worksheets):
        rows = _extract_worksheet_values(sheet)
        if not rows:
            continue
        story.append(Paragraph(sheet.title, styles["Heading2"]))
        story.append(Spacer(1, 6))
        table_rows: List[Tuple[List[str], bool]] = []
        max_cols = 0
        for row_idx, row in enumerate(rows):
            expanded = _expand_row_cells(row)
            for exp_idx, expanded_row in enumerate(expanded):
                is_header = row_idx == 0 and exp_idx == 0
                table_rows.append((expanded_row, is_header))
                max_cols = max(max_cols, len(expanded_row))

        if max_cols == 0 or not table_rows:
            continue
        col_width = usable_width / max_cols
        col_widths = [col_width] * max_cols
        table_data = []
        for expanded_row, is_header in table_rows:
            padded = list(expanded_row) + [""] * (max_cols - len(expanded_row))
            formatted = []
            for cell in padded:
                text = escape(str(cell or ""))
                text = text.replace("\n", "<br/>")
                style = header_style if is_header else body_style
                formatted.append(Paragraph(text, style))
            table_data.append(formatted)
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF5FB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FDF2E9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
        if sheet_index < len(workbook.worksheets) - 1:
            story.append(PageBreak())

    if not story:
        story.append(Paragraph("Workbook was empty.", styles["BodyText"]))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=(page_width, page_height),
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )
    doc.build(story)

    if rotate:
        _rotate_pdf_in_place(pdf_path, degrees=90)


def _extract_worksheet_values(sheet) -> List[List[str]]:
    values: List[List[str]] = []
    for row in sheet.iter_rows(values_only=True):
        rendered = ["" if cell is None else str(cell) for cell in row]
        if any(cell.strip() for cell in rendered):
            while rendered and rendered[-1] == "":
                rendered.pop()
            values.append(rendered)
    return values


def _rotate_pdf_in_place(path: str, degrees: int) -> None:
    reader = PdfReader(path)
    writer = PdfWriter()
    for page in reader.pages:
        if hasattr(page, "rotate"):
            page.rotate(degrees)
        elif hasattr(page, "rotate_clockwise"):
            page.rotate_clockwise(degrees)
        else:
            current = getattr(page, "rotation", 0) or 0
            page.rotation = (current + degrees) % 360
        writer.add_page(page)
    with open(path, "wb") as f:
        writer.write(f)


def convert_ppt_to_pdf(ppt_path: str, pdf_path: str) -> None:
    presentation = Presentation(ppt_path)
    styles = getSampleStyleSheet()
    story = []

    for slide_idx, slide in enumerate(presentation.slides, start=1):
        story.append(Paragraph(f"Slide {slide_idx}", styles["Heading2"]))
        story.append(Spacer(1, 6))
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame and shape.text_frame.text.strip():
                story.append(Paragraph(shape.text_frame.text.replace("\n", "<br/>"), styles["BodyText"]))
                story.append(Spacer(1, 4))
            elif getattr(shape, "has_table", False):
                table_data = []
                for row in shape.table.rows:
                    table_data.append([cell.text for cell in row.cells])
                if table_data:
                    table = Table(table_data, repeatRows=1)
                    table.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FDF2E9")),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 6))
        story.append(PageBreak())

    if not story:
        story.append(Paragraph("Presentation was empty.", styles["BodyText"]))

    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(LETTER), leftMargin=36, rightMargin=36, topMargin=40, bottomMargin=36)
    doc.build(story)


def render_instance_with_intermediate_artifacts(
    *,
    dataset_root: str,
    instance_name: str,
    level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates XLSX and PPT assets, converts them into rotated PDFs, and enforces coverage.
    """
    dataset_root = os.path.abspath(dataset_root)
    _, instance_data = _load_instance_data(dataset_root, instance_name, level)

    samples: List[Dict[str, Any]] = instance_data.get("dataset", [])
    if not samples:
        raise ValueError("No samples available for rendering.")

    title = instance_data.get("name") or instance_name
    label_map = _label_mapping(instance_data)

    xlsx_indices, ppt_indices = _split_indices(len(samples))
    xlsx_subset = _build_indexed_subset(samples, xlsx_indices)
    ppt_subset = _build_indexed_subset(samples, ppt_indices)

    workbook, xlsx_used = _build_workbook(
        title=title,
        subset=xlsx_subset,
        total_samples=len(samples),
        label_mapping=label_map,
    )

    presentation, ppt_used = _build_presentation(
        title=title,
        subset=ppt_subset if ppt_subset else xlsx_subset,
    )

    renders_root = os.path.join(dataset_root, "renders")
    interim_xlsx_root = os.path.join(renders_root, "interim_xlsx")
    interim_ppt_root = os.path.join(renders_root, "interim_ppt")
    final_root = os.path.join(renders_root, "files")

    def _resolve_interim(base_root: str, tag: str, suffix: str) -> str:
        level_root = os.path.join(base_root, level) if level else base_root
        os.makedirs(level_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = _slug(instance_name)
        return os.path.join(level_root, f"{safe_name}_{tag}_{timestamp}.{suffix}")

    def _final_pdf_path(tag: str) -> str:
        os.makedirs(final_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [_slug(instance_name)]
        if level:
            parts.append(_slug(level))
        parts.extend([timestamp, tag])
        return os.path.join(final_root, f"{'_'.join(parts)}.pdf")

    xlsx_path = _resolve_interim(interim_xlsx_root, "xlsx", "xlsx")
    workbook.save(xlsx_path)

    ppt_path = _resolve_interim(interim_ppt_root, "ppt", "pptx")
    presentation.save(ppt_path)

    xlsx_pdf_path = _final_pdf_path("xlsx")
    ppt_pdf_path = _final_pdf_path("ppt")

    convert_xlsx_to_pdf(xlsx_path, xlsx_pdf_path, rotate=True)
    convert_ppt_to_pdf(ppt_path, ppt_pdf_path)

    all_used = xlsx_used | ppt_used
    missing = set(range(len(samples))) - all_used
    if missing:
        raise ValueError(f"Sample coverage validation failed. Missing indices: {sorted(missing)}")

    return {
        "final_pdfs": [xlsx_pdf_path, ppt_pdf_path],
        "interim_files": {"xlsx": xlsx_path, "ppt": ppt_path},
        "coverage": {
            "xlsx": sorted(xlsx_used),
            "ppt": sorted(ppt_used),
            "total": len(samples),
        },
    }
