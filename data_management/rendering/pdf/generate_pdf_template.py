#!/usr/bin/env python3
"""
Generate creative PDF reports for a DatasetInstance using an LLM-designed layout.

This module defines a PdfTemplate class which can:
  1) Take a list of samples (each with at least 'instance_text' and 'label'),
     an instance title/type, and render them into a single PDF using ReportLab.
  2) Ask an LLM to design a JSON "layout spec" describing sections/paragraphs/tables,
     then render that spec.

Dependencies (pure Python):
  pip install reportlab jinja2

Example usage:

from llm_wrapper import get_llm_wrapper, LLMModel
from generate_pdf_template import PdfTemplate

samples = [
    {"instance_text": "Hello world.", "label": 1, "score": 0.87, "assignment": {"A": 1}},
    {"instance_text": "Second sample.", "label": 0, "score": 0.12, "assignment": {"A": 0}},
]

pdf = PdfTemplate(font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
# Let the LLM design a creative layout spec
llm = get_llm_wrapper(LLMModel.GPT_5)
spec = pdf.generate_spec_with_llm(
    llm=llm,
    instance_title="Crime Investigation Case Brief",
    style_hint=(
        "Use a strong title, a short intro paragraph, a summary table of label counts, "
        "and then a per-sample section with the text as a paragraph and the label emphasized."
    ),
    samples=samples,
)
# Render
pdf.render(samples=samples, instance_title="Crime Investigation Case Brief", out_pdf="creative.pdf", spec=spec)

# Or skip LLM and use the default minimalist spec
pdf.render(samples=samples, instance_title="Case Brief", out_pdf="minimal.pdf")

"""
from __future__ import annotations

import json
import os
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ReportLab (pure-Python)
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Optional: LLM wrapper interface
from inference.model_wrappers.llm_wrapper import LLMWrapper


# ------------------
# Utility functions
# ------------------

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unnamed"


def _pagesize(name: str):
    name = (name or "A4").upper()
    if name == "LETTER":
        return LETTER
    return A4


def _register_font(font_path: Optional[str]) -> Tuple[str, str]:
    """Register a TTF font (for Unicode). Returns (regular_font_name, bold_font_name)."""
    if not font_path:
        return "Helvetica", "Helvetica-Bold"
    base = os.path.splitext(os.path.basename(font_path))[0]
    try:
        pdfmetrics.registerFont(TTFont(base, font_path))
        return base, base
    except Exception as e:
        print(f"⚠️ Failed to register font '{font_path}': {e}. Falling back to Helvetica.")
        return "Helvetica", "Helvetica-Bold"


def _kv_table(d: Dict[str, Any], styles) -> Table:
    # (Kept for potential global metadata; NOT used for per-sample content anymore.)
    rows = [("Key", "Value")] + [(str(k), str(v)) for k, v in d.items()]
    t = Table(rows, hAlign="LEFT", colWidths=[140, None])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), styles["Heading4"].fontName),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
            ]
        )
    )
    return t


# ------------------------------------
# LLM layout spec (JSON) specification
# ------------------------------------
SCHEMA_DOC = r"""
You are designing a JSON layout spec for rendering a PDF report.
Return ONLY valid JSON (no markdown fences) matching this schema:
{
  "title": "<string>",                // required, creative report title (NO placeholders)
  "blocks": [                         // ordered content blocks
    {
      "type": "paragraph",
      "text": "<string with {placeholders}>"  // Allowed placeholders (global only): {instance_title}, {date}, {num_samples}
    },
    {
      "type": "labels_table",         // optional summary table of label counts
      "title": "<string>"
    },
    {
      "type": "per_sample",           // repeats the nested blocks once per sample
      "title": "<string>",            // optional, may include {index}, {label}
      "blocks": [
        { "type": "paragraph", "text": "<string with {index} {label} {instance_text}>" }
      ]
    }
  ]
}
Hard constraints:
- PER-SAMPLE CONTENT MUST ONLY USE the sample's text and label ({instance_text}, {label}, {index}).
- DO NOT include or mention scores, assignments/variables, or category mappings anywhere.
- The title MUST NOT contain placeholders like {instance_title} or any {...}.
- Be compact and readable for potentially hundreds of samples.
- You MAY add creative global paragraphs/sections ("fluff") to frame the report.
"""


# --------------
# PdfTemplate API
# --------------
@dataclass
class PdfTemplate:
    font_path: Optional[str] = None
    page_size: str = "A4"
    left_margin: int = 36
    right_margin: int = 36
    top_margin: int = 42
    bottom_margin: int = 36

    # -------------------------
    # Public: render entrypoint
    # -------------------------
    def render(
        self,
        *,
        samples: List[Dict[str, Any]],
        instance_title: str,
        out_pdf: str,
        spec: Optional[Dict[str, Any]] = None,
        footer: Optional[str] = None,
        print_layout: bool = True,
    ) -> str:
        """Render the given samples using the provided (or default) spec into a PDF file.
        Returns the output path.
        """
        if spec is None:
            spec = self.default_spec(instance_title)
        self._validate_spec(spec)

        # Sanitize the spec (privacy + per-sample constraints + title cleanup)
        spec = self._sanitize_spec(spec, instance_title)

        if print_layout:
            self.print_spec_layout(spec, instance_title=instance_title, num_samples=len(samples))

        # font + styles
        base_font, base_bold = _register_font(self.font_path)
        styles = getSampleStyleSheet()
        # Set font names
        for key in ["Normal", "BodyText", "Title", "Heading1", "Heading2", "Heading3", "Heading4"]:
            if key in styles:
                styles[key].fontName = base_bold if key.startswith("Heading") or key == "Title" else base_font
        # Avoid duplicate style errors
        if "Italic" not in styles.byName:
            styles.add(ParagraphStyle(name="Italic", parent=styles["Normal"], fontName=base_font, italic=True))
        if "Bold" not in styles.byName:
            styles.add(ParagraphStyle(name="Bold", parent=styles["Normal"], fontName=base_bold))

        # Build story from spec
        story = self._build_story_from_spec(spec, samples, styles, instance_title)

        # Output
        pagesize = _pagesize(self.page_size)
        os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or ".", exist_ok=True)
        doc = SimpleDocTemplate(
            out_pdf,
            pagesize=pagesize,
            leftMargin=self.left_margin,
            rightMargin=self.right_margin,
            topMargin=self.top_margin,
            bottomMargin=self.bottom_margin,
        )
        # Footer behavior:
        # - None  -> no left footer text (and we won't draw page number either)
        # - ""    -> show only page number
        # - text  -> show text on left + page number on right
        footer_text = footer  # do not auto-fill with instance_title anymore
        doc.build(
            story,
            onFirstPage=lambda c, d: self._on_page(c, d, footer_text),
            onLaterPages=lambda c, d: self._on_page(c, d, footer_text),
        )
        print(f"✅ Wrote PDF with {len(samples)} samples → {out_pdf}")
        return out_pdf

    # --------------------------------------
    # Public: ask an LLM to design the spec
    # --------------------------------------
    def generate_spec_with_llm(
        self,
        *,
        llm: LLMWrapper,
        instance_title: str,
        style_hint: Optional[str],
        samples: List[Dict[str, Any]],
        max_preview_samples: int = 8,
        error_feedback: Optional[str] = None,
        previous_response: Optional[str] = None,  # Capture the actual text returned last time
    ) -> Dict[str, Any]:
        """Ask the LLM to design a JSON layout spec. Returns a dict.
        The prompt includes a small preview of the data and SCHEMA_DOC.
        """
        preview = self._make_preview(samples, max_preview_samples)
        prompt = (
                "Design a JSON layout spec for a PDF report.\n\n"
                + SCHEMA_DOC + "\n\n"
                + f"INSTANCE_TITLE: {instance_title}\n"
                + (f"STYLE_HINT: {style_hint}\n" if style_hint else "")
                + (
                    "\nTitle rules:\n"
                    "- Invent a vivid, original report title.\n"
                    "- Do NOT include the literal text '{instance_title}' in the title.\n"
                    "- Avoid curly-brace placeholders in the title.\n"
                )
                + "DATA_PREVIEW (list of samples):\n"
                + json.dumps(preview, ensure_ascii=False, indent=2)
        )

        if error_feedback and previous_response:
            prompt += (
                "\n\n### CRITICAL: FIX PREVIOUS ERRORS ###\n"
                "Your previous output was invalid and caused a system crash. "
                "Below is your PREVIOUS RESPONSE and the resulting ERROR LOG.\n\n"
                f"--- YOUR PREVIOUS RESPONSE ---\n{previous_response}\n\n"
                f"--- ERROR LOG ---\n{error_feedback}\n\n"
                "Analyze why the engine failed to parse your response. "
            )

        text = llm.make_call(prompt)
        # Try to parse JSON; if the model wrapped in fences, strip them
        cleaned = self._strip_fences(text)
        try:
            spec = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"LLM did not return valid JSON. Raw text was:\n{text}") from e
        self._validate_spec(spec)
        # Final safety pass for title/placeholders/blocks
        spec = self._sanitize_spec(spec, instance_title)
        return spec

    # --------------------
    # Spec helper methods
    # --------------------
    def default_spec(self, instance_title: str) -> Dict[str, Any]:
        """A minimal, readable default spec: a title, a tiny intro paragraph,
        then a per-sample paragraph with label emphasized and instance_text.
        """
        return {
            "title": "Signals in the Noise — A Curated Reader",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "Report generated on {date}. Total samples: {num_samples}. "
                        "Each sample below shows the label and the associated text."
                    ),
                },
                {"type": "labels_table", "title": "Label distribution"},
                {
                    "type": "per_sample",
                    "title": "Sample #{index} — label: {label}",
                    "blocks": [
                        {"type": "paragraph", "text": "{instance_text}"},
                    ],
                },
            ],
        }

    def _validate_spec(self, spec: Dict[str, Any]) -> None:
        if not isinstance(spec, dict):
            raise ValueError("Spec must be a dict")
        if "title" not in spec or "blocks" not in spec:
            raise ValueError("Spec must have 'title' and 'blocks'")
        if not isinstance(spec["blocks"], list):
            raise ValueError("'blocks' must be a list")

    def _strip_fences(self, text: str) -> str:
        # Remove ```json ... ``` fences if present
        m = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        return (m.group(1).strip() if m else text.strip())

    def _make_preview(self, samples: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in samples[:k]:
            out.append({
                "label": s.get("label"),
                # NO score/assignment in preview
                "instance_text": (s.get("instance_text") or "")[:280],
            })
        return out

    # ------------------------
    # Story-building from spec
    # ------------------------
    def _format_allowed(self, text: str, mapping: Dict[str, Any], allowed: List[str]) -> str:
        """Replace placeholders not in `allowed` with empty string, then format."""
        def repl(m):
            key = m.group(1)
            return "{" + key + "}" if key in allowed else ""
        cleaned = re.sub(r"\{([^}]+)\}", repl, text or "")
        try:
            return cleaned.format(**mapping)
        except Exception:
            return cleaned

    def _pick_creative_title(self, instance_title: str) -> str:
        """Deterministically pick a creative fallback title (no placeholders)."""
        options = [
            "Signals in the Noise — A Curated Reader",
            "Patterns & Parables — A Sampled Anthology",
            "Field Notes from the Dataset",
            "Fragments & Findings — An Interpretive Report",
            "Pages from a Synthetic Casebook",
        ]
        h = int(hashlib.md5(instance_title.encode("utf-8")).hexdigest(), 16)
        return options[h % len(options)]

    def _fix_title(self, title: Optional[str], instance_title: str) -> str:
        """Ensure title has no braces/placeholders and isn't trivially generic."""
        t = (title or "").strip()
        # strip all {...}
        t = re.sub(r"\{[^}]*\}", "", t).strip(" -:—")
        # If empty or too generic, choose a creative deterministic fallback
        if not t or re.fullmatch(r"(?i)(report|overview|analysis|summary)", t):
            return self._pick_creative_title(instance_title)
        return t

    def _sanitize_spec(self, spec: Dict[str, Any], instance_title: str) -> Dict[str, Any]:
        """Enforce constraints:
        - Title is creative and contains no placeholders.
        - Per-sample blocks may only contain paragraphs using {index},{label},{instance_text}.
        - Remove any assignment/score references and unknown per-sample blocks.
        """
        out: Dict[str, Any] = {"title": self._fix_title(spec.get("title"), instance_title), "blocks": []}

        for blk in spec.get("blocks", []):
            btype = (blk.get("type") or "").lower()

            if btype == "paragraph":
                # Global paragraphs: allow {instance_title},{date},{num_samples}
                txt = str(blk.get("text") or "")
                # Nuke any references to {score} or 'assignment'
                txt = txt.replace("{score}", "")
                if "assignment" in txt.lower():
                    txt = re.sub(r"assignment[s]?:?\\s*", "", txt, flags=re.IGNORECASE)
                out["blocks"].append({"type": "paragraph", "text": txt})

            elif btype == "labels_table":
                out["blocks"].append({"type": "labels_table", "title": blk.get("title") or "Label distribution"})

            elif btype == "per_sample":
                inner_clean: List[Dict[str, Any]] = []
                for ib in blk.get("blocks") or []:
                    it = (ib.get("type") or "").lower()
                    if it == "paragraph":
                        txt = str(ib.get("text") or "")
                        # strictly forbid score/assignments/mappings
                        for bad in ("{score}", "score", "assignment", "variable", "mapping"):
                            txt = re.sub(bad, "", txt, flags=re.IGNORECASE)
                        inner_clean.append({"type": "paragraph", "text": txt})
                    # ignore all other inner block types
                clean_blk: Dict[str, Any] = {"type": "per_sample", "blocks": inner_clean}
                if blk.get("title"):
                    # Allow {index},{label} in the per-sample heading
                    ttl = str(blk.get("title"))
                    for bad in ("{score}", "score", "assignment", "variable", "mapping"):
                        ttl = re.sub(bad, "", ttl, flags=re.IGNORECASE)
                    clean_blk["title"] = ttl
                out["blocks"].append(clean_blk)

            # ignore any other top-level block types

        return out

    def _build_story_from_spec(self, spec: Dict[str, Any], samples: List[Dict[str, Any]], styles, instance_title: str):
        story: List[Any] = []
        # Title
        story.append(Paragraph(spec.get("title") or instance_title, styles["Title"]))
        story.append(Spacer(1, 8))

        # Context for global placeholders (allowed only here)
        global_ctx = {
            "instance_title": instance_title,
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
            "num_samples": len(samples),
        }

        for blk in spec.get("blocks", []):
            btype = (blk.get("type") or "").lower()
            if btype == "paragraph":
                txt = self._format_allowed(blk.get("text") or "", global_ctx, ["instance_title", "date", "num_samples"])
                story.append(Paragraph(txt.replace("\n", "<br/>"), styles["BodyText"]))
                story.append(Spacer(1, 6))

            elif btype == "labels_table":
                title = blk.get("title") or "Labels"
                story.append(Paragraph(title, styles["Heading3"]))
                counts = Counter([s.get("label") for s in samples])
                rows = [("Label", "Count")]
                for k in sorted(counts):
                    rows.append((str(k), str(counts[k])))
                t = Table(rows, hAlign="LEFT", colWidths=[120, 100])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ("ALIGN", (1,1), (-1,-1), "RIGHT"),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))

            elif btype == "per_sample":
                title_tmpl = blk.get("title")
                inner = blk.get("blocks") or []
                for i, s in enumerate(samples):
                    sample_ctx = {
                        "index": i + 1,
                        "label": s.get("label"),
                        "instance_text": s.get("instance_text", ""),
                    }
                    if title_tmpl:
                        heading = self._format_allowed(title_tmpl, sample_ctx, ["index", "label"])
                        story.append(Paragraph(heading, styles["Heading4"]))
                        story.append(Spacer(1, 4))
                    for ib in inner:
                        txt = self._format_allowed(ib.get("text") or "", sample_ctx, ["index", "label", "instance_text"])
                        story.append(Paragraph(txt.replace("\n", "<br/>"), styles["BodyText"]))
                    story.append(Spacer(1, 8))

        return story

    # --------------
    # Page callbacks
    # --------------
    def _on_page(self, canvas, doc, footer_text: Optional[str]):
        canvas.saveState()
        w, h = doc.pagesize
        canvas.setFont("Helvetica", 9)
        canvas.setFillGray(0.5)
        # Only draw page number if footer_text == "" (explicit) or draw both if text provided.
        if footer_text is not None:
            if footer_text:
                canvas.drawString(36, 20, footer_text)
                canvas.drawRightString(w - 36, 20, f"Page {doc.page}")
            else:
                # Only page number
                canvas.drawRightString(w - 36, 20, f"Page {doc.page}")
        # If footer_text is None: draw nothing (no page number)
        canvas.restoreState()

    def print_spec_layout(self, spec: Dict[str, Any], *, instance_title: str, num_samples: int) -> None:
        """Pretty-print a concise outline of the spec to stdout."""
        print("\n===== PDF TEMPLATE LAYOUT =====")
        title = spec.get("title") or instance_title
        print(f"Title: {title}")
        print(f"Samples: {num_samples}")
        print("Blocks:")
        for i, blk in enumerate(spec.get("blocks", []), 1):
            btype = (blk.get("type") or "").lower()
            line = f"  {i:02d}. {btype}"
            if btype in {"paragraph", "per_sample", "labels_table"}:
                extra = []
                if btype == "paragraph" and blk.get("text"):
                    extra.append("text")
                if btype == "labels_table" and blk.get("title"):
                    extra.append(f"title='{blk.get('title')}'")
                if btype == "per_sample" and blk.get("title"):
                    extra.append(f"title='{blk.get('title')}'")
                if extra:
                    line += " (" + ", ".join(extra) + ")"
            print(line)
            if btype == "per_sample":
                inner = blk.get("blocks") or []
                for j, ib in enumerate(inner, 1):
                    it = (ib.get("type") or "").lower()
                    ix = f"      - {j}. {it}"
                    if it == "paragraph" and ib.get("text"):
                        ix += " (text)"
                    print(ix)
        print("===== END TEMPLATE LAYOUT =====\n")
