#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from data_management.rendering.errors import InvalidTemplateError
from data_management.rendering.trace_logger import RenderingTraceLogger

# New Modular Imports
from data_management.rendering.samples_tracker import SamplesTracker
from data_management.rendering.pdf.block_runtime import parse_block_call
from data_management.rendering.pdf.code_blocks.block_registry import library
# IMPORTANT: Import this to ensure @register_block decorators execute
from data_management.rendering.pdf.code_blocks.general_pdf_blocks import *  # Do not remove this line!

# ReportLab
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tqdm

# Optional LLM wrapper
try:
    from inference.model_wrappers.llm_wrapper import get_llm_wrapper, LLMModel, LLMWrapper
except Exception:
    LLMWrapper = object
    LLMModel = None


    def get_llm_wrapper(*args, **kwargs):
        raise RuntimeError("llm_wrapper.get_llm_wrapper not available")

DEFAULT_PROMPT_MD = os.path.join(os.path.dirname(__file__), "prompts", "block_template_prompt.md")


# ======================
# Utilities
# ======================

def _pagesize(name: str):
    name = (name or "A4").upper()
    return LETTER if name == "LETTER" else A4


def _register_font(font_path: Optional[str]) -> Tuple[str, str]:
    if not font_path:
        return "Helvetica", "Helvetica-Bold"
    base = os.path.splitext(os.path.basename(font_path))[0]
    try:
        pdfmetrics.registerFont(TTFont(base, font_path))
        return base, base
    except Exception as e:
        print(f"⚠️ Failed to register font '{font_path}': {e}. Falling back to Helvetica.")
        return "Helvetica", "Helvetica-Bold"


def _read_prompt_file(md_path: Optional[str]) -> Optional[str]:
    if not md_path: return None
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


# =============================================
# Engine: Orchestrate Layout & Rendering
# =============================================

@dataclass
class ProgrammaticPdfEngine:
    font_path: Optional[str] = None
    page_size: str = "A4"
    left_margin: int = 36
    right_margin: int = 36
    top_margin: int = 42
    bottom_margin: int = 36
    default_prompt_md: str = DEFAULT_PROMPT_MD
    raw_response_tracker: Optional[Any] = None

    def _get_styles(self):
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        base_font, base_bold = _register_font(self.font_path)
        styles = getSampleStyleSheet()

        # 1. Update standard styles
        for key in ["Normal", "BodyText", "Title", "Heading1", "Heading2", "Heading3", "Heading4"]:
            if key in styles:
                styles[key].fontName = base_bold if key.startswith("Heading") or key == "Title" else base_font

        # 2. Define and Add 'Small' explicitly
        # We use styles.has_key() or 'Small' in styles.byName for checking
        if 'Small' not in styles:
            small_style = ParagraphStyle(
                'Small',
                parent=styles['Normal'],
                fontName=base_font,
                fontSize=8,
                leading=10,
                textColor=colors.grey
            )
            styles.add(small_style)

        # 3. Add 'Italic' if missing
        if 'Italic' not in styles:
            styles.add(ParagraphStyle(
                'Italic',
                parent=styles['Normal'],
                fontName=base_font,
                italic=True
            ))

        # Debug check: ensure the style is actually there before returning
        print(f"DEBUG: Styles available: {list(styles.byName.keys())}")

        return styles

    def get_dataset_overview_prompt_snippet(self, samples, max_preview_samples, theme_hint) -> str:
        total_samples = len(samples)
        samples_preview = samples[:max_preview_samples] if max_preview_samples > 0 else []
        preview = [{"label": s.get("label"), "instance_text": (s.get("instance_text") or "")[:300]} for s in
                   samples_preview]

        # Label distribution
        label_counts = {}
        for s in samples:
            lbl = s.get("label", "UNLABELED")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Get label mapping if available
        label_mapping = {}
        for s in samples:
            lbl = s.get("label", "UNLABELED")
            label_txt = s.get("instance_label_text") or {}
            if(label_txt is not None and 'label' in label_txt):
                label_mapping[lbl] = label_txt['label']
        if label_mapping:
            label_mapping_str = "\n".join([f"- {k}: '{v}'" for k, v in label_mapping.items()])
        else:
            label_mapping_str = ""

        # Sort the label counts for better readability
        label_counts = dict(sorted(label_counts.items(), key=lambda item: item[0]))

        # Create a list mapping sample_ids to their labels
        sample_label_str = "\n".join([f"- {i}: {s.get('label', 'UNLABELED')}" for i, s in
                                      enumerate(samples)])

        prompt = (
                "\n---\nDATASET OVERVIEW:\n"
                + f"THEME_HINT: {json.dumps(theme_hint or '')}\n"
                + f"Label distribution: {json.dumps(label_counts, indent=2)}\n"
                + (f"Label mapping:\n{label_mapping_str}\n\n" if label_mapping_str else "")
                + f"Sample Preview: {json.dumps(preview, indent=2)}\n"
                + f"Sample Labels (SampleID: Label):\n{sample_label_str}\n"
                + f"\nIMPORTANT:\n- Total Samples: {total_samples}\n"
                + "- REQUIREMENT: You MUST ensure 100% coverage of all the samples in the file. However, a single sample can be used in multiple blocks if needed.\n"
        )

        return prompt

    # ---- LLM Communication ----
    def prompt_llm_for_layout(
            self,
            llm: LLMWrapper,
            samples: List[Dict[str, Any]],
            *,
            max_preview_samples=0,
            theme_hint: Optional[str],
            layout_prompt_md_path: Optional[str] = None,
            error_feedback: Optional[str] = None,
            previous_response: Optional[str] = None,
            trace_logger: Optional[RenderingTraceLogger] = None,
    ) -> Dict[str, Any]:
        md_path = layout_prompt_md_path or self.default_prompt_md
        print(f"Using layout prompt file: {md_path}")
        base_instructions = _read_prompt_file(md_path)
        if not base_instructions:
            raise FileNotFoundError(f"Required prompt file not found: {md_path}")

        # NEW: Get dynamic documentation from the registry decorators
        ui_components_doc = library.get_prompt_snippet()

        # Dataset overview snippet
        overview_prompt = self.get_dataset_overview_prompt_snippet(
            samples, max_preview_samples, theme_hint
        )

        # Assemble the prompt: Instructions + UI Library + Data Preview
        prompt = (
                base_instructions
                + "\n\n" + ui_components_doc + "\n\n"
                + overview_prompt
        )

        if error_feedback and previous_response:
            prompt += (
                "\n\n### CRITICAL: FIX PREVIOUS ERRORS ###\n"
                f"--- PREVIOUS RESPONSE ---\n{previous_response}\n\n"
                f"--- ERROR LOG ---\n{error_feedback}\n\n"
                "Analyze why the layout failed (e.g. missing samples or invalid block calls) and fix it."
            )

        # input(f"--- Prompt to LLM ---\n{prompt}\n\nPress Enter to send to LLM...")
        raw = llm.make_call(prompt)
        # print(f"--- Raw LLM Response ---\n{raw}\n")
        if self.raw_response_tracker:
            self.raw_response_tracker.response = raw

        if trace_logger:
            trace_logger.log_llm_interaction(stage="pdf_layout_response", prompt=prompt, total_samples=len(samples),
                                             response=raw)

        # Extract JSON
        m = re.search(r"```json\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        jtxt = m.group(1).strip() if m else raw.strip()

        try:
            return json.loads(jtxt)
        except Exception as exc:
            raise InvalidTemplateError("LLM did not return valid JSON for PDF layout.", llm_response=raw) from exc

    # ---- Core Rendering Logic ----

    def render(
            self,
            *,
            samples: List[Dict[str, Any]],
            out_pdf: str,
            layout: Optional[Dict[str, Any]] = None,
            llm_template_model: Optional[str] = LLMModel.GEMINI_3_PRO,
            llm_text_model: Optional[str] = LLMModel.GEMINI_2_5_FLASH,
            theme_hint: Optional[str] = None,
            max_preview_samples: int = 2,
            print_layout: bool = True,
            layout_prompt_md_path: Optional[str] = None,
            error_feedback: Optional[str] = None,
            previous_response: Optional[str] = None,
            trace_logger: Optional[RenderingTraceLogger] = None,
    ) -> Tuple[str, Dict[str, Any]]:

        # 1. Setup Tracker and Models
        tracker = SamplesTracker(samples)

        def _res(m):
            return m if isinstance(m, LLMModel) else next(i for i in LLMModel if i.value == str(m))

        llm_template = get_llm_wrapper(_res(llm_template_model))
        llm_text = get_llm_wrapper(_res(llm_text_model))

        # 2. Ask for the Layout (UI Block calls)
        if layout is None:
            layout = self.prompt_llm_for_layout(
                llm_template,
                samples=samples,
                theme_hint=theme_hint,
                max_preview_samples=max_preview_samples,
                layout_prompt_md_path=layout_prompt_md_path,
                error_feedback=error_feedback, previous_response=previous_response,
                trace_logger=trace_logger,
            )

        theme = {"title": layout.get("title", theme_hint or "Report")}
        blocks = layout.get("blocks") or []

        # 3. Execution
        styles = self._get_styles()
        print(f"🖨️ Starting PDF rendering to '{out_pdf}' with {len(samples)} samples.")
        story: List[Any] = []

        # Get label mapping if available
        label_mapping = {}
        for s in samples:
            lbl = s.get("label", "UNLABELED")
            label_txt = s.get("instance_label_text") or {}
            if(label_txt is not None and 'label' in label_txt):
                label_mapping[lbl] = label_txt['label']

        for call_str in tqdm.tqdm(blocks, desc="Rendering PDF", disable=not print_layout):
            try:
                name, pos, kw = parse_block_call(call_str)
                func = library.registry.get(name)

                if not func:
                    print(f"⚠️ Unknown block: {name}. Skipping.")
                    continue

                # Pass the tracker and the text-synthesis LLM to the block
                func(
                    theme=theme,
                    tracker=tracker,
                    story=story,
                    styles=styles,
                    llm=llm_text,
                    label_mapping=label_mapping,
                    **kw
                )
            except Exception as e:
                # Log error for specific block but try to continue
                print(f"❌ Error in block '{call_str}': {e}")
                if trace_logger:
                    trace_logger.log_error(f"block_render_{name}", e)

        # 4. Final Data Integrity Check
        if not tracker.is_fully_covered():
            missing = tracker.get_missing_indices()
            if trace_logger:
                trace_logger.log_error("pdf_render_missing_samples", f"Missing: {missing}")
            raise InvalidTemplateError(f"Template missing samples: {missing}. Use render_catchall().",
                                       llm_response=layout)

        # 5. Build
        os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or ".", exist_ok=True)
        doc = SimpleDocTemplate(
            out_pdf,
            pagesize=_pagesize(self.page_size),
            leftMargin=self.left_margin, rightMargin=self.right_margin,
            topMargin=self.top_margin, bottomMargin=self.bottom_margin,
            title=theme['title']
        )
        doc.build(story)

        return out_pdf, layout
