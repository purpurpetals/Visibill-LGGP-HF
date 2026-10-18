"""
Expense Automator — Streamlit Frontend
HackForge '26 · LGGP-HF

End-to-end pipeline:
  st.file_uploader
      ↓  np.frombuffer / cv2.imdecode
  extractor.extract_full(cv_array)   ← Vision/OCR zone
      ↓  ocr_dict
  ReceiptAuditEngine.audit(ocr_dict) ← Policy/Audit zone
      ↓  AuditResult
  render_audit_result()               ← UI unpacking
"""

import hashlib
import io
import sys
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pyzbar import pyzbar

# ─────────────────────────────────────────────
# MODULE IMPORTS  (resolve sibling paths)
# ─────────────────────────────────────────────
# extractor.py lives in LGGP-HF/extractor.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_LGGP = os.path.join(_HERE, "LGGP-HF")
if _LGGP not in sys.path:
    sys.path.insert(0, _LGGP)

from extractor import extract_full          # noqa: E402
from engine import ReceiptAuditEngine       # noqa: E402

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Automator",
    page_icon="🧾",
    layout="wide",
)

# ─────────────────────────────────────────────
# RESPONSIVE CSS  (phone ≤ 768 px  /  desktop)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── shared tokens ─────────────────────────── */
    :root {
        --radius: 10px;
        --shadow: 0 2px 12px rgba(0,0,0,.15);
    }

    /* ── desktop baseline ──────────────────────── */
    .upload-card {
        padding: 2rem;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
    }
    .stButton > button {
        border-radius: 8px;
        font-size: 1rem;
        padding: .55rem 1.4rem;
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(0,0,0,.2);
    }
    /* primary submit button */
    .stButton > button[kind="primary"] {
        width: 100%;
        font-size: 1.05rem;
        padding: .7rem 0;
        letter-spacing: .03em;
    }

    /* ── Extracted field cards ──────────────────── */
    .field-card {
        border-radius: 8px;
        padding: .75rem 1rem;
        margin-bottom: .5rem;
        background: rgba(255,255,255,.04);
        border-left: 4px solid transparent;
    }
    .field-card.review {
        border-left: 4px solid #EAB308;
        background: rgba(234,179,8,.08);
    }

    /* ── phone overrides (≤ 768 px) ────────────── */
    @media screen and (max-width: 768px) {
        /* stack Streamlit columns vertically */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* larger tap targets */
        .stButton > button {
            font-size: 1.15rem !important;
            padding: .85rem 1rem !important;
            width: 100% !important;
        }
        /* uploader area bigger on touch screens */
        [data-testid="stFileUploader"] {
            min-height: 140px;
        }
        /* headings scale down */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }
        /* reduce sidebar width on mobile */
        [data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 220px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# GLOBAL SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
if "approved_receipts" not in st.session_state:
    st.session_state.approved_receipts = []   # list[str] — filenames

if "rejected_receipts" not in st.session_state:
    st.session_state.rejected_receipts = []   # list[str] — filenames

if "pending_reviews" not in st.session_state:
    # Each entry: {"image_bytes": bytes, "filename": str,
    #              "audit_result": dict, "claim_code": str}
    st.session_state.pending_reviews = []

if "claim_status_db" not in st.session_state:
    # Maps claim_code -> {filename, status, submitted_at}
    # status: "Pending Review" | "Approved" | "Rejected" | "Auto-Approved" | "Denied"
    st.session_state.claim_status_db = {}

# ─────────────────────────────────────────────
# DUMMY DATABASE  (Pandas DataFrame)
# ─────────────────────────────────────────────
SEEN_RECEIPT_DB: pd.DataFrame = pd.DataFrame(
    {
        "id_hash": [
            "abc123_placeholder_hash_already_seen",
            "def456_placeholder_hash_already_seen",
        ]
    }
)

CLAIM_CODE_PREFIX = "EXP"

# Shared audit engine instance (stateless — safe to reuse across reruns)
_AUDIT_ENGINE = ReceiptAuditEngine(confidence_threshold=60.0)


# ─────────────────────────────────────────────
# PIPELINE HELPERS
# ─────────────────────────────────────────────

def uploaded_file_to_cv2(uploaded_file) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile byte buffer into an OpenCV BGR matrix.

    1. Read raw bytes from the UploadedFile buffer.
    2. np.frombuffer  → 1-D uint8 array (zero-copy view).
    3. cv2.imdecode   → H×W×3 BGR numpy array.
    """
    raw_bytes = uploaded_file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(
            f"cv2.imdecode could not decode '{uploaded_file.name}'. "
            "Ensure the file is a valid JPEG/PNG image."
        )
    uploaded_file.seek(0)  # reset so PIL can re-read if needed
    return bgr


def run_pipeline(cv2_image: np.ndarray) -> dict:
    """
    Run the complete Vision → Audit pipeline on a cv2 BGR matrix.

    Flow:
      cv2_image (np.ndarray BGR)
        → cv2.imencode → bytes
        → extractor.extract_full()   {payload, ocr_dict}
        → ReceiptAuditEngine.audit() AuditResult
        → model_dump()               plain dict

    Returns the AuditResult as a plain dict, with the extractor metadata
    stashed under key '_extractor_payload' for UI display.
    """
    # Re-encode BGR matrix to bytes so extractor._load_image can accept it
    success, encoded = cv2.imencode(".jpg", cv2_image)
    if not success:
        raise RuntimeError("cv2.imencode failed — cannot convert image to bytes.")
    img_bytes = encoded.tobytes()

    # Extraction (preprocessing + OCR)
    full_result = extract_full(img_bytes)   # {"payload": {...}, "ocr_dict": {...}}
    payload  = full_result["payload"]
    ocr_dict = full_result["ocr_dict"]

    # Policy audit
    audit_result = _AUDIT_ENGINE.audit(ocr_dict)

    result_dict = audit_result.model_dump()
    result_dict["_extractor_payload"] = payload   # stash for UI metadata
    return result_dict


def simulate_approve_notification(receipt_filename: str) -> None:
    """Push approval into session state so the Employee view can display it."""
    if receipt_filename not in st.session_state.approved_receipts:
        st.session_state.approved_receipts.append(receipt_filename)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def hash_payload(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_duplicate(id_hash: str, db: pd.DataFrame) -> bool:
    return id_hash in db["id_hash"].values


def generate_claim_code() -> str:
    """Return a unique, human-readable claim code: EXP-XXXXXXXX."""
    import random, string
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=8))
    code = f"{CLAIM_CODE_PREFIX}-{suffix}"
    # Collision guard (extremely unlikely but safe)
    while code in st.session_state.claim_status_db:
        suffix = "".join(random.choices(chars, k=8))
        code = f"{CLAIM_CODE_PREFIX}-{suffix}"
    return code


def pil_to_jpg_bytes(pil_image: Image.Image) -> bytes:
    """Convert any PIL image to JPEG bytes (strips alpha channel if present)."""
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return buf.getvalue()


def decode_barcodes(pil_image: Image.Image) -> list:
    """Return list of pyzbar decoded objects from the image."""
    return pyzbar.decode(pil_image)


def confidence_style(value: str, confidence: float) -> str:
    """Return an HTML span with colour corresponding to confidence band."""
    if confidence < 0.60:
        bg, fg = "#EF4444", "#fff"   # red
    elif confidence < 0.80:
        bg, fg = "#EAB308", "#000"   # yellow
    else:
        bg, fg = "#22C55E", "#fff"   # green
    return (
        f'<span style="background:{bg};color:{fg};'
        f'padding:2px 8px;border-radius:5px;font-weight:600;">'
        f"{value}</span>"
    )


def confidence_rgb(confidence: float) -> tuple:
    """Return an (R, G, B) tuple matching the confidence band."""
    if confidence < 0.60:
        return (239, 68, 68)    # red
    elif confidence < 0.80:
        return (234, 179, 8)    # yellow
    else:
        return (34, 197, 94)    # green


# ─────────────────────────────────────────────
# UI UNPACKING: render_audit_result
# ─────────────────────────────────────────────

def render_audit_result(audit_result: dict, pil_img: Image.Image) -> None:
    """
    Unpack an AuditResult dict and render the full results UI:

    • Green / Red / Yellow alert banner (APPROVED / REJECTED / FLAGGED)
    • 2-column layout: receipt image  |  extracted fields
    • Yellow warning on any field with requires_review: True
    • Bulleted list of all policy violation reasons
    """
    status          = audit_result.get("status", "FLAGGED_FOR_REVIEW")
    verdict_reasons = audit_result.get("verdict_reasons", [])
    conf_flags      = audit_result.get("confidence_flags", {})
    extracted       = audit_result.get("extracted_data", {})
    math_ok         = audit_result.get("math_verified", True)
    extractor_meta  = audit_result.get("_extractor_payload", {}).get("metadata", {})

    # ── Status banner ─────────────────────────────────────────────────────
    if status == "APPROVED":
        st.success("✅ **APPROVED** — Receipt passed all policy checks.")
    elif status == "REJECTED":
        st.error("❌ **REJECTED** — Receipt violates one or more hard policies.")
    else:
        st.warning("⚠️ **FLAGGED FOR REVIEW** — Receipt requires manual inspection.")

    st.markdown("---")

    # ── 2-column layout ───────────────────────────────────────────────────
    col_img, col_fields = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("📄 Receipt Image")
        st.image(pil_img, use_container_width=True)
        ocr_avg = extractor_meta.get("ocr_confidence_avg", None)
        if ocr_avg is not None:
            st.caption(f"OCR avg confidence: **{ocr_avg:.1f}%**")
        if extractor_meta.get("requires_review"):
            st.warning("⚠️ Low OCR confidence — manual verification recommended.")

    with col_fields:
        st.subheader("🔍 Extracted Fields")

        def _field_card(label: str, value, conf_flag: dict | None = None) -> None:
            """Render one field card, highlighted yellow if requires_review."""
            requires_review = conf_flag.get("requires_review", False) if conf_flag else False
            card_class = "field-card review" if requires_review else "field-card"
            review_badge = (
                ' &nbsp;<span style="background:#EAB308;color:#000;'
                'padding:1px 6px;border-radius:4px;font-size:.75rem;">'
                '⚠ review</span>'
                if requires_review else ""
            )
            if conf_flag:
                raw_conf   = conf_flag.get("conf", 0.0)
                # Engine stores conf in 0–100 OCR-token range
                conf_norm  = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
                color_span = confidence_style(str(value), conf_norm)
                conf_pct   = f"<small style='color:grey;'>confidence: {raw_conf:.0f}%</small><br>"
            else:
                color_span = f"<strong>{value}</strong>"
                conf_pct   = ""

            st.markdown(
                f'<div class="{card_class}">'
                f"<strong>{label}</strong>{review_badge}<br>"
                f"{conf_pct}"
                f"{color_span}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Total Amount
        _field_card("Total Amount",
                    f"${extracted.get('total_amount', 0.0):.2f}",
                    conf_flags.get("total_amount"))
        # Date
        _field_card("Date",
                    extracted.get("date", "—") or "—",
                    conf_flags.get("date"))
        # Vendor
        _field_card("Vendor",
                    extracted.get("vendor", "—") or "—",
                    conf_flags.get("vendor"))
        # Category (derived — no per-field confidence flag)
        _field_card("Category", extracted.get("category", "—") or "—")

        # Math integrity
        if not math_ok:
            st.markdown(
                '<div class="field-card review">'
                "<strong>Math Integrity</strong> &nbsp;"
                '<span style="background:#EF4444;color:#fff;padding:1px 6px;'
                'border-radius:4px;font-size:.75rem;">✗ mismatch</span><br>'
                f"Subtotal: ${extracted.get('subtotal') or 0:.2f} + "
                f"Tax: ${extracted.get('tax') or 0:.2f} + "
                f"Tip: ${extracted.get('tip') or 0:.2f}"
                "</div>",
                unsafe_allow_html=True,
            )

        # Line Items
        line_items = extracted.get("line_items", [])
        if line_items:
            st.markdown("**Line Items**")
            rows = [
                {"Item": li.get("name", ""), "Price": f"${li.get('price', 0):.2f}"}
                for li in line_items
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Policy violations ─────────────────────────────────────────────────
    if verdict_reasons:
        st.markdown("---")
        st.subheader("📋 Policy & Audit Notes")
        for reason in verdict_reasons:
            st.markdown(f"- {reason}")

    # ── Raw audit JSON (collapsible) ──────────────────────────────────────
    with st.expander("🔧 View raw audit payload"):
        safe = {k: v for k, v in audit_result.items() if k != "_extractor_payload"}
        st.json(safe)


def build_projection_image(pil_image: Image.Image, extraction: dict) -> Image.Image:
    """
    Return an annotated copy of the receipt image.

    For every extracted field a confidence-coloured ellipse is drawn at an
    evenly spaced position along the right edge of the image.  An arrow line
    connects each annotation to a representative row of the receipt.
    Colour bands: green ≥ 80 %, yellow 60–79 %, red < 60 %.
    """
    # ── Prepare canvas ────────────────────────────────────────────────────
    img = pil_image.copy().convert("RGB")

    # Cap width so annotations remain proportional
    MAX_W = 900
    if img.width > MAX_W:
        scale = MAX_W / img.width
        img = img.resize((MAX_W, int(img.height * scale)), Image.LANCZOS)

    # Widen canvas to the right to give annotations unobstructed space
    W, H = img.size
    ANNOT_W = 220           # pixels reserved for annotation column
    canvas = Image.new("RGB", (W + ANNOT_W, H), (30, 30, 30))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")

    # ── Font (graceful fallback) ──────────────────────────────────────────
    try:
        font_label = ImageFont.truetype("arial.ttf", 15)
        font_value = ImageFont.truetype("arial.ttf", 13)
        font_conf  = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font_label = ImageFont.load_default()
        font_value = font_label
        font_conf  = font_label

    # ── Gather fields ─────────────────────────────────────────────────────
    fields = [
        (f, extraction[f])
        for f in DISPLAY_FIELDS
        if f in extraction
    ]
    n = len(fields)
    if n == 0:
        return canvas

    # ── Layout constants ──────────────────────────────────────────────────
    BOX_H      = 68
    BOX_W      = ANNOT_W - 20
    PADDING    = 18
    # Annotation centres along the right strip
    total_span = H - 2 * PADDING
    step       = total_span / max(n - 1, 1)

    for i, (field, field_data) in enumerate(fields):
        value      = str(field_data.get("value", "N/A"))
        conf       = float(field_data.get("confidence", 1.0))
        color      = confidence_rgb(conf)
        label      = field.replace("_", " ").title()
        conf_pct   = f"{conf * 100:.0f}%"

        # Centre of this annotation in the right strip
        cx = W + ANNOT_W // 2
        cy = PADDING + int(i * step) if n > 1 else H // 2

        # Ellipse bounds
        ex0 = W + 10
        ey0 = cy - BOX_H // 2
        ex1 = W + 10 + BOX_W
        ey1 = cy + BOX_H // 2

        # ── Semi-transparent fill, solid border ──
        draw.ellipse([ex0, ey0, ex1, ey1],
                     fill=(*color, 55), outline=(*color, 240), width=3)

        # ── Arrow: from image edge to ellipse ────
        arrow_start = (W - 1, cy)
        arrow_end   = (ex0 + 2, cy)
        draw.line([arrow_start, arrow_end], fill=(*color, 200), width=2)
        # Arrowhead (small triangle pointing right → left at ellipse edge)
        ah = 6
        draw.polygon([
            (ex0 + 2, cy),
            (ex0 + 2 + ah, cy - ah // 2),
            (ex0 + 2 + ah, cy + ah // 2),
        ], fill=(*color, 220))

        # ── Text inside ellipse ───────────────────
        text_x = ex0 + 12
        draw.text((text_x, ey0 + 6),  label[:22],          fill=(255, 255, 255), font=font_label)
        draw.text((text_x, ey0 + 24), value[:24],           fill=(240, 240, 240), font=font_value)
        draw.text((text_x, ey0 + 42), f"confidence: {conf_pct}", fill=color,          font=font_conf)

    return canvas


# ─────────────────────────────────────────────
# EMPLOYEE VIEW
# ─────────────────────────────────────────────

def render_employee_view() -> None:
    st.header("📤 Submit a Receipt")

    # ── Status notification banners ───────────────────────────────────────
    if st.session_state.approved_receipts:
        for fname in st.session_state.approved_receipts:
            st.success(f"✅ Approved: your receipt **{fname}** was approved.")

    if st.session_state.rejected_receipts:
        for fname in st.session_state.rejected_receipts:
            st.error(f"❌ Rejected: your receipt **{fname}** was rejected by the accountant.")

    st.markdown("---")

    # ── Claim Status Lookup ───────────────────────────────────────────────
    with st.expander("🔍 Check claim status", expanded=False):
        lookup_code = st.text_input(
            "Enter your Claim Code (e.g. EXP-AB12CD34)",
            key="claim_lookup_input",
            placeholder="EXP-XXXXXXXX",
        ).strip().upper()
        if st.button("Look up", key="claim_lookup_btn"):
            db = st.session_state.claim_status_db
            if not lookup_code:
                st.warning("Please enter a claim code.")
            elif lookup_code not in db:
                st.error("❌ Claim code not found. Double-check and try again.")
            else:
                record = db[lookup_code]
                _status = record["status"]
                _fname  = record["filename"]
                _time   = record["submitted_at"]
                st.markdown(
                    f"""
                    | Field | Detail |
                    |---|---|
                    | **Claim Code** | `{lookup_code}` |
                    | **File** | {_fname} |
                    | **Submitted** | {_time} |
                    | **Status** | {_status} |
                    """
                )
                if _status == "Approved" or _status == "Auto-Approved":
                    st.success(f"✅ {_status}")
                elif _status == "Rejected" or _status == "Denied":
                    st.error(f"❌ {_status}")
                else:
                    st.info("⏳ Still under review — check back soon.")

    st.markdown("---")

    # ── Upload ─────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload Receipt (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        key="employee_uploader",
    )

    if uploaded_file is None:
        st.info("Please upload a receipt image to begin.")
        return

    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Receipt", use_container_width=True)

    # ── Submit gate ────────────────────────────────────────────────────────
    if not st.button("Submit Receipt", key="submit_receipt_btn", type="primary"):
        return

    # ── Automated ID Check (QR / Barcode) ─────────────────────────────────
    with st.spinner("Scanning for receipt identifier (QR / Barcode)…"):
        decoded_symbols = decode_barcodes(pil_image)

    original_filename = uploaded_file.name
    jpg_bytes = pil_to_jpg_bytes(pil_image)
    uploaded_file.seek(0)   # reset buffer for cv2 decode later

    if decoded_symbols:
        symbol_data: bytes = decoded_symbols[0].data
        id_hash = hash_payload(symbol_data)
        st.caption(f"🔍 Identifier found — Hash: `{id_hash[:16]}…`")
        if is_duplicate(id_hash, SEEN_RECEIPT_DB):
            st.error("⛔ Duplicate Receipt Detected.")
            return
        else:
            st.info("Identifier found and verified as unique. Forwarding to extraction…")
            _forward_to_accountant(jpg_bytes, original_filename, uploaded_file)
    else:
        st.info("No barcode / QR code found. Forwarding image to extraction…")
        _forward_to_accountant(jpg_bytes, original_filename, uploaded_file)


def _forward_to_accountant(jpg_bytes: bytes, filename: str, uploaded_file) -> None:
    """
    1. Convert the UploadedFile → cv2 BGR matrix (np.frombuffer + cv2.imdecode).
    2. Run the extraction + audit pipeline (cached in session_state per filename).
    3. Push AuditResult into the accountant review queue.
    4. Show an instant audit preview to the submitting employee.
    """
    existing_filenames = [r["filename"] for r in st.session_state.pending_reviews]
    if filename in existing_filenames:
        st.warning("This receipt is already in the review queue.")
        return

    # ── Image handoff: UploadedFile → OpenCV BGR ──────────────────────────
    with st.spinner("Decoding image…"):
        try:
            cv2_img = uploaded_file_to_cv2(uploaded_file)
        except RuntimeError as exc:
            st.error(f"Image decode error: {exc}")
            return

    # ── Run pipeline (OCR + policy audit), cache to avoid re-runs on UI interactions
    cache_key = f"audit_{filename}"
    if cache_key not in st.session_state:
        with st.spinner("Running OCR extraction and policy audit…"):
            try:
                audit_result = run_pipeline(cv2_img)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return
        st.session_state[cache_key] = audit_result
    else:
        audit_result = st.session_state[cache_key]

    # ── Register in claim DB ──────────────────────────────────────────────
    from datetime import datetime
    claim_code   = generate_claim_code()
    submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine_status = audit_result.get("status", "FLAGGED_FOR_REVIEW")
    db_status = "Auto-Approved" if engine_status == "APPROVED" else "Pending Review"

    st.session_state.claim_status_db[claim_code] = {
        "filename":     filename,
        "status":       db_status,
        "submitted_at": submitted_at,
    }

    st.session_state.pending_reviews.append(
        {
            "image_bytes":  jpg_bytes,
            "filename":     filename,
            "audit_result": audit_result,
            "claim_code":   claim_code,
        }
    )

    st.success("Receipt forwarded to the Accountant review queue.")
    st.markdown(
        f"""
        <div style="border:2px solid #22C55E;border-radius:10px;
                    padding:1rem 1.4rem;margin-top:.6rem;
                    background:rgba(34,197,94,.08);">
            <p style="margin:0 0 .3rem;font-size:.85rem;color:grey;">Your Claim Code</p>
            <p style="margin:0;font-size:1.6rem;font-weight:700;
                       letter-spacing:.12em;font-family:monospace;">{claim_code}</p>
            <p style="margin:.4rem 0 0;font-size:.8rem;color:grey;">
                Save this code — use it to check your claim status at any time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Instant audit preview for the submitter ───────────────────────────
    st.markdown("---")
    st.subheader("🧾 Audit Preview")
    pil_img = Image.open(io.BytesIO(jpg_bytes))
    render_audit_result(audit_result, pil_img)


# ─────────────────────────────────────────────
# ACCOUNTANT VIEW
# ─────────────────────────────────────────────

def render_accountant_view() -> None:
    st.header("🔎 Accountant Review Dashboard")

    if not st.session_state.pending_reviews:
        st.info("No receipts pending review.")
        return

    for idx, record in enumerate(st.session_state.pending_reviews):
        filename:     str   = record["filename"]
        img_bytes:    bytes = record["image_bytes"]
        audit_result: dict  = record["audit_result"]
        claim_code:   str   = record.get("claim_code", "")
        engine_status        = audit_result.get("status", "FLAGGED_FOR_REVIEW")

        with st.expander(
            f"Receipt #{idx + 1} — {filename}  [{engine_status}]",
            expanded=True,
        ):
            pil_img = Image.open(io.BytesIO(img_bytes))

            # ── Full audit result render ───────────────────────────────────
            render_audit_result(audit_result, pil_img)

            st.markdown("---")

            # ── Human Override buttons ────────────────────────────────────
            needs_override = engine_status in ("FLAGGED_FOR_REVIEW", "REJECTED")

            if needs_override:
                st.markdown("**Manual Override**")
                btn_col_a, btn_col_b, *_ = st.columns([1, 1, 3])

                with btn_col_a:
                    if st.button(
                        "✅ Approve",
                        key=f"approve_{idx}_{filename}",
                        use_container_width=True,
                    ):
                        simulate_approve_notification(filename)
                        if claim_code and claim_code in st.session_state.claim_status_db:
                            st.session_state.claim_status_db[claim_code]["status"] = "Approved"
                        st.session_state.pending_reviews.pop(idx)
                        st.success(f"Approved — notification sent for **{filename}**.")
                        st.rerun()

                with btn_col_b:
                    if st.button(
                        "❌ Reject",
                        key=f"reject_{idx}_{filename}",
                        use_container_width=True,
                    ):
                        if filename not in st.session_state.rejected_receipts:
                            st.session_state.rejected_receipts.append(filename)
                        if claim_code and claim_code in st.session_state.claim_status_db:
                            st.session_state.claim_status_db[claim_code]["status"] = "Rejected"
                        st.session_state.pending_reviews.pop(idx)
                        st.error(f"Receipt **{filename}** has been rejected.")
                        st.rerun()
            else:
                st.success("✅ Engine auto-approved this receipt — no manual action required.")


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.title("🧾 Expense Automator")
        st.markdown("---")
        view = st.radio(
            "Select View",
            options=["Employee — Submit Receipt", "Accountant — Review Dashboard"],
            index=0,
            key="sidebar_view_selector",
        )
        st.markdown("---")
        st.caption("HackForge '26 · LGGP-HF")
    return view


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    selected_view = render_sidebar()

    if selected_view == "Employee — Submit Receipt":
        render_employee_view()
    else:
        render_accountant_view()


if __name__ == "__main__":
    main()
