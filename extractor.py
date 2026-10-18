# =============================================================================
# extractor.py  —  EXTRACTION Zone
# Expense Automator | Receipt Image Preprocessing & OCR Pipeline
#
# Position in data-flow:
#   [Employee Frontend] ──► [extractor.py] ──► [Analysis Zone (Regex + Policy)]
#
# Dependencies:
#   pip install opencv-python pytesseract numpy pillow
#   Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
# =============================================================================

import cv2
import json
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION  —  Set the path to your Tesseract binary (Windows)
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
CONFIDENCE_REVIEW_THRESHOLD   = 60    # words / overall scans below this → requires_review
CONFIDENCE_DISTORTION_FLOOR   = 50    # overall avg below this → trigger Pivot 1 bypass  [BUG-7: raised from 30 → activates on genuinely poor scans]
LOW_CONF_WORD_RATIO_THRESHOLD = 0.15  # >15 % of words low-conf → requires_review        [BUG-3: replaces bare bool]
TESSERACT_CONFIG              = r"--oem 3 --psm 6"  # LSTM engine, uniform block of text
USE_EASYOCR                   = False  # Set True to swap OCR engine to easyocr           [BUG-1: feature flag replaces dead-code block]

# ===========================================================================
# SECTION 1 — OpenCV Preprocessing Pipeline
# ===========================================================================

def _load_image(source) -> np.ndarray:
    """
    Load a receipt image from:
      - file path (str / Path)
      - raw bytes / bytearray
      - file-like object with .read() (e.g. BytesIO, Streamlit UploadedFile)  [BUG-5]
      - PIL.Image object                                                        [BUG-5]
    Returns an OpenCV BGR matrix.
    Raises RuntimeError on failure.
    """
    # BUG-5 fix: handle PIL.Image objects (e.g. passed from st.file_uploader callbacks)
    if isinstance(source, Image.Image):
        arr = np.array(source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # BUG-5 fix: handle file-like objects (BytesIO, Streamlit UploadedFile, etc.)
    if hasattr(source, "read"):
        source = source.read()   # normalise to bytes and fall through

    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise RuntimeError(f"OpenCV could not open image at path: {source}")
        return img

    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("OpenCV could not decode the supplied byte stream.")
        return img

    raise TypeError(
        f"Unsupported source type '{type(source)}'. "
        "Expected path, bytes, file-like object, or PIL.Image."
    )


def _to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR matrix to single-channel grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Remove pixel grain while preserving sharp character edges.
    bilateralFilter is chosen because it respects edge gradients,
    keeping thin receipt typefaces intact.
    """
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def _binarize(denoised: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding normalises uneven lighting, thermal-paper
    fade, and shadow creases across the receipt surface.

    BUG-6 fix: blockSize is now computed dynamically. cv2.adaptiveThreshold
    requires blockSize to be ODD and strictly less than min(image dimensions).
    A hardcoded 31 caused a cv2.error on any image narrower than 31 px.
    """
    h, w = denoised.shape[:2]
    # Clamp to image dimensions, then enforce odd and >= 3
    block = min(31, h, w)
    if block % 2 == 0:
        block -= 1
    block = max(3, block)
    return cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block,
        C=15,
    )


def _correct_orientation(binary: np.ndarray, original_bgr: np.ndarray):
    """
    Detect and correct receipt orientation (0, 90, 180, 270 degrees).

    Strategy (two-tier):
      1. Try pytesseract.image_to_osd for an accurate rotation reading.
      2. Fall back to an aspect-ratio heuristic if OSD fails (e.g. on
         very small or featureless images).

    Returns:
        corrected_binary   (np.ndarray)
        corrected_bgr      (np.ndarray)   — kept for Pivot-1 bypass path
        rotation_applied   (int)          — degrees rotated
    """
    rotation = 0

    try:
        pil_img  = Image.fromarray(binary)
        osd_data = pytesseract.image_to_osd(pil_img, config="--psm 0", nice=0)
        for line in osd_data.splitlines():
            if "Rotate:" in line:
                rotation = int(line.split(":")[1].strip())
                break
    except Exception:
        # OSD unavailable — use aspect ratio as a coarse orientation cue.
        h, w = binary.shape
        if h < w:          # landscape receipt handed in on its side
            rotation = 90

    rotate_map = {
        0:   None,
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    flag = rotate_map.get(rotation)
    if flag is not None:
        binary       = cv2.rotate(binary, flag)
        original_bgr = cv2.rotate(original_bgr, flag)

    return binary, original_bgr, rotation


def preprocess(source) -> dict:
    """
    Execute the full OpenCV preprocessing pipeline in canonical order:
      load -> grayscale -> denoise -> binarize -> orient

    Returns a dict with:
        'processed_matrix'   : final binary np.ndarray (ready for OCR)
        'original_bgr'       : orientation-corrected raw colour matrix
        'steps_applied'      : ordered list of step labels (for metadata)
    """
    steps = []

    bgr = _load_image(source)
    steps.append("image_load")

    gray = _to_grayscale(bgr)
    steps.append("grayscale_conversion")

    denoised = _denoise(gray)
    steps.append("bilateral_denoise")

    binary = _binarize(denoised)
    steps.append("adaptive_threshold_binarization")

    binary, bgr, rotation = _correct_orientation(binary, bgr)
    if rotation != 0:
        steps.append(f"orientation_correction_{rotation}deg")
    else:
        steps.append("orientation_verified_upright")

    return {
        "processed_matrix": binary,
        "original_bgr":     bgr,
        "steps_applied":    steps,
    }


# ===========================================================================
# SECTION 2 — OCR Extraction & Confidence Scoring
# ===========================================================================

def _run_tesseract(matrix: np.ndarray) -> dict:
    """
    OCR engine dispatcher. Toggle USE_EASYOCR at the top of this file to
    switch between Tesseract (default) and easyocr without editing this function.

    BUG-1 fix: the previous easyocr block was permanently dead code — it sat
    after an unconditional `return` and could never execute regardless of what
    was uncommented. Engine selection is now controlled via the USE_EASYOCR flag.

    Returns a pytesseract-style DICT: {text, conf, line_num, word_num, ...}
    """
    if not USE_EASYOCR:
        # ── Tesseract path (default) ──────────────────────────────────────
        pil_img = Image.fromarray(matrix)
        return pytesseract.image_to_data(
            pil_img,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

    # ── PIVOT 2 — easyocr fallback ────────────────────────────────────────
    # Set USE_EASYOCR = True at the top of this file to activate.
    # First install: pip install easyocr  (downloads ~100 MB model on first run)
    import easyocr
    reader  = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(matrix)
    # Normalise to pytesseract DICT shape so the rest of the pipeline is identical.
    synthetic: dict = {"text": [], "conf": [], "line_num": [], "word_num": []}
    for idx, (_, word, conf_float) in enumerate(results):
        synthetic["text"].append(word)
        synthetic["conf"].append(int(conf_float * 100))
        synthetic["line_num"].append(idx)
        synthetic["word_num"].append(1)
    return synthetic


def _parse_ocr_data(data: dict) -> dict:
    """
    Parse the raw pytesseract DICT into:
      - raw_text            : full joined string
      - raw_itemized_list   : one string per detected line (preserves layout)
      - ocr_confidence_avg  : float
      - low_confidence_words: list of words with conf < CONFIDENCE_REVIEW_THRESHOLD
      - requires_review     : bool

    BUG-8 fix: words within each line are now sorted by word_num to guarantee
               left-to-right reading order even on skewed or multi-column receipts.
    BUG-3 fix: requires_review now uses LOW_CONF_WORD_RATIO_THRESHOLD (ratio of
               low-conf words) instead of a bare bool — prevents every scan from
               being flagged just because one price digit was borderline.
    """
    words      = data.get("text",     [])
    confs      = data.get("conf",     [])
    line_nums  = data.get("line_num", [])
    word_nums  = data.get("word_num", [])   # BUG-8: needed for stable word ordering

    valid_confs    = []
    low_conf_words = []
    # BUG-8 fix: lines_dict maps line_num → {word_num: word} for ordered reconstruction
    lines_dict: dict = defaultdict(dict)

    for word, conf, line, wnum in zip(words, confs, line_nums, word_nums):
        conf = int(conf)
        word = word.strip()

        if conf == -1 or not word:   # tesseract non-text rows
            continue

        valid_confs.append(conf)

        if conf < CONFIDENCE_REVIEW_THRESHOLD:
            low_conf_words.append(word)

        lines_dict[line][wnum] = word   # BUG-8: key by word_num, not append

    ocr_confidence_avg = round(float(np.mean(valid_confs)), 2) if valid_confs else 0.0

    # BUG-8 fix: sort words within each line by word_num before joining
    raw_itemized_list = [
        " ".join(v for _, v in sorted(lines_dict[ln].items()))
        for ln in sorted(lines_dict.keys())
        if lines_dict[ln]
    ]
    raw_text = "\n".join(raw_itemized_list)

    # BUG-3 fix: ratio-based check — flag only if a significant fraction is low-conf
    low_conf_ratio  = len(low_conf_words) / max(len(valid_confs), 1)
    requires_review = (
        ocr_confidence_avg < CONFIDENCE_REVIEW_THRESHOLD
        or low_conf_ratio > LOW_CONF_WORD_RATIO_THRESHOLD
    )

    return {
        "raw_text":             raw_text,
        "raw_itemized_list":    raw_itemized_list,
        "ocr_confidence_avg":   ocr_confidence_avg,
        "low_confidence_words": low_conf_words,
        "requires_review":      requires_review,
    }


# ===========================================================================
# SECTION 3 — JSON Packaging & Handoff Contract
# ===========================================================================

def _build_payload(
    status:               str,
    raw_text:             str,
    raw_itemized_list:    list,
    ocr_confidence_avg:   float,
    low_confidence_words: list,
    requires_review:      bool,
    preprocessing_applied: list,
    error_message:        str = None,
) -> dict:
    """
    Assemble the canonical JSON contract expected by the Analysis Zone
    'Receive JSON file' node.
    """
    payload = {
        "status":            status,
        "raw_text":          raw_text,
        "raw_itemized_list": raw_itemized_list,
        "metadata": {
            "ocr_confidence_avg":    ocr_confidence_avg,
            "low_confidence_words":  low_confidence_words,
            "requires_review":       requires_review,
            "preprocessing_applied": preprocessing_applied,
        },
    }
    if error_message:
        payload["error_message"] = error_message
    return payload


# ===========================================================================
# SECTION 4 — Master Extraction Entry Point (with Fail-Safe Pivots)
# ===========================================================================

def extract(source) -> dict:
    """
    Primary public interface.

    Accepts:
        source  —  file path (str / Path) OR raw image bytes

    Returns:
        dict matching the Analysis Zone JSON contract
        (serialisable via json.dumps)
    """
    # Stage 1: Preprocessing
    try:
        prep = preprocess(source)
    except Exception as exc:
        return _build_payload(
            status="error",
            raw_text="",
            raw_itemized_list=[],
            ocr_confidence_avg=0.0,
            low_confidence_words=[],
            requires_review=True,
            preprocessing_applied=[],
            error_message=f"Preprocessing failed: {exc}",
        )

    processed_matrix = prep["processed_matrix"]
    original_bgr     = prep["original_bgr"]
    steps_applied    = prep["steps_applied"]

    # Stage 2: OCR on preprocessed image
    try:
        ocr_data = _run_tesseract(processed_matrix)
        parsed   = _parse_ocr_data(ocr_data)
    except Exception as exc:
        return _build_payload(
            status="error",
            raw_text="",
            raw_itemized_list=[],
            ocr_confidence_avg=0.0,
            low_confidence_words=[],
            requires_review=True,
            preprocessing_applied=steps_applied,
            error_message=f"OCR engine error: {exc}",
        )

    # PIVOT 1: OpenCV Distortion Bypass
    # If thresholding eroded faded ink -> empty result or very low confidence,
    # discard the processed matrix and re-run OCR on the orientation-corrected
    # raw (colour) image to salvage what the camera actually captured.
    needs_bypass = (
        not parsed["raw_text"].strip()
        or parsed["ocr_confidence_avg"] < CONFIDENCE_DISTORTION_FLOOR
    )

    if needs_bypass:
        steps_applied.append("pivot1_opencv_bypass_activated")
        try:
            raw_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            ocr_data = _run_tesseract(raw_gray)
            parsed   = _parse_ocr_data(ocr_data)
        except Exception as exc:
            return _build_payload(
                status="error",
                raw_text="",
                raw_itemized_list=[],
                ocr_confidence_avg=0.0,
                low_confidence_words=[],
                requires_review=True,
                preprocessing_applied=steps_applied,
                error_message=f"Pivot-1 bypass OCR failed: {exc}",
            )

    # Stage 3: Package & Return
    return _build_payload(
        status="success",
        raw_text=parsed["raw_text"],
        raw_itemized_list=parsed["raw_itemized_list"],
        ocr_confidence_avg=parsed["ocr_confidence_avg"],
        low_confidence_words=parsed["low_confidence_words"],
        requires_review=parsed["requires_review"],
        preprocessing_applied=steps_applied,
    )


def extract_full(source) -> dict:
    """
    Extended public interface that returns both the standard Analysis Zone
    payload AND the raw pytesseract OCR dict (word-level tokens).

    This is the handoff contract for the Policy Engine:

        result = extract_full(image_bytes)
        payload   = result["payload"]    # standard JSON contract
        ocr_dict  = result["ocr_dict"]   # raw pytesseract DICT for engine.audit()

    Accepts the same ``source`` types as ``extract()``:
        file path (str / Path)  OR  raw image bytes / bytearray.
    """
    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    try:
        prep = preprocess(source)
    except Exception as exc:
        empty_payload = _build_payload(
            status="error",
            raw_text="",
            raw_itemized_list=[],
            ocr_confidence_avg=0.0,
            low_confidence_words=[],
            requires_review=True,
            preprocessing_applied=[],
            error_message=f"Preprocessing failed: {exc}",
        )
        return {"payload": empty_payload, "ocr_dict": {}}

    processed_matrix = prep["processed_matrix"]
    original_bgr     = prep["original_bgr"]
    steps_applied    = prep["steps_applied"]

    # ── Stage 2: OCR ─────────────────────────────────────────────────────
    try:
        ocr_dict = _run_tesseract(processed_matrix)
        parsed   = _parse_ocr_data(ocr_dict)
    except Exception as exc:
        err_payload = _build_payload(
            status="error",
            raw_text="",
            raw_itemized_list=[],
            ocr_confidence_avg=0.0,
            low_confidence_words=[],
            requires_review=True,
            preprocessing_applied=steps_applied,
            error_message=f"OCR engine error: {exc}",
        )
        return {"payload": err_payload, "ocr_dict": {}}

    # PIVOT 1: OpenCV Distortion Bypass
    needs_bypass = (
        not parsed["raw_text"].strip()
        or parsed["ocr_confidence_avg"] < CONFIDENCE_DISTORTION_FLOOR
    )
    if needs_bypass:
        steps_applied.append("pivot1_opencv_bypass_activated")
        try:
            raw_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            ocr_dict = _run_tesseract(raw_gray)
            parsed   = _parse_ocr_data(ocr_dict)
        except Exception as exc:
            err_payload = _build_payload(
                status="error",
                raw_text="",
                raw_itemized_list=[],
                ocr_confidence_avg=0.0,
                low_confidence_words=[],
                requires_review=True,
                preprocessing_applied=steps_applied,
                error_message=f"Pivot-1 bypass OCR failed: {exc}",
            )
            return {"payload": err_payload, "ocr_dict": {}}

    # ── Stage 3: Package ──────────────────────────────────────────────────
    payload = _build_payload(
        status="success",
        raw_text=parsed["raw_text"],
        raw_itemized_list=parsed["raw_itemized_list"],
        ocr_confidence_avg=parsed["ocr_confidence_avg"],
        low_confidence_words=parsed["low_confidence_words"],
        requires_review=parsed["requires_review"],
        preprocessing_applied=steps_applied,
    )
    return {"payload": payload, "ocr_dict": ocr_dict}


# ===========================================================================
# SELF-TEST BLOCK
# Generates a synthetic receipt image with PIL when no real image is found,
# runs the full pipeline, and prints the JSON output.
# Usage:  python extractor.py [optional_image_path]
# ===========================================================================

def _generate_dummy_receipt(output_path: str = "dummy_receipt.png") -> str:
    """
    Draw a plausible receipt layout onto a white canvas using PIL
    so the module can be verified without a real scan.
    """
    W, H   = 400, 700
    canvas = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_large  = ImageFont.truetype("arial.ttf", 20)
        font_medium = ImageFont.truetype("arial.ttf", 15)
        font_small  = ImageFont.truetype("arial.ttf", 13)
    except (IOError, OSError):
        font_large = font_medium = font_small = ImageFont.load_default()

    lines = [
        ("QUICKMART GROCERY",           30,  font_large,  (0,   0,   0)),
        ("123 Main St, Springfield",    60,  font_small,  (60,  60,  60)),
        ("Tel: (555) 867-5309",         80,  font_small,  (60,  60,  60)),
        ("--------------------------------", 105, font_small, (0, 0, 0)),
        ("Date: 2026-10-17   14:32",    125, font_small,  (0,   0,   0)),
        ("Receipt #: 00471892",         145, font_small,  (0,   0,   0)),
        ("--------------------------------", 165, font_small, (0, 0, 0)),
        ("Whole Milk 1gal         $3.49", 185, font_medium, (0, 0, 0)),
        ("Sourdough Bread         $4.19", 205, font_medium, (0, 0, 0)),
        ("Organic Eggs 12ct       $5.99", 225, font_medium, (0, 0, 0)),
        ("Cheddar Cheese 8oz      $3.79", 245, font_medium, (0, 0, 0)),
        ("Sparkling Water 6pk     $6.49", 265, font_medium, (0, 0, 0)),
        ("--------------------------------", 285, font_small, (0, 0, 0)),
        ("Subtotal:              $23.95", 305, font_medium, (0, 0, 0)),
        ("Tax (8.5%):             $2.04", 325, font_medium, (0, 0, 0)),
        ("--------------------------------", 345, font_small, (0, 0, 0)),
        ("TOTAL:                 $25.99", 365, font_large,  (0, 0, 0)),
        ("--------------------------------", 395, font_small, (0, 0, 0)),
        ("Payment: VISA ****4821",        415, font_small,  (0, 0, 0)),
        ("Approved",                      435, font_small,  (0, 128, 0)),
        ("     Thank you for shopping!",  475, font_small,  (80, 80, 80)),
        ("     Please keep your receipt.", 495, font_small, (80, 80, 80)),
    ]

    for text, y, font, colour in lines:
        draw.text((20, y), text, fill=colour, font=font)

    # Minimal barcode strip at the bottom
    rng = np.random.default_rng(seed=42)
    for x in range(20, 380, 4):
        bar_height = int(rng.integers(30, 60))
        bar_width  = int(rng.integers(1, 3))
        draw.rectangle([x, 540, x + bar_width, 540 + bar_height], fill=(0, 0, 0))
    draw.text((150, 610), "00471892", fill=(0, 0, 0), font=font_small)

    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    import sys

    TEST_IMAGE_PATH = "test_receipt.jpg"

    if len(sys.argv) > 1:
        image_source = sys.argv[1]
        print(f"[extractor] Using supplied image: {image_source}")
    elif Path(TEST_IMAGE_PATH).exists():
        image_source = TEST_IMAGE_PATH
        print(f"[extractor] Found local test image: {TEST_IMAGE_PATH}")
    else:
        print("[extractor] No test image found — generating synthetic receipt...")
        image_source = _generate_dummy_receipt("dummy_receipt.png")
        print(f"[extractor] Dummy receipt saved to: {image_source}")

    print("[extractor] Running extraction pipeline...\n")
    result = extract(image_source)

    print("=" * 60)
    print("EXTRACTION RESULT  (Analysis Zone JSON Contract)")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)

    if result["status"] == "success":
        meta = result["metadata"]
        print(f"\nStatus          : {result['status'].upper()}")
        print(f"OCR Confidence  : {meta['ocr_confidence_avg']}%")
        print(f"Requires Review : {meta['requires_review']}")
        print(f"Lines Extracted : {len(result['raw_itemized_list'])}")
        print(f"Steps Applied   : {', '.join(meta['preprocessing_applied'])}")
        if meta["low_confidence_words"]:
            print(f"Low-Conf Words  : {meta['low_confidence_words']}")
    else:
        print(f"\n[!] Pipeline error: {result.get('error_message', 'unknown')}")
