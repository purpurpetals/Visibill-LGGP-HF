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

import os
import shutil
import cv2
import json
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION  —  Set the path to your Tesseract binary (Windows OR Linux - OS-Aware)
# ---------------------------------------------------------------------------
if os.name == "nt":
    # Windows: Look for local installation
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        pytesseract.pytesseract.tesseract_cmd = windows_path
else:
    # Linux / Streamlit Cloud: Resolve executable from system PATH
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
#CONFIDENCE_REVIEW_THRESHOLD   = 60    # words / overall scans below this → requires_review
CONFIDENCE_DISTORTION_FLOOR   = 50    # overall avg below this → trigger Pivot 1 bypass
TESSERACT_CONFIG              = r"--oem 3 --psm 6"  # LSTM engine, uniform block of text
USE_EASYOCR                   = False  # Set True to swap OCR engine to easyocr

# CV / Spatial Constants
TARGET_MIN_WIDTH              = 1500  # images narrower than this scaled to 1500px width
ADAPTIVE_THRESHOLD_C          = 11    # lowered to 11 to stop eroding thin text
SPATIAL_GAP_THRESHOLD         = 15    # horizontal gap > 15px indicates tabular column separator


# ===========================================================================
# SECTION 1 — OpenCV Preprocessing Pipeline
# ===========================================================================

def _load_image(source) -> np.ndarray:
    """
    Load a receipt image from:
      - file path (str / Path)
      - raw bytes / bytearray
      - file-like object with .read() (e.g. BytesIO, Streamlit UploadedFile)
      - PIL.Image object
    Returns an OpenCV BGR matrix.
    Raises RuntimeError on failure.
    """
    if isinstance(source, Image.Image):
        arr = np.array(source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if hasattr(source, "read"):
        source = source.read()

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


def _normalize_resolution(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    DPI / Resolution Scaling:
    If width is less than 1500 pixels, scale the BGR matrix up using
    cv2.resize (with cv2.INTER_CUBIC) so the width is 1500px (maintaining aspect ratio).
    """
    h, w = bgr.shape[:2]
    if w < TARGET_MIN_WIDTH:
        scale = TARGET_MIN_WIDTH / float(w)
        new_w = TARGET_MIN_WIDTH
        new_h = int(round(h * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return bgr, scale
    return bgr, 1.0


def _to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR matrix to single-channel grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Remove pixel grain while preserving sharp character edges.
    bilateralFilter is chosen because it respects edge gradients.
    """
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Micro-Deskewing (sub-90° rotation correction) via projection-profile variance.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle    = 0.0
    best_variance = -1.0

    for angle in np.arange(-15.0, 15.5, 0.5):
        h, w  = thresh.shape
        cx, cy = w / 2.0, h / 2.0
        M      = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            thresh, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        row_sums = rotated.sum(axis=1).astype(np.float64)
        variance = row_sums.var()
        if variance > best_variance:
            best_variance = variance
            best_angle    = angle

    if abs(best_angle) < 0.5:
        return gray, 0.0

    h, w   = gray.shape
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), best_angle, 1.0)
    deskewed = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed, best_angle


def _binarize(denoised: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding normalises uneven lighting and thermal fade.
    C parameter is set to 11 to stop eroding thin text.
    Immediately after thresholding, applies cv2.MORPH_CLOSE operation using
    a 2x2 numpy ones kernel (np.ones((2, 2), np.uint8)) to bridge broken thermal ink lines.
    """
    h, w = denoised.shape[:2]
    block = min(31, h, w)
    if block % 2 == 0:
        block -= 1
    block = max(3, block)

    thresh = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block,
        C=ADAPTIVE_THRESHOLD_C,
    )

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return binary


def _morphological_close(binary: np.ndarray) -> np.ndarray:
    """Standalone morphological close with 2x2 numpy ones kernel."""
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _correct_orientation(binary: np.ndarray, original_bgr: np.ndarray):
    """Detect and correct receipt orientation (0, 90, 180, 270 degrees)."""
    rotation = 0
    try:
        pil_img  = Image.fromarray(binary)
        osd_data = pytesseract.image_to_osd(pil_img, config="--psm 0", nice=0)
        for line in osd_data.splitlines():
            if "Rotate:" in line:
                rotation = int(line.split(":")[1].strip())
                break
    except Exception:
        h, w = binary.shape
        if h < w:
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
    Execute OpenCV preprocessing pipeline in canonical order:
      1. Load image
      2. DPI Scaling: if width < 1500, scale up to 1500px width (cv2.INTER_CUBIC)
      3. Grayscale conversion
      4. Micro-deskewing
      5. Denoising
      6. Adaptive thresholding (C=11) + 2x2 morphological closing
      7. Orientation correction
    """
    steps = []

    bgr = _load_image(source)
    steps.append("image_load")

    # 1. DPI Scaling: check width, scale to 1500px width if width < 1500
    h, w = bgr.shape[:2]
    if w < TARGET_MIN_WIDTH:
        scale = TARGET_MIN_WIDTH / float(w)
        new_w = TARGET_MIN_WIDTH
        new_h = int(round(h * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        steps.append(f"dpi_scale_upscale_to_{new_w}px_{scale:.2f}x")

    gray = _to_grayscale(bgr)
    steps.append("grayscale_conversion")

    gray, skew_angle = _deskew(gray)
    if abs(skew_angle) >= 0.5:
        steps.append(f"micro_deskew_{skew_angle:.1f}deg")
    else:
        steps.append("deskew_not_required")

    denoised = _denoise(gray)
    steps.append("bilateral_denoise")

    # 2. Binarization with C=11 and 2x2 morphological close
    binary = _binarize(denoised)
    steps.append("adaptive_threshold_c11_morph_close")

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
    switch between Tesseract (default) and easyocr.
    """
    if not USE_EASYOCR:
        pil_img = Image.fromarray(matrix)
        return pytesseract.image_to_data(
            pil_img,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

    try:
        import easyocr
    except ImportError as exc:
        raise ImportError(
            "easyocr is not installed. Run: pip install easyocr\n"
            "Note: easyocr requires PyTorch, which supports Python <= 3.12 only."
        ) from exc
    reader  = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(matrix)
    synthetic: dict = {
        "text":     [],
        "conf":     [],
        "line_num": [],
        "word_num": [],
        "left":     [],
        "width":    [],
    }
    for idx, (bbox, word, conf_float) in enumerate(results):
        synthetic["text"].append(word)
        synthetic["conf"].append(int(conf_float * 100))
        synthetic["line_num"].append(idx)
        synthetic["word_num"].append(1)
        try:
            xs = [pt[0] for pt in bbox]
            synthetic["left"].append(int(min(xs)))
            synthetic["width"].append(int(max(xs) - min(xs)))
        except Exception:
            synthetic["left"].append(-1)
            synthetic["width"].append(-1)
    return synthetic


def _parse_ocr_data(data: dict) -> dict:
    """
    Parse raw OCR DICT into canonical payload structures.
    Preserves spatial layout:
      Calculates horizontal gap between right edge (left + width) of word[i]
      and left edge of word[i+1]. If gap > 15 pixels, inserts a tab character '\t'
      so tabular layout between 'Item' and 'Price' is preserved for downstream regex.
    """
    words     = data.get("text",     [])
    confs     = data.get("conf",     [])
    line_nums = data.get("line_num", [])
    word_nums = data.get("word_num", [])
    lefts     = data.get("left",     [])
    widths    = data.get("width",    [])

    valid_confs    = []
    low_conf_words = []

    # Map line_num -> list of (left, width, word_num, word)
    lines_dict: dict = defaultdict(list)

    total_tokens = len(words)
    for i in range(total_tokens):
        word = str(words[i]).strip()
        try:
            conf = int(confs[i])
        except (ValueError, TypeError):
            conf = -1

        if conf == -1 or not word or not any(c.isalnum() for c in word):
            continue

        valid_confs.append(conf)
        if conf < CONFIDENCE_REVIEW_THRESHOLD:
            low_conf_words.append(word)

        line = line_nums[i] if i < len(line_nums) else 0
        wnum = word_nums[i] if i < len(word_nums) else 0
        lft  = int(lefts[i]) if i < len(lefts) else -1
        wid  = int(widths[i]) if i < len(widths) else -1

        lines_dict[line].append((lft, wid, wnum, word))

    ocr_confidence_avg = round(float(np.mean(valid_confs)), 2) if valid_confs else 0.0

    raw_itemized_list = []
    for ln in sorted(lines_dict.keys()):
        # Sort words on line by physical left coordinate, falling back to word_num
        line_tokens = sorted(
            lines_dict[ln],
            key=lambda t: (t[0] if t[0] >= 0 else t[2], t[2])
        )
        if not line_tokens:
            continue

        line_parts: list[str] = []
        for idx, (lft, wid, wnum, word) in enumerate(line_tokens):
            if idx == 0:
                line_parts.append(word)
                continue

            prev_lft, prev_wid, _, _ = line_tokens[idx - 1]
            # Calculate horizontal gap between right edge (left + width) of word[i-1] and left edge of word[i]
            if lft >= 0 and prev_lft >= 0 and prev_wid >= 0:
                gap = lft - (prev_lft + prev_wid)
                if gap > SPATIAL_GAP_THRESHOLD:
                    line_parts.append("\t")
                else:
                    line_parts.append(" ")
            else:
                line_parts.append(" ")

            line_parts.append(word)

        raw_itemized_list.append("".join(line_parts))

    raw_text = "\n".join(raw_itemized_list)

    requires_review = (ocr_confidence_avg < 50.0)

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
    status:                str,
    raw_text:              str,
    raw_itemized_list:     list,
    ocr_confidence_avg:    float,
    low_confidence_words:  list,
    requires_review:       bool,
    preprocessing_applied: list,
    error_message:         str = None,
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

    Handoff contract for the Policy Engine:
        result = extract_full(image_bytes)
        payload   = result["payload"]    # standard JSON contract
        ocr_dict  = result["ocr_dict"]   # raw pytesseract DICT for engine.audit()
    """
    # Stage 1: Preprocessing
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

    # Stage 2: OCR
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

    # Stage 3: Package
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
# ===========================================================================

def _generate_dummy_receipt(output_path: str = "dummy_receipt.png") -> str:
    """Generate a synthetic receipt for self-testing."""
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

    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    import sys

    TEST_IMAGE_PATH = "test_receipt.png"

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
