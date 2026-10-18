from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from dateutil import parser as date_parser
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Data Schemas
# ---------------------------------------------------------------------------

class ConfidenceFlag(BaseModel):
    value: Any
    conf: float
    requires_review: bool


class LineItem(BaseModel):
    name: str
    price: float
    confidence: float = 100.0


class ExtractedReceiptData(BaseModel):
    vendor: str
    date: str
    total_amount: float
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    category: str
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    currency: str = "$"
    original_total: float = 0.0


class AuditResult(BaseModel):
    status: Literal["APPROVED", "FLAGGED_FOR_REVIEW", "REJECTED"]
    verdict_reasons: List[str] = Field(default_factory=list)
    confidence_flags: Dict[str, ConfidenceFlag]
    extracted_data: ExtractedReceiptData
    math_verified: bool
    audit_timestamp: str

    def to_dict(self) -> dict:
        """Serialize AuditResult to standard dictionary."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Currency Conversion Helpers
# ---------------------------------------------------------------------------

CURRENCY_REGEX = re.compile(
    r"([\$€£₹]?)\s*([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]{1,2})|[0-9]+(?:\.[0-9]{1,2}))"
)

MOCK_EXCHANGE_RATES: Dict[str, float] = {
    "EUR": 1.10,
    "GBP": 1.30,
    "INR": 0.012,
    "USD": 1.0,
    "€": 1.10,
    "£": 1.30,
    "₹": 0.012,
    "$": 1.0,
}


def _convert_to_usd(amount: float, currency: str) -> float:
    """Helper to convert amount to USD using mock exchange rates."""
    if amount is None:
        return 0.0
    curr = str(currency).strip() if currency else "USD"
    rate = MOCK_EXCHANGE_RATES.get(curr.upper(), MOCK_EXCHANGE_RATES.get(curr, 1.0))
    return round(float(amount) * rate, 2)


def _parse_currency_amount(text: str) -> Tuple[Optional[float], str]:
    """Parse text for numeric amount and detect currency symbol."""
    matches = list(CURRENCY_REGEX.finditer(text))
    if not matches:
        return None, "USD"
    last_match = matches[-1]
    symbol = last_match.group(1).strip()
    raw_val = last_match.group(2).replace(",", "")

    currency_map = {
        "€": "EUR",
        "£": "GBP",
        "₹": "INR",
        "$": "USD",
    }
    currency = currency_map.get(symbol, "USD")
    try:
        return float(raw_val), currency
    except ValueError:
        return None, currency


# ---------------------------------------------------------------------------
# Internal Token & Layout Representation
# ---------------------------------------------------------------------------

class OCRToken:
    __slots__ = (
        "text",
        "clean_text",
        "conf",
        "left",
        "top",
        "width",
        "height",
        "right",
        "bottom",
        "line_num",
        "word_num",
        "block_num",
    )

    def __init__(
        self,
        text: Any,
        conf: Union[int, float, str, None] = 0.0,
        left: Any = 0,
        top: Any = 0,
        width: Any = 0,
        height: Any = 0,
        line_num: Any = 0,
        word_num: Any = 0,
        block_num: Any = 0,
    ):
        self.text = str(text) if text is not None else ""
        self.clean_text = self.text.strip()
        try:
            parsed_conf = float(conf) if conf is not None else 0.0
            self.conf = max(0.0, parsed_conf) if parsed_conf >= 0 else 0.0
        except (ValueError, TypeError):
            self.conf = 0.0

        def _safe_int(val: Any) -> int:
            if val is None or val == "":
                return 0
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return 0

        self.left = _safe_int(left)
        self.top = _safe_int(top)
        self.width = _safe_int(width)
        self.height = _safe_int(height)
        self.right = self.left + self.width
        self.bottom = self.top + self.height
        self.line_num = _safe_int(line_num)
        self.word_num = _safe_int(word_num)
        self.block_num = _safe_int(block_num)


class OCRLine:
    def __init__(self, tokens: List[OCRToken]):
        self.tokens = sorted(
            tokens,
            key=lambda t: (t.left if t.left > 0 else t.word_num, t.word_num),
        )
        self.text = " ".join(t.clean_text for t in self.tokens if t.clean_text)
        if self.tokens:
            self.left = min(t.left for t in self.tokens)
            self.top = min(t.top for t in self.tokens)
            self.right = max(t.right for t in self.tokens)
            self.bottom = max(t.bottom for t in self.tokens)
            self.width = self.right - self.left
            self.height = self.bottom - self.top
            self.conf = sum(t.conf for t in self.tokens) / len(self.tokens)
        else:
            self.left = self.top = self.right = self.bottom = self.width = self.height = 0
            self.conf = 0.0


# ---------------------------------------------------------------------------
# Receipt Audit Engine
# ---------------------------------------------------------------------------

class ReceiptAuditEngine:
    PROHIBITED_KEYWORDS = [
        "alcohol",
        "beer",
        "wine",
        "cocktail",
        "liquor",
        "bar",
        "tobacco",
        "gift card",
        "casino",
    ]

    CATEGORY_KEYWORDS = {
        "Meal": [
            "restaurant", "bistro", "cafe", "café", "grill", "burger", "food",
            "diner", "bakery", "kitchen", "steak", "pizza", "sushi", "taco",
            "lunch", "dinner", "breakfast", "coffee", "beverage", "menu", "eats"
        ],
        "Travel/Transit": [
            "uber", "lyft", "taxi", "cab", "airline", "flight", "transit",
            "subway", "metro", "railway", "train", "parking", "fuel", "gas",
            "petrol", "shell", "chevron", "bp", "exxon", "mobil", "toll"
        ],
        "Office Supplies": [
            "staples", "office depot", "stationery", "paper", "printer", "ink",
            "toner", "desk", "supplies", "best buy", "hardware", "software",
            "electronics", "officemax"
        ],
        "Lodging": [
            "hotel", "motel", "inn", "suites", "resort", "airbnb", "marriott",
            "hilton", "hyatt", "sheraton", "westin", "lodging", "room"
        ],
    }

    CURRENCY_REGEX = re.compile(
        r"([\$€£₹]?)\s*([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]{1,2})|[0-9]+(?:\.[0-9]{1,2}))"
    )
    TOTAL_KEYWORDS = ["total", "balance due", "amount paid", "grand total", "net total", "payment due", "total due", "final total"]
    SUBTOTAL_KEYWORDS = ["subtotal", "sub-total", "sub total", "net amount"]
    TAX_KEYWORDS = ["tax", "vat", "gst", "hst", "sales tax"]
    TIP_KEYWORDS = ["tip", "gratuity", "tips", "additional tip"]
    EXCLUDE_HEADER_WORDS = {"receipt", "invoice", "tax invoice", "welcome", "customer copy", "merchant copy", "order", "sale", "bill"}

    def __init__(self, confidence_threshold: float = 60.0):
        self.confidence_threshold = confidence_threshold

    _convert_to_usd = staticmethod(_convert_to_usd)
    _parse_currency_amount = staticmethod(_parse_currency_amount)

    # -----------------------------------------------------------------------
    # Stage 1: Parsing & Extraction
    # -----------------------------------------------------------------------

    def parse_receipt_data(
        self, ocr_data: Dict[str, List[Any]]
    ) -> Tuple[ExtractedReceiptData, Dict[str, ConfidenceFlag]]:
        tokens = self._extract_tokens(ocr_data)
        lines = self._group_tokens_into_lines(tokens)
        full_text = " ".join(t.clean_text for t in tokens if t.clean_text)

        vendor, vendor_conf = self._extract_vendor(lines, tokens)
        date_str, date_conf = self._extract_date(lines, tokens)
        raw_total, currency, total_conf = self._extract_total(lines, tokens)

        # Fallback: check if receipt text explicitly specifies non-USD symbol
        if currency == "USD":
            for line in lines:
                m = self.CURRENCY_REGEX.search(line.text)
                if m and m.group(1).strip() in ("€", "£", "₹"):
                    sym = m.group(1).strip()
                    currency = {"€": "EUR", "£": "GBP", "₹": "INR"}.get(sym, "USD")
                    break

        usd_total = self._convert_to_usd(raw_total, currency)
        subtotal, _ = self._extract_labeled_amount(lines, self.SUBTOTAL_KEYWORDS)
        tax, _ = self._extract_labeled_amount(lines, self.TAX_KEYWORDS)
        tip, _ = self._extract_labeled_amount(lines, self.TIP_KEYWORDS)
        line_items = self._extract_line_items(lines, currency)
        category = self._classify_category(vendor, line_items, full_text)

        confidence_flags = {
            "total_amount": ConfidenceFlag(
                value=usd_total,
                conf=round(total_conf, 2),
                requires_review=total_conf < self.confidence_threshold or usd_total <= 0.0,
            ),
            "date": ConfidenceFlag(
                value=date_str,
                conf=round(date_conf, 2),
                requires_review=date_conf < self.confidence_threshold or date_str == "",
            ),
            "vendor": ConfidenceFlag(
                value=vendor,
                conf=round(vendor_conf, 2),
                requires_review=vendor_conf < self.confidence_threshold or vendor == "Unknown Vendor",
            ),
        }

        extracted_data = ExtractedReceiptData(
            vendor=vendor,
            date=date_str,
            total_amount=round(usd_total, 2),
            subtotal=round(subtotal, 2) if subtotal is not None else None,
            tax=round(tax, 2) if tax is not None else None,
            tip=round(tip, 2) if tip is not None else None,
            category=category,
            line_items=line_items,
            currency=currency,
            original_total=round(raw_total, 2),
        )

        return extracted_data, confidence_flags

    # -----------------------------------------------------------------------
    # Stage 2: Policy & Compliance Rules Engine
    # -----------------------------------------------------------------------

    def evaluate_policy(
        self,
        extracted_data: ExtractedReceiptData,
        confidence_flags: Dict[str, ConfidenceFlag],
        raw_text: str = "",
        historical_receipts: Optional[pd.DataFrame] = None,
        employee_id: Optional[str] = None,
        attendee_count: int = 1,
        is_travel: bool = False,
        destination_city: str = "",
        attendees: Optional[List[str]] = None,
    ) -> AuditResult:
        attendees = attendees or []
        verdict_reasons: List[str] = []
        is_rejected = False
        is_flagged = False

        # Rule 0: Missing Data
        if extracted_data.total_amount == 0.0 or not extracted_data.date or not extracted_data.date.strip():
            is_flagged = True
            verdict_reasons.append("Mandatory data missing (Date or Total)")

        # Rule 1: Prohibited Items & Keyword Scanning
        banned_found = self._check_prohibited_keywords(extracted_data, raw_text)
        if banned_found:
            is_rejected = True
            for item in banned_found:
                verdict_reasons.append(f"Prohibited item/activity detected: '{item}'")

        # Rule 2: Layered Meal Policy
        if is_travel:
            tier_cities = ["New York", "San Francisco", "London", "Tokyo"]
            tier_cities_lower = {c.strip().lower() for c in tier_cities}
            is_tier = (
                destination_city in tier_cities
                or destination_city.strip().lower() in tier_cities_lower
            )
            daily_limit = 100.00 if is_tier else 75.00

            daily_spent = self._calculate_daily_meal_spend(
                extracted_data=extracted_data,
                historical_receipts=historical_receipts,
                employee_id=employee_id,
            )
            if daily_spent > daily_limit:
                is_flagged = True
                verdict_reasons.append(
                    f"Daily meal allowance exceeded: Total daily meal expenses on {extracted_data.date} "
                    f"is ${daily_spent:.2f}, exceeding the ${daily_limit:.2f} limit"
                )
        else:
            if extracted_data.total_amount > 25.00 and len(attendees) <= 1:
                is_flagged = True
                verdict_reasons.append(
                    "Office meal limit exceeded (requires attendee list for totals over $25)"
                )
            elif len(attendees) > 1 and extracted_data.total_amount > (25.00 * len(attendees)):
                is_flagged = True
                verdict_reasons.append(
                    f"Office meal limit exceeded: Total ${extracted_data.total_amount:.2f} "
                    f"exceeds cap of ${25.00 * len(attendees):.2f} for {len(attendees)} attendees ($25.00 per attendee)"
                )

        # Rule 3: Mathematical Integrity Verification
        math_verified = self._verify_math(extracted_data)
        if not math_verified:
            is_flagged = True
            verdict_reasons.append(
                f"Mathematical discrepancy: Subtotal ({extracted_data.subtotal}), "
                f"Tax ({extracted_data.tax}), and Tip ({extracted_data.tip}) do not sum to Total ({extracted_data.total_amount})"
            )

        # Rule 4: Multi-Dimensional Duplicate Check
        if self._check_duplicate(extracted_data, historical_receipts):
            is_flagged = True
            amount_disp = (
                extracted_data.original_total
                if extracted_data.original_total > 0.0
                else extracted_data.total_amount
            )
            verdict_reasons.append(
                f"DUPLICATE_SUSPECTED: Historical receipt found matching vendor '{extracted_data.vendor}', "
                f"date '{extracted_data.date}', and amount ${amount_disp:.2f}"
            )

        # Confidence checks on critical fields
        for field_name, flag in confidence_flags.items():
            if flag.requires_review:
                is_flagged = True
                verdict_reasons.append(
                    f"Low confidence or missing extraction for '{field_name}' "
                    f"(conf: {flag.conf:.1f}%, value: '{flag.value}')"
                )

        # Overall Status resolution
        if is_rejected:
            status = "REJECTED"
        elif is_flagged:
            status = "FLAGGED_FOR_REVIEW"
        else:
            status = "APPROVED"

        return AuditResult(
            status=status,
            verdict_reasons=verdict_reasons,
            confidence_flags=confidence_flags,
            extracted_data=extracted_data,
            math_verified=math_verified,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # -----------------------------------------------------------------------
    # Orchestrator Interface
    # -----------------------------------------------------------------------

    def audit(
        self,
        ocr_data: Dict[str, List[Any]],
        historical_receipts: Optional[pd.DataFrame] = None,
        employee_id: Optional[str] = None,
        attendee_count: int = 1,
        is_travel: bool = False,
        destination_city: str = "",
        attendees: Optional[List[str]] = None,
    ) -> AuditResult:
        attendees = attendees or []
        extracted_data, confidence_flags = self.parse_receipt_data(ocr_data)
        raw_text = " ".join(str(t) for t in ocr_data.get("text", []) if str(t).strip())
        return self.evaluate_policy(
            extracted_data=extracted_data,
            confidence_flags=confidence_flags,
            raw_text=raw_text,
            historical_receipts=historical_receipts,
            employee_id=employee_id,
            attendee_count=attendee_count,
            is_travel=is_travel,
            destination_city=destination_city,
            attendees=attendees,
        )

    # -----------------------------------------------------------------------
    # Token & Line Layout Helpers
    # -----------------------------------------------------------------------

    def _extract_tokens(self, ocr_data: Dict[str, List[Any]]) -> List[OCRToken]:
        texts = ocr_data.get("text", [])
        n = len(texts)
        confs = ocr_data.get("conf", [0] * n)
        lefts = ocr_data.get("left", [0] * n)
        tops = ocr_data.get("top", [0] * n)
        widths = ocr_data.get("width", [0] * n)
        heights = ocr_data.get("height", [0] * n)
        line_nums = ocr_data.get("line_num", [0] * n)
        word_nums = ocr_data.get("word_num", [0] * n)
        block_nums = ocr_data.get("block_num", [0] * n)

        tokens: List[OCRToken] = []
        for i in range(n):
            raw_txt = str(texts[i]).strip()
            if not raw_txt:
                continue
            token = OCRToken(
                text=raw_txt,
                conf=confs[i] if i < len(confs) else 0,
                left=lefts[i] if i < len(lefts) else 0,
                top=tops[i] if i < len(tops) else 0,
                width=widths[i] if i < len(widths) else 0,
                height=heights[i] if i < len(heights) else 0,
                line_num=line_nums[i] if i < len(line_nums) else 0,
                word_num=word_nums[i] if i < len(word_nums) else 0,
                block_num=block_nums[i] if i < len(block_nums) else 0,
            )
            tokens.append(token)
        return tokens

    def _group_tokens_into_lines(self, tokens: List[OCRToken]) -> List[OCRLine]:
        if not tokens:
            return []

        # Coordinate loss safety check: if max top is <= 1, spatial coords are missing
        if max((t.top for t in tokens), default=0) <= 1:
            grouped: Dict[Tuple[int, int], List[OCRToken]] = defaultdict(list)
            for token in sorted(tokens, key=lambda t: (t.block_num, t.line_num, t.word_num)):
                grouped[(token.block_num, token.line_num)].append(token)
            ocr_lines = [OCRLine(grouped[k]) for k in sorted(grouped.keys())]
            return ocr_lines

        # Sort tokens primarily top to bottom, secondarily left to right
        sorted_tokens = sorted(tokens, key=lambda t: (t.top, t.left))
        lines: List[List[OCRToken]] = []

        for token in sorted_tokens:
            placed = False
            for line_group in lines:
                anchor = line_group[0]
                # Compare vertical overlap within dynamic height threshold
                vertical_tolerance = max(8, int(anchor.height * 0.6))
                if abs(token.top - anchor.top) <= vertical_tolerance:
                    line_group.append(token)
                    placed = True
                    break
            if not placed:
                lines.append([token])

        ocr_lines = [OCRLine(grp) for grp in lines]
        ocr_lines.sort(key=lambda l: l.top)
        return ocr_lines

    # -----------------------------------------------------------------------
    # Entity Extraction Implementations
    # -----------------------------------------------------------------------

    def _extract_vendor(self, lines: List[OCRLine], tokens: List[OCRToken]) -> Tuple[str, float]:
        if not lines:
            return "Unknown Vendor", 0.0

        max_bottom = max((t.bottom for t in tokens), default=1)
        header_cutoff = max_bottom * 0.30

        candidate_scores: List[Tuple[float, OCRLine]] = []
        candidate_lines = lines[:3] if max_bottom <= 1 else lines

        for idx, line in enumerate(candidate_lines):
            if max_bottom > 1 and line.top > header_cutoff:
                continue
            cleaned = line.text.strip().lower()
            if not cleaned or len(cleaned) < 2:
                continue
            if any(cleaned == excl for excl in self.EXCLUDE_HEADER_WORDS):
                continue
            if re.match(r"^[0-9\-\:\/\.\s\(\)\+]+$", cleaned):
                continue
            if any(k in cleaned for k in ["tel:", "phone:", "fax:", "date:", "time:", "tax #"]):
                continue

            # Prioritize larger font height and top proximity
            height_score = line.height * 2.0
            position_score = max(0, 100 - (line.top / (max_bottom or 1)) * 100) if max_bottom > 1 else max(0, 100 - idx * 25)
            score = height_score + position_score
            candidate_scores.append((score, line))

        if candidate_scores:
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            best_line = candidate_scores[0][1]
            return best_line.text.strip(), best_line.conf

        # Fallback to first line
        first_line = lines[0]
        return first_line.text.strip(), first_line.conf

    def _extract_date(self, lines: List[OCRLine], tokens: List[OCRToken]) -> Tuple[str, float]:
        date_patterns = [
            r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b",
            r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
            r"\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b",
            r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b",
        ]

        for line in lines:
            line_str = line.text
            for pattern in date_patterns:
                match = re.search(pattern, line_str)
                if match:
                    candidate = match.group(1)
                    parsed_date = self._normalize_date(candidate)
                    if parsed_date:
                        matching_tokens = [t for t in line.tokens if t.clean_text in candidate or any(c in t.clean_text for c in candidate.split())]
                        conf = sum(t.conf for t in matching_tokens) / len(matching_tokens) if matching_tokens else line.conf
                        return parsed_date, conf

        # Fuzzy line scan fallback
        for line in lines:
            if "date" in line.text.lower():
                try:
                    dt = date_parser.parse(line.text, fuzzy=True)
                    return dt.strftime("%Y-%m-%d"), line.conf
                except (ValueError, OverflowError):
                    continue

        return "", 0.0

    def _normalize_date(self, date_str: str) -> Optional[str]:
        try:
            dt = date_parser.parse(date_str)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return None

    def _extract_total(self, lines: List[OCRLine], tokens: List[OCRToken]) -> Tuple[float, str, float]:
        if not lines:
            return 0.0, "USD", 0.0

        anchor_candidates: List[Tuple[float, str, float, int]] = []
        for idx, line in enumerate(lines):
            line_lower = line.text.lower()
            if any(re.search(r"\b" + re.escape(kw) + r"\b", line_lower) for kw in self.TOTAL_KEYWORDS):
                if any(sub_kw in line_lower for sub_kw in self.SUBTOTAL_KEYWORDS):
                    continue
                amt, curr = self._parse_currency_amount(line.text)
                if amt is not None:
                    pos = line.top if line.top > 0 else idx
                    anchor_candidates.append((amt, curr, line.conf, pos))
                elif idx + 1 < len(lines):
                    # Check next adjacent line or same horizontal level
                    next_line = lines[idx + 1]
                    next_amt, next_curr = self._parse_currency_amount(next_line.text)
                    if next_amt is not None:
                        pos = next_line.top if next_line.top > 0 else idx + 1
                        anchor_candidates.append((next_amt, next_curr, next_line.conf, pos))

        if anchor_candidates:
            # Pick the lowest (nearest to receipt bottom) total line with an anchor
            anchor_candidates.sort(key=lambda x: x[3], reverse=True)
            return anchor_candidates[0][0], anchor_candidates[0][1], anchor_candidates[0][2]

        # Fallback: Find maximum numeric monetary value near bottom 50%
        max_bottom = max((t.bottom for t in tokens), default=1)
        bottom_half_cutoff = max_bottom * 0.50
        bottom_amounts: List[Tuple[float, str, float, int]] = []

        for idx, line in enumerate(lines):
            if max_bottom <= 1 or line.top >= bottom_half_cutoff:
                amt, curr = self._parse_currency_amount(line.text)
                if amt is not None and amt > 0.0:
                    pos = line.top if line.top > 0 else idx
                    bottom_amounts.append((amt, curr, line.conf, pos))

        if bottom_amounts:
            # Sort primarily by amount descending, then by position descending
            bottom_amounts.sort(key=lambda x: (x[0], x[3]), reverse=True)
            return bottom_amounts[0][0], bottom_amounts[0][1], bottom_amounts[0][2]

        return 0.0, "USD", 0.0

    def _extract_labeled_amount(self, lines: List[OCRLine], keywords: List[str]) -> Tuple[Optional[float], float]:
        for idx, line in enumerate(lines):
            line_lower = line.text.lower()
            if any(re.search(r"\b" + re.escape(kw) + r"\b", line_lower) for kw in keywords):
                amt, _ = self._parse_currency_amount(line.text)
                if amt is not None:
                    return amt, line.conf
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    next_amt, _ = self._parse_currency_amount(next_line.text)
                    if next_amt is not None:
                        return next_amt, next_line.conf
        return None, 0.0

    def _extract_line_items(self, lines: List[OCRLine], currency: str = "USD") -> List[Dict[str, Any]]:
        line_items: List[Dict[str, Any]] = []
        excluded_kws = self.TOTAL_KEYWORDS + self.SUBTOTAL_KEYWORDS + self.TAX_KEYWORDS + self.TIP_KEYWORDS + [
            "cash", "change", "visa", "mastercard", "amex", "card", "subtotal", "balance", "thank you"
        ]

        for line in lines:
            text = line.text.strip()
            line_lower = text.lower()
            if any(kw in line_lower for kw in excluded_kws):
                continue

            matches = list(self.CURRENCY_REGEX.finditer(text))
            if not matches:
                continue

            last_match = matches[-1]
            raw_val = last_match.group(2).replace(",", "")
            try:
                price = float(raw_val)
            except ValueError:
                continue

            usd_price = self._convert_to_usd(price, currency)

            item_name = text[:last_match.start()].strip()
            item_name = re.sub(r"^[\d\*\.\-\#\s]+", "", item_name).strip()
            if len(item_name) >= 2 and price > 0.0:
                line_items.append({
                    "name": item_name,
                    "price": round(usd_price, 2),
                    "original_price": round(price, 2),
                    "confidence": round(line.conf, 2),
                })

        return line_items

    def _classify_category(self, vendor: str, line_items: List[Dict[str, Any]], full_text: str) -> str:
        text_corpus = f"{vendor} {' '.join(item['name'] for item in line_items)} {full_text}".lower()

        scores: Dict[str, int] = {cat: 0 for cat in self.CATEGORY_KEYWORDS}
        for category, kws in self.CATEGORY_KEYWORDS.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", text_corpus):
                    scores[category] += 2 if kw in vendor.lower() else 1

        if not scores:
            return "Miscellaneous"
        best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best_category] > 0:
            return best_category
        return "Miscellaneous"

    # -----------------------------------------------------------------------
    # Policy Evaluation Rules
    # -----------------------------------------------------------------------

    def _check_prohibited_keywords(self, extracted_data: ExtractedReceiptData, raw_text: str) -> List[str]:
        corpus_elements = [raw_text, extracted_data.vendor]
        for item in extracted_data.line_items:
            corpus_elements.append(item.get("name", ""))
        full_corpus = " ".join(corpus_elements).lower()

        detected = []
        for kw in self.PROHIBITED_KEYWORDS:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, full_corpus):
                detected.append(kw)
        return detected

    def _calculate_daily_meal_spend(
        self,
        extracted_data: ExtractedReceiptData,
        historical_receipts: Optional[pd.DataFrame],
        employee_id: Optional[str],
    ) -> float:
        current_date = extracted_data.date
        current_amount = extracted_data.total_amount

        if historical_receipts is None or historical_receipts.empty or not current_date:
            return current_amount

        df = historical_receipts.copy()
        if "date" not in df.columns or "total_amount" not in df.columns:
            return current_amount

        df["date_str"] = df["date"].astype(str).str.strip()
        filtered = df[df["date_str"] == current_date]

        if employee_id and "employee_id" in df.columns:
            filtered = filtered[filtered["employee_id"].astype(str) == str(employee_id)]

        if "category" in df.columns:
            filtered = filtered[filtered["category"].astype(str).str.lower() == "meal"]

        hist_sum = float(filtered["total_amount"].sum())
        return hist_sum + current_amount

    def _verify_math(self, extracted_data: ExtractedReceiptData) -> bool:
        if extracted_data.subtotal is None and extracted_data.tax is None and extracted_data.tip is None:
            return True

        subtotal = extracted_data.subtotal if extracted_data.subtotal is not None else 0.0
        tax = extracted_data.tax if extracted_data.tax is not None else 0.0
        tip = extracted_data.tip if extracted_data.tip is not None else 0.0
        calculated_total = subtotal + tax + tip

        # Compare against USD converted total or original raw total
        return (
            abs(calculated_total - extracted_data.total_amount) <= 0.05
            or (extracted_data.original_total > 0.0 and abs(calculated_total - extracted_data.original_total) <= 0.05)
        )

    def _check_duplicate(
        self,
        extracted_data: ExtractedReceiptData,
        historical_receipts: Optional[pd.DataFrame],
    ) -> bool:
        if historical_receipts is None or historical_receipts.empty:
            return False

        if "original_total" in historical_receipts.columns:
            amount_col = "original_total"
            cur_amount = extracted_data.original_total
        elif "total_amount" in historical_receipts.columns:
            amount_col = "total_amount"
            cur_amount = (
                extracted_data.original_total
                if extracted_data.original_total > 0.0
                else extracted_data.total_amount
            )
        else:
            return False

        if not {"vendor", "date"}.issubset(set(historical_receipts.columns)):
            return False

        cur_vendor = extracted_data.vendor.strip().lower()
        cur_date = extracted_data.date.strip()

        df = historical_receipts.copy()
        df["_hist_vendor"] = df["vendor"].astype(str).str.strip().str.lower()
        df["_hist_date"] = df["date"].astype(str).str.strip()
        df["_hist_amount"] = pd.to_numeric(df[amount_col], errors="coerce")

        mask = (
            (df["_hist_vendor"] == cur_vendor)
            & (df["_hist_date"] == cur_date)
            & ((df["_hist_amount"] - cur_amount).abs() <= 0.01)
        )
        return bool(mask.any())
