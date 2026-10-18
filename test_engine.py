import unittest
import pandas as pd
from engine import ReceiptAuditEngine, AuditResult


class TestReceiptAuditEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ReceiptAuditEngine(confidence_threshold=60.0)

    def _build_ocr_dict(self, lines_data):
        """
        lines_data: list of tuples: (line_num, top, height, [(text, conf, left, width)])
        """
        ocr = {
            "text": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "line_num": [],
            "word_num": [],
            "block_num": [],
        }
        for line_num, top, height, tokens in lines_data:
            for word_num, (text, conf, left, width) in enumerate(tokens, start=1):
                ocr["text"].append(text)
                ocr["conf"].append(conf)
                ocr["left"].append(left)
                ocr["top"].append(top)
                ocr["width"].append(width)
                ocr["height"].append(height)
                ocr["line_num"].append(line_num)
                ocr["word_num"].append(word_num)
                ocr["block_num"].append(1)
        return ocr

    def test_clean_receipt_approval(self):
        # Sample clean bistro meal under $50
        lines = [
            (1, 20, 30, [("Corner", 95, 100, 80), ("Bistro", 95, 190, 80)]),
            (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 92, 100, 80)]),
            (3, 100, 15, [("Grilled", 88, 50, 60), ("Chicken", 88, 115, 60), ("25.00", 90, 300, 50)]),
            (4, 130, 15, [("Subtotal", 90, 50, 70), ("25.00", 92, 300, 50)]),
            (5, 150, 15, [("Tax", 90, 50, 40), ("2.50", 90, 300, 50)]),
            (6, 170, 15, [("Tip", 85, 50, 30), ("4.00", 88, 300, 50)]),
            (7, 200, 20, [("Total", 95, 50, 50), ("$31.50", 95, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)
        result = self.engine.audit(ocr, employee_id="EMP001", attendee_count=1)

        self.assertIsInstance(result, AuditResult)
        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(result.extracted_data.vendor, "Corner Bistro")
        self.assertEqual(result.extracted_data.date, "2026-10-17")
        self.assertEqual(result.extracted_data.total_amount, 31.50)
        self.assertEqual(result.extracted_data.subtotal, 25.00)
        self.assertEqual(result.extracted_data.tax, 2.50)
        self.assertEqual(result.extracted_data.tip, 4.00)
        self.assertEqual(result.extracted_data.category, "Meal")
        self.assertTrue(result.math_verified)
        self.assertFalse(result.confidence_flags["total_amount"].requires_review)
        self.assertFalse(result.confidence_flags["date"].requires_review)
        self.assertFalse(result.confidence_flags["vendor"].requires_review)

    def test_single_meal_cap_exceeded(self):
        lines = [
            (1, 20, 30, [("Grand", 92, 100, 70), ("Cafe", 92, 180, 60)]),
            (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 92, 100, 80)]),
            (3, 100, 15, [("Steak", 90, 50, 50), ("Dinner", 90, 110, 50), ("65.00", 90, 300, 50)]),
            (4, 150, 20, [("Total", 95, 50, 50), ("$65.00", 95, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)

        # Single attendee -> flagged
        result_single = self.engine.audit(ocr, employee_id="EMP001", attendee_count=1)
        self.assertEqual(result_single.status, "FLAGGED_FOR_REVIEW")
        self.assertTrue(any("exceeds individual cap of $50.00" in r for r in result_single.verdict_reasons))

        # Multiple attendees (> 1) -> approved
        result_multi = self.engine.audit(ocr, employee_id="EMP001", attendee_count=2)
        self.assertEqual(result_multi.status, "APPROVED")

    def test_daily_meal_limit_exceeded(self):
        history = pd.DataFrame([
            {"employee_id": "EMP001", "date": "2026-10-17", "vendor": "Coffee Shop", "total_amount": 75.00, "category": "Meal"}
        ])
        lines = [
            (1, 20, 30, [("Burger", 90, 100, 70), ("Joint", 90, 180, 60)]),
            (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 92, 100, 80)]),
            (3, 100, 20, [("Total", 95, 50, 50), ("$35.00", 95, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)
        result = self.engine.audit(ocr, historical_receipts=history, employee_id="EMP001", attendee_count=1)

        self.assertEqual(result.status, "FLAGGED_FOR_REVIEW")
        self.assertTrue(any("Total daily meal expenses" in r and "exceeding the $100.00 daily limit" in r for r in result.verdict_reasons))

    def test_prohibited_keyword_rejection(self):
        banned_tests = ["Beer", "Red Wine", "Cocktail", "Tobacco", "Casino night", "Gift card"]
        for banned in banned_tests:
            lines = [
                (1, 20, 30, [("City", 90, 100, 50), ("Grill", 90, 160, 50)]),
                (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 92, 100, 80)]),
                (3, 100, 15, [(banned, 90, 50, 80), ("15.00", 90, 300, 50)]),
                (4, 150, 20, [("Total", 95, 50, 50), ("$15.00", 95, 300, 60)]),
            ]
            ocr = self._build_ocr_dict(lines)
            result = self.engine.audit(ocr, employee_id="EMP001")
            self.assertEqual(result.status, "REJECTED", f"Failed to reject for {banned}")
            self.assertTrue(any("Prohibited item/activity detected" in r for r in result.verdict_reasons))

    def test_mathematical_discrepancy(self):
        lines = [
            (1, 20, 30, [("Metro", 90, 100, 60), ("Diner", 90, 170, 60)]),
            (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 92, 100, 80)]),
            (3, 100, 15, [("Subtotal", 90, 50, 70), ("20.00", 90, 300, 50)]),
            (4, 120, 15, [("Tax", 90, 50, 40), ("2.00", 90, 300, 50)]),
            (5, 140, 15, [("Tip", 90, 50, 40), ("3.00", 90, 300, 50)]),
            # 20 + 2 + 3 = 25.00, but total says 30.00
            (6, 170, 20, [("Total", 95, 50, 50), ("$30.00", 95, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)
        result = self.engine.audit(ocr, employee_id="EMP001")
        self.assertFalse(result.math_verified)
        self.assertEqual(result.status, "FLAGGED_FOR_REVIEW")
        self.assertTrue(any("Mathematical discrepancy" in r for r in result.verdict_reasons))

    def test_duplicate_check(self):
        history = pd.DataFrame([
            {"vendor": "Corner Bistro", "date": "2026-10-17", "total_amount": 42.00}
        ])
        lines = [
            (1, 20, 30, [("Corner", 90, 100, 60), ("Bistro", 90, 170, 60)]),
            (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 92, 100, 80)]),
            (3, 100, 20, [("Total", 95, 50, 50), ("$42.00", 95, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)
        result = self.engine.audit(ocr, historical_receipts=history)
        self.assertEqual(result.status, "FLAGGED_FOR_REVIEW")
        self.assertTrue(any("DUPLICATE_SUSPECTED" in r for r in result.verdict_reasons))

    def test_low_confidence_flagging(self):
        # Total amount has confidence 45 (< 60)
        lines = [
            (1, 20, 30, [("Corner", 90, 100, 60), ("Bistro", 90, 170, 60)]),
            (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 90, 100, 80)]),
            (3, 100, 20, [("Total", 45, 50, 50), ("$22.00", 45, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)
        result = self.engine.audit(ocr)
        self.assertEqual(result.status, "FLAGGED_FOR_REVIEW")
        self.assertTrue(result.confidence_flags["total_amount"].requires_review)
        self.assertEqual(result.confidence_flags["total_amount"].conf, 45.0)

    def test_category_classification(self):
        test_cases = [
            ("Lyft Ride", "Travel/Transit"),
            ("Staples Store #102", "Office Supplies"),
            ("Hilton Garden Inn", "Lodging"),
            ("Starbucks Coffee", "Meal"),
        ]
        for vendor_name, expected_cat in test_cases:
            lines = [
                (1, 20, 30, [(w, 90, 100 + i * 60, 50) for i, w in enumerate(vendor_name.split())]),
                (2, 60, 15, [("Date:", 90, 50, 40), ("2026-10-17", 90, 100, 80)]),
                (3, 100, 20, [("Total", 90, 50, 50), ("$25.00", 90, 300, 60)]),
            ]
            ocr = self._build_ocr_dict(lines)
            result = self.engine.audit(ocr)
            self.assertEqual(result.extracted_data.category, expected_cat)

    def test_currencies_and_date_variants(self):
        currencies = ["€", "£", "₹", "$"]
        for curr in currencies:
            lines = [
                (1, 20, 30, [("Express", 90, 100, 60), ("Cab", 90, 170, 40)]),
                (2, 60, 15, [("Date:", 90, 50, 40), ("17/10/2026", 90, 100, 80)]),
                (3, 100, 20, [("Total", 90, 50, 50), (f"{curr}45.50", 90, 300, 60)]),
            ]
            ocr = self._build_ocr_dict(lines)
            result = self.engine.audit(ocr)
            self.assertEqual(result.extracted_data.total_amount, 45.50)
            self.assertEqual(result.extracted_data.date, "2026-10-17")
            self.assertEqual(result.extracted_data.category, "Travel/Transit")

    def test_json_serialization(self):
        lines = [
            (1, 20, 30, [("Corner", 92, 100, 60), ("Bistro", 92, 170, 60)]),
            (2, 60, 15, [("Date:", 54, 50, 40), ("2026-10-17", 54, 100, 80)]),
            (3, 100, 20, [("Total", 88, 50, 50), ("$45.50", 88, 300, 60)]),
        ]
        ocr = self._build_ocr_dict(lines)
        result = self.engine.audit(ocr)
        dump = result.model_dump()
        self.assertIn("status", dump)
        self.assertIn("verdict_reasons", dump)
        self.assertIn("confidence_flags", dump)
        self.assertIn("extracted_data", dump)
        self.assertIn("math_verified", dump)
        self.assertIn("audit_timestamp", dump)
        self.assertTrue(dump["confidence_flags"]["date"]["requires_review"])
        self.assertFalse(dump["confidence_flags"]["total_amount"]["requires_review"])



if __name__ == "__main__":
    unittest.main()
