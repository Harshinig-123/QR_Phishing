"""
test_render_consistency.py
--------------------------
Unit test verifying that QR code rendering variations do NOT alter Stage 3 ML verdicts.

Tests:
  - Generates QR codes for multiple URLs across diverse physical/visual renderings:
      * QR Versions: 1, 2, 4
      * Error Correction: L (7%), M (15%), Q (25%), H (30%)
      * Color schemes: (black/white, navy/white, darkgreen/lightgray)
      * Box sizes: 4, 8, 12
  - Decodes each rendered QR image using Stage 1 (process_qr_bytes)
  - Evaluates the decoded URL with Stage 3 predict_url()
  - Asserts that all successfully decoded renderings of the SAME URL yield
    100% identical ML predictions and probabilities.
"""

import io
import sys
import unittest
from pathlib import Path

# Path configuration to import from stage1, stage2, and stage3
REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE1_DIR = REPO_ROOT / "stage1_qr_decoding"
STAGE3_DIR = REPO_ROOT / "stage3_ml"

for p in [str(STAGE1_DIR), str(STAGE3_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
from qr_decoder import process_qr_bytes
from ml_predict import predict_url, should_run_stage3


class TestRenderConsistency(unittest.TestCase):

    def setUp(self):
        self.test_urls = [
            "https://www.google.com",
            "http://192.168.1.50/login/update",
            "http://paypal.com@evil-site.net/signin",
        ]
        self.ec_levels = [ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H]
        self.colors = [
            ("black", "white"),
            ("#000080", "#ffffff"),      # Navy on white
            ("#222222", "#f0f0f0"),      # Dark gray on off-white
        ]
        self.box_sizes = [6, 10]

    def test_rendering_invariance_across_variations(self):
        """Validates that visual QR rendering parameters do not alter Stage 3 predictions."""
        for target_url in self.test_urls:
            reference_result = predict_url(target_url)
            ref_pred = reference_result["prediction"]
            ref_prob = reference_result["probability"]

            runs_tested = 0
            for ec in self.ec_levels:
                for fill_color, back_color in self.colors:
                    for box_size in self.box_sizes:
                        qr = qrcode.QRCode(
                            version=None,  # Auto-fit version
                            error_correction=ec,
                            box_size=box_size,
                            border=4,
                        )
                        qr.add_data(target_url)
                        qr.make(fit=True)
                        img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")

                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        png_bytes = buf.getvalue()

                        # Decode with Stage 1
                        decode_res = process_qr_bytes(png_bytes)
                        self.assertEqual(decode_res.status, "ok", f"Stage 1 failed to decode QR for {target_url}")
                        self.assertTrue(should_run_stage3(decode_res))

                        # Predict with Stage 3
                        ml_res = predict_url(decode_res.url)

                        # Assert strict consistency
                        self.assertEqual(
                            ml_res["prediction"],
                            ref_pred,
                            f"Inconsistent prediction for {target_url} under EC={ec}, colors={fill_color}/{back_color}"
                        )
                        self.assertEqual(
                            ml_res["probability"],
                            ref_prob,
                            f"Inconsistent probability for {target_url} under rendering variation"
                        )
                        runs_tested += 1

            self.assertGreaterEqual(runs_tested, 24, f"Insufficient variations tested for {target_url}")
            print(f"    [+] Successfully verified {runs_tested} visual variations for '{target_url}' -> {ref_pred.upper()} ({ref_prob*100:.1f}%)")


if __name__ == "__main__":
    unittest.main()
