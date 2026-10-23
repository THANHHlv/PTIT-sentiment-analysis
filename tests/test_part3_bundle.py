"""Integrity checks for the portable handoff before loading a model."""
import tempfile
import unittest
from pathlib import Path

from ptit_sentiment.common import file_hash, write_json
from scripts.verify_part3_bundle import bundle_path, verify_bundle


class Part3BundleTest(unittest.TestCase):
    def test_modified_member_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = root / "requirements-part3.txt"
            member.write_text("joblib==1.4.2\n", encoding="utf-8")
            write_json(root / "bundle_manifest.json", {"files_sha256": {member.name: file_hash(member)}})
            member.write_text("joblib==0.0.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_bundle(root)

    def test_manifest_cannot_reference_outside_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "ngoài gói"):
                bundle_path(directory, "../outside.joblib")


if __name__ == "__main__":
    unittest.main()
