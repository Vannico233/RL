import json
from pathlib import Path
import unittest

from chart_rl.framework_source import verify_framework_source


class FrameworkSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.config = json.loads((cls.project / "configs/project.json").read_text(encoding="utf-8"))

    def test_archive_requires_correct_hash_and_expected_source(self):
        direct = {"url": self.config["framework_source_url"], "archive_info": {
            "hashes": {"sha256": self.config["framework_archive_sha256"]}}}
        result = verify_framework_source(direct, self.config)
        self.assertTrue(result["verified"])
        self.assertEqual(result["commit"], self.config["framework_commit"])
        direct["archive_info"]["hashes"]["sha256"] = "0" * 64
        self.assertFalse(verify_framework_source(direct, self.config)["verified"])
        direct["archive_info"]["hashes"]["sha256"] = self.config["framework_archive_sha256"]
        direct["url"] = "https://example.invalid/unknown-source.tar.gz"
        self.assertFalse(verify_framework_source(direct, self.config)["verified"])

    def test_local_verified_archive_and_existing_pinned_git_remain_supported(self):
        direct = {"url": "file:///tmp/ms-swift.tar.gz", "archive_info": {
            "hash": "sha256=" + self.config["framework_archive_sha256"]}}
        self.assertTrue(verify_framework_source(direct, self.config)["verified"])
        direct = {"url": "https://github.com/modelscope/ms-swift.git", "vcs_info": {
            "vcs": "git", "commit_id": self.config["framework_commit"]}}
        self.assertTrue(verify_framework_source(direct, self.config)["verified"])
        direct["vcs_info"]["commit_id"] = "0" * 40
        self.assertFalse(verify_framework_source(direct, self.config)["verified"])
        self.assertFalse(verify_framework_source({}, self.config)["verified"])

    def test_installer_and_preflight_pin_identical_source_bytes(self):
        line = next(line for line in (self.project / "requirements-server.txt").read_text().splitlines()
                    if line.startswith("ms-swift @ "))
        self.assertEqual(line, "ms-swift @ " + self.config["framework_source_url"]
                         + "#sha256=" + self.config["framework_archive_sha256"])


if __name__ == "__main__":
    unittest.main()
