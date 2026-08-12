from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app.utils.tools as tools


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_backup_file_creates_copy(self):
        source = self.root / "sample.txt"
        source.write_text("hello", encoding="utf-8")

        with patch.object(tools, "BACKUP_DIR", str(self.root / "backup")):
            backup_path = tools.backup_file(str(source))

        self.assertTrue(backup_path)
        self.assertTrue(Path(backup_path).exists())
        self.assertEqual("hello", Path(backup_path).read_text(encoding="utf-8"))

    def test_read_write_file_round_trip(self):
        target = self.root / "sample.txt"

        with patch.object(tools, "BACKUP_DIR", str(self.root / "backup")):
            result = tools.write_file(str(target), "hello world")

        self.assertTrue(result)
        self.assertEqual("hello world", tools.read_file(str(target)))

    def test_compile_check_reports_valid_python(self):
        target = self.root / "valid.py"
        target.write_text("x = 1\n", encoding="utf-8")

        ok, error = tools.compile_check(str(target))

        self.assertTrue(ok)
        self.assertEqual("", error)

    def test_compile_check_reports_invalid_python(self):
        target = self.root / "invalid.py"
        target.write_text("def broken(:\n", encoding="utf-8")

        ok, error = tools.compile_check(str(target))

        self.assertFalse(ok)
        self.assertIn("SyntaxError", error)

    def test_restart_invokes_restart_script(self):
        with patch.object(tools.subprocess, "Popen") as popen:
            tools.restart()

        popen.assert_called_once_with(["./restart.sh"], shell=False)


if __name__ == "__main__":
    unittest.main()
