import importlib
import os
import unittest
from unittest.mock import patch


class ConfigTest(unittest.TestCase):
    def load_settings(self, env):
        # Block pydantic-settings from reading .env so defaults are tested
        with patch("os.path.isfile", side_effect=lambda p, _orig=os.path.isfile: False if str(p).endswith(".env") else _orig(p)):
            with patch.dict(os.environ, env, clear=True):
                import app.config.settings as settings

                return importlib.reload(settings)

    def test_default_settings_are_loaded(self):
        settings = self.load_settings({})

        self.assertEqual("http://localhost:11434/v1", settings.OLLAMA_BASE_URL)
        self.assertEqual("ornith:9b", settings.MODEL)
        self.assertEqual("", settings.BOT_TOKEN)
        self.assertEqual("data/faiss_index", settings.RAG_INDEX_DIR)
        self.assertEqual(4, settings.RAG_TOP_K)
        self.assertEqual(["python3", "venv/bin/python", "ls", "pwd", "cat", "rg", "claude", "aider"], settings.ALLOWED_SHELL_COMMANDS)
        self.assertEqual([], settings.TELEGRAM_ALLOWED_USER_IDS)

    def test_environment_overrides_are_parsed(self):
        settings = self.load_settings(
            {
                "OLLAMA_BASE_URL": "http://example.com/v1",
                "MODEL": "test-model",
                "BOT_TOKEN": "secret",
                "TELEGRAM_ALLOWED_USER_IDS": "1, 2,3",
                "RAG_INDEX_DIR": "indexes/faiss",
                "RAG_TOP_K": "8",
                "ALLOWED_FILE_DIRS": "/tmp/a,/tmp/b",
                "ALLOWED_SHELL_COMMANDS": "python3,ls",
            }
        )

        self.assertEqual("http://example.com/v1", settings.OLLAMA_BASE_URL)
        self.assertEqual("test-model", settings.MODEL)
        self.assertEqual("secret", settings.BOT_TOKEN)
        self.assertEqual(["1", "2", "3"], settings.TELEGRAM_ALLOWED_USER_IDS)
        self.assertEqual("indexes/faiss", settings.RAG_INDEX_DIR)
        self.assertEqual(8, settings.RAG_TOP_K)
        self.assertEqual(["/tmp/a", "/tmp/b"], settings.ALLOWED_FILE_DIRS)
        self.assertEqual(["python3", "ls"], settings.ALLOWED_SHELL_COMMANDS)


if __name__ == "__main__":
    unittest.main()
