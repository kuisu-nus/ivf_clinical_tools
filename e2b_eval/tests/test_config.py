"""Environment configuration and secret redaction."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from e2b_eval.config import AgentLLMConfig, ConfigError, E2BConfig
from e2b_eval.util import case_key, redact


class E2BConfigTests(unittest.TestCase):
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigError, "E2B_API_KEY"):
                E2BConfig.from_env()

    def test_self_hosted_needs_sandbox_url(self):
        with patch.dict(os.environ, {"E2B_API_KEY": "k", "E2B_API_URL": "http://host:3000"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "E2B_SANDBOX_URL"):
                E2BConfig.from_env()

    def test_full_self_hosted_config(self):
        env = {
            "E2B_API_KEY": "k",
            "E2B_API_URL": "http://host:3000",
            "E2B_SANDBOX_URL": "http://host:3002",
            "E2B_TEMPLATE": "tpl",
            "E2B_SANDBOX_TIMEOUT_SEC": "123",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = E2BConfig.from_env()
            self.assertEqual(cfg.template, "tpl")
            self.assertEqual(cfg.timeout_sec, 123)
            cfg.export_env()
            self.assertEqual(os.environ["E2B_SANDBOX_URL"], "http://host:3002")

    def test_default_template_requires_port_31000(self):
        env = {
            "E2B_API_KEY": "k",
            "E2B_API_URL": "http://host:3000",
            "E2B_SANDBOX_URL": "http://host:3002",
            "DRESSAGE_E2B_BLACKBOX_PORT": "32000",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "31000"):
                E2BConfig.from_env()

    def test_rejects_non_positive_timeout(self):
        env = {
            "E2B_API_KEY": "k",
            "E2B_SANDBOX_TIMEOUT_SEC": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "timeout"):
                E2BConfig.from_env()


class AgentConfigTests(unittest.TestCase):
    def test_openclaw_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigError, "AGENT_LLM_API_KEY"):
                AgentLLMConfig.from_env(mode="openclaw")

    def test_mock_does_not_require_key(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = AgentLLMConfig.from_env(mode="mock")
            self.assertEqual(cfg.mode, "mock")
            self.assertEqual(cfg.api_key, "mock")

    def test_openclaw_reads_env(self):
        env = {
            "AGENT_LLM_API_KEY": "sk-test",
            "AGENT_LLM_BASE_URL": "http://localhost:8000/v1",
            "AGENT_LLM_MODEL": "m",
            "AGENT_MAX_STEPS": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AgentLLMConfig.from_env(mode="openclaw")
            self.assertEqual(cfg.api_key, "sk-test")
            self.assertEqual(cfg.base_url, "http://localhost:8000/v1")
            self.assertEqual(cfg.model, "m")
            self.assertIsNone(cfg.max_steps)

    def test_unknown_mode_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigError, "agent mode"):
                AgentLLMConfig.from_env(mode="other", require_key=False)


class RedactionTests(unittest.TestCase):
    def test_redacts_secrets_but_not_token_limits(self):
        value = {
            "api_key": "secret",
            "backend": {"Authorization": "Bearer secret", "max_tokens": 4096, "input_tokens": 10},
            "nested": [{"agent_api_key": "secret"}],
        }
        cleaned = redact(value)
        self.assertEqual(cleaned["api_key"], "<redacted>")
        self.assertEqual(cleaned["backend"]["Authorization"], "<redacted>")
        self.assertEqual(cleaned["backend"]["max_tokens"], 4096)
        self.assertEqual(cleaned["backend"]["input_tokens"], 10)
        self.assertEqual(cleaned["nested"][0]["agent_api_key"], "<redacted>")

    def test_case_key(self):
        self.assertEqual(case_key("claw/ivf:0974", index=0), "claw-ivf-0974")
        self.assertEqual(case_key("", index=3), "case-0003")


if __name__ == "__main__":
    unittest.main()
