"""Sandbox endpoint and BlackboxServer client unit tests (no E2B calls)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from e2b_eval.blackbox import BlackboxClient, BlackboxError
from e2b_eval.samples import Upload
from e2b_eval.sandbox import Endpoint, Sandbox


class FakeRaw:
    sandbox_id = "sb-test"

    def __init__(self) -> None:
        self.files = FakeFiles()
        self.commands = FakeCommands()

    def get_host(self, port: int) -> str:
        return f"{port}-sb-test.e2b.app"


class FakeFiles:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    async def write_files(self, entries):
        self.entries.extend(entries)
        return []

    async def read(self, path, format="text"):
        return f"content:{path}"


class FakeCommands:
    async def run(self, cmd, **kwargs):
        class Result:
            stdout = "done"
            stderr = ""
            exit_code = 0
            error = None

        return Result()


class SandboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_uses_host_header_for_self_hosted_proxy(self):
        sandbox = Sandbox(
            FakeRaw(),
            sandbox_proxy_url="http://proxy:3002/",
            blackbox_port=31000,
        )
        endpoint = await sandbox.endpoint()
        self.assertEqual(endpoint.url, "http://proxy:3002")
        self.assertEqual(endpoint.headers["Host"], "31000-sb-test.e2b.app")

    async def test_endpoint_direct_host_when_no_proxy(self):
        sandbox = Sandbox(FakeRaw(), sandbox_proxy_url=None, blackbox_port=31000)
        endpoint = await sandbox.endpoint()
        self.assertEqual(endpoint.url, "https://31000-sb-test.e2b.app")
        self.assertEqual(endpoint.headers, {})

    async def test_write_files_reads_local_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("hello", encoding="utf-8")
            sandbox = Sandbox(FakeRaw(), sandbox_proxy_url=None, blackbox_port=31000)
            uploaded = await sandbox.write_files(
                [Upload(sandbox_path="/tmp/a.txt", local_path=path)]
            )
            self.assertEqual(uploaded[0]["path"], "/tmp/a.txt")
            self.assertEqual(sandbox.raw.files.entries[0]["data"], b"hello")

    async def test_run_normalizes_result(self):
        sandbox = Sandbox(FakeRaw(), sandbox_proxy_url=None, blackbox_port=31000)
        result = await sandbox.run("echo hi")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "done")


class BlackboxClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_multiline_execute_cmd(self):
        client = BlackboxClient(Endpoint("http://127.0.0.1:9", {}))
        try:
            with self.assertRaisesRegex(BlackboxError, "single-line"):
                await client.execute_cmd("sid", "echo a\necho b")
        finally:
            await client.close()

    async def test_http_error_is_wrapped(self):
        client = BlackboxClient(Endpoint("http://127.0.0.1:9", {}))
        try:
            with self.assertRaises(BlackboxError):
                await client.health()
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
