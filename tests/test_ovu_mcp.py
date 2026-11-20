"""The oocyte-yield tools over MCP: the OVU specs served by task/runtime/claw_tools_mcp.py (stdio),
which forwards each call to the OVU HTTP service (synthetic data)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OVU_DB_BACKEND", "sqlite")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "task" / "services"))
sys.path.insert(0, str(ROOT / "tests"))

import uvicorn  # noqa: E402

from test_ovu_ml import prepared  # noqa: E402
from test_ovu_services import load_service  # noqa: E402

BRIDGE = ROOT / "task" / "runtime" / "claw_tools_mcp.py"
LOCAL = "127.0.0.1,localhost"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class McpTest(unittest.TestCase):
    """initialize, tools/list and tools/call through the bridge, as OpenClaw sends them."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.data = prepared(Path(cls.tmp.name), 2500)
        for module in ("ovu.ml.train", "ovu.similar.build", "ovu.db.build"):
            done = subprocess.run([sys.executable, "-m", module, "--data", str(cls.data)], cwd=ROOT, capture_output=True, text=True)
            assert done.returncode == 0, done.stderr
        server, specs = load_service(cls.data)
        cls.tools_dir = Path(cls.tmp.name) / "ovu_tools"
        specs.main(["--tools", str(cls.tools_dir)])
        cls.port = free_port()
        cls.http = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=cls.port, log_level="warning"))
        threading.Thread(target=cls.http.run, daemon=True).start()
        deadline = time.time() + 60
        while not cls.http.started:
            assert time.time() < deadline, "the OVU service did not start"
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.http.should_exit = True
        time.sleep(0.2)
        cls.tmp.cleanup()
        os.environ.pop("OVU_DATA_DIR", None)
        os.environ.pop("OVU_HIDE_FOLD", None)

    def session(self, *requests: dict) -> dict:
        """Send JSON-RPC requests to the bridge over stdio; answers by request id."""
        env = {**os.environ, "OMNI_MCP_SERVER_NAME": "ovu-tools", "OMNI_TOOL_SPECS_DIR": str(self.tools_dir),
               "OMNI_TRACE_DIR": str(Path(self.tmp.name) / "trace"), "OMNI_TOOL_PORT_MAP": f"9211:{self.port}",
               "NO_PROXY": LOCAL, "no_proxy": LOCAL}
        lines = "".join(json.dumps({"jsonrpc": "2.0", "id": i, **r}, ensure_ascii=False) + "\n"
                        for i, r in enumerate(requests, 1))
        done = subprocess.run([sys.executable, str(BRIDGE)], input=lines, env=env, capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return {m["id"]: m for m in map(json.loads, done.stdout.splitlines())}

    def call(self, name: str, arguments: dict) -> tuple[bool, dict]:
        answer = self.session({"method": "tools/call", "params": {"name": name, "arguments": arguments}})[1]["result"]
        return answer["isError"], json.loads(answer["content"][0]["text"])

    def test_initialize_and_list_the_five_tools(self) -> None:
        answers = self.session({"method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                               {"method": "tools/list"})
        self.assertEqual(answers[1]["result"]["serverInfo"]["name"], "ovu-tools")
        tools = {t["name"]: t for t in answers[2]["result"]["tools"]}
        self.assertEqual(sorted(tools), ["ovu_case_sql", "ovu_data_overview", "ovu_predict_oocytes",
                                         "ovu_similar_cases", "ovu_sql_examples"])
        self.assertIn("age", tools["ovu_predict_oocytes"]["inputSchema"]["properties"])
        self.assertNotIn("backend", tools["ovu_predict_oocytes"])

    def test_calls_reach_the_service(self) -> None:
        error, body = self.call("ovu_predict_oocytes", {"age": 35, "amh": 2.1})
        self.assertFalse(error, body)
        self.assertAlmostEqual(sum(body["各分组概率"].values()), 1.0, places=2)
        error, body = self.call("ovu_similar_cases", {"age": 30, "amh": 3.0, "k": 30})
        self.assertFalse(error, body)
        self.assertEqual(sum(g["例数"] for g in body["获卵数分组"].values()), 30)
        error, body = self.call("ovu_case_sql", {"sql": "SELECT COUNT(*) AS n FROM cases"})
        self.assertFalse(error, body)
        self.assertEqual(body["返回行数"], 1)

    def test_refusals_come_back_as_tool_errors(self) -> None:
        error, body = self.call("ovu_predict_oocytes", {"age": 500})
        self.assertTrue(error)
        self.assertEqual(body["error"], "invalid_arguments")
        error, body = self.call("ovu_case_sql", {"sql": "DELETE FROM cases"})
        self.assertTrue(error)


if __name__ == "__main__":
    unittest.main()
