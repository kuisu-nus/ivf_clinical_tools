"""Tool service ports when several Dressage sandboxes (local_bwrap slots) share one host network."""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "task" / "scripts"
BRIDGE = ROOT / "task" / "runtime" / "claw_tools_mcp.py"
BASH = shutil.which("bash")
sys.path.insert(0, str(BRIDGE.parent))
import claw_tools_mcp  # noqa: E402


def port(**env):
    out = subprocess.run([BASH, "-c", f'set -euo pipefail; source "{SCRIPTS / "ports.sh"}"; echo $IVF_TOOLS_PORT'],
                         env={k: v for k, v in {**os.environ, **env}.items() if k != "IVF_TOOLS_PORT" or k in env},
                         capture_output=True, text=True, check=True)
    return int(out.stdout)


@unittest.skipUnless(BASH, "needs bash")
class PortTests(unittest.TestCase):
    def test_port_choice(self):
        self.assertEqual(port(), 9201)                                            # no slot: one sandbox per network
        self.assertEqual(port(DRESSAGE_BLACKBOX_SLOT_ID="e2b"), 9201)             # Dressage's E2B sandboxes
        self.assertEqual(port(DRESSAGE_BLACKBOX_SLOT_ID="08"), 20008)             # not read as octal
        self.assertEqual(port(DRESSAGE_BLACKBOX_SLOT_ID="3"), 20003)                # local_bwrap slot 3
        self.assertEqual(port(DRESSAGE_BLACKBOX_SLOT_ID="3", IVF_PORT_BASE="30000"), 30003)
        self.assertEqual(port(DRESSAGE_BLACKBOX_SLOT_ID="3", IVF_TOOLS_PORT="9301"), 9301)   # explicit port wins

    def test_bridge_rewrites_the_spec_ports(self):
        url = "http://127.0.0.1:9201/tools/ivf_similar_patients"
        os.environ.pop("OMNI_TOOL_PORT_MAP", None)
        self.assertEqual(claw_tools_mcp.resolve_url(url), url)
        os.environ["OMNI_TOOL_PORT_MAP"] = "9201:20003"
        try:
            self.assertEqual(claw_tools_mcp.resolve_url(url), "http://127.0.0.1:20003/tools/ivf_similar_patients")
            self.assertEqual(claw_tools_mcp.resolve_url("http://127.0.0.1:9999/x"), "http://127.0.0.1:9999/x")
        finally:
            del os.environ["OMNI_TOOL_PORT_MAP"]

    def test_register_mcp_passes_the_port_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "runtime" / "bbs-s1" / "home").mkdir(parents=True)
            stub = tmp / "openclaw"                       # records `openclaw mcp set <name> <json>`
            stub.write_text('#!/usr/bin/env bash\necho "$HOME" > "$STUB_OUT.home"; echo "$4" > "$STUB_OUT"\n')
            stub.chmod(0o755)
            env = {**os.environ, "BBS_RUNTIME_ROOT": str(tmp / "runtime"), "OPENCLAW_BIN": str(stub),
                   "STUB_OUT": str(tmp / "out"), "DRESSAGE_BLACKBOX_SLOT_ID": "2"}
            env.pop("IVF_TOOLS_PORT", None)
            subprocess.run([BASH, str(SCRIPTS / "register_mcp.sh")], env=env, check=True, capture_output=True)
            config = json.loads((tmp / "out").read_text())
            self.assertEqual(config["env"]["OMNI_TOOL_PORT_MAP"], "9201:20002")
            self.assertEqual(config["env"]["OMNI_TOOL_SPECS_DIR"], "/tmp/omni_task/tools")
            self.assertEqual((tmp / "out.home").read_text().strip(), str(tmp / "runtime" / "bbs-s1" / "home"))
            del env["DRESSAGE_BLACKBOX_SLOT_ID"]
            subprocess.run([BASH, str(SCRIPTS / "register_mcp.sh")], env=env, check=True, capture_output=True)
            self.assertNotIn("OMNI_TOOL_PORT_MAP", json.loads((tmp / "out").read_text())["env"])


@unittest.skipUnless(BASH, "needs bash")
class TwoSlotsTests(unittest.TestCase):
    """Two slots on one network, each with its own case settings, each reaching only its own service."""

    def task_dir(self):
        # Removed after the services are stopped (cleanups run last-in first-out): stop_services.sh
        # needs the pid file.
        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def start(self, slot, task_dir, **env):
        full = {**os.environ, "PYTHON": sys.executable, "OMNI_TASK_DIR": str(task_dir),
                "DRESSAGE_BLACKBOX_SLOT_ID": str(slot), "IVF_PORT_BASE": str(self.base), **env}
        full.pop("IVF_TOOLS_PORT", None)
        out = subprocess.run([BASH, str(SCRIPTS / "start_services.sh")], env=full, capture_output=True, text=True)
        self.addCleanup(subprocess.run, [BASH, str(SCRIPTS / "stop_services.sh")], env=full, capture_output=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]

    def similar_via_bridge(self, slot):
        port = self.base + slot
        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "ivf_similar_patients", "arguments": {"age": 32, "amh": 1.38}}}
        with tempfile.TemporaryDirectory() as trace:
            out = subprocess.run([sys.executable, str(BRIDGE)], input=json.dumps(message) + "\n", text=True,
                                 capture_output=True, check=True,
                                 env={**os.environ, "OMNI_TOOL_SPECS_DIR": str(ROOT / "task" / "tools"),
                                      "OMNI_TRACE_DIR": trace,
                                      "OMNI_TOOL_PORT_MAP": f"9201:{port}"})
            dispatch = json.loads((Path(trace) / "dispatches.jsonl").read_text().splitlines()[0])
        self.assertEqual(dispatch["endpoint_url"], f"http://127.0.0.1:{port}/tools/ivf_similar_patients")
        return json.loads(json.loads(out.stdout)["result"]["content"][0]["text"])

    def test_each_slot_reaches_its_own_services(self):
        self.base = 24000 + random.randrange(0, 1000, 10)
        a, b = self.task_dir(), self.task_dir()
        held_out = self.start(0, a, IVF_EXCLUDE_CASE="0062", IVF_MODEL_NAME="lr_35_indicators_fold0.json")
        full = self.start(1, b)
        held_out, full = held_out[0]["tool_info"], full[0]["tool_info"]
        self.assertEqual(held_out["ivf_similar_patients"]["cases"], 905)
        self.assertEqual(full["ivf_similar_patients"]["cases"], 1132)
        self.assertEqual(held_out["ivf_predict_outcome"]["model"], "lr_35_indicators_fold0.json")
        self.assertEqual(held_out["ivf_case_sql"]["cases"], 905)
        self.assertEqual(full["ivf_case_sql"]["cases"], 1132)
        self.assertGreater(self.similar_via_bridge(1)["符合条件例数"], self.similar_via_bridge(0)["符合条件例数"])

    def test_an_earlier_ivf_service_on_the_slot_port_is_replaced(self):
        # An aborted run left its service (and lost its pid file with /tmp): the next case in the
        # same slot must get a service with its own settings, not the leftover.
        self.base = 25000 + random.randrange(0, 1000, 10)
        a, b = self.task_dir(), self.task_dir()
        old = self.start(0, a)
        new = self.start(0, b, IVF_EXCLUDE_CASE="0062", IVF_MODEL_NAME="lr_35_indicators_fold0.json")
        self.assertEqual(old[0]["tool_info"]["ivf_similar_patients"]["cases"], 1132)
        self.assertEqual(new[0]["tool_info"]["ivf_similar_patients"]["cases"], 905)
        self.assertNotEqual(old[0]["instance"], new[0]["instance"])
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                f"http://127.0.0.1:{self.base}/health", timeout=5) as resp:
            self.assertEqual(json.loads(resp.read())["instance"], new[0]["instance"])   # the old one is gone

    def test_a_port_held_by_another_program_is_refused(self):
        self.base = 26000 + random.randrange(0, 1000, 10)
        other = subprocess.Popen([sys.executable, "-m", "http.server", str(self.base), "--bind", "127.0.0.1"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(other.wait)
        self.addCleanup(other.kill)
        import time
        time.sleep(1)
        a = self.task_dir()
        env = {**os.environ, "PYTHON": sys.executable, "OMNI_TASK_DIR": a, "DRESSAGE_BLACKBOX_SLOT_ID": "0",
               "IVF_PORT_BASE": str(self.base)}
        env.pop("IVF_TOOLS_PORT", None)
        out = subprocess.run([BASH, str(SCRIPTS / "start_services.sh")], env=env, capture_output=True, text=True)
        self.addCleanup(subprocess.run, [BASH, str(SCRIPTS / "stop_services.sh")], env=env, capture_output=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("held by another service", out.stderr)
        self.assertIsNone(other.poll())                       # not ours: left running
        self.assertFalse((Path(a) / "service-ivf_tools.pid").exists())   # the one we started is stopped


if __name__ == "__main__":
    unittest.main()
