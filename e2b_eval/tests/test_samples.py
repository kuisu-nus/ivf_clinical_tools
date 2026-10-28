"""Sample validation and local_path remapping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from e2b_eval.samples import (
    SampleError,
    load_samples,
    resolve_local_path,
    resolve_uploads,
    validate_upload_plan,
)


def make_sample(root: Path, *, instance_id: str = "claw-test-0001", **overrides) -> dict:
    system_prompt = root / "system_prompt_prefix.txt"
    setup = root / "setup.sh"
    metadata_file = root / "metadata.json"
    for path, text in (
        (system_prompt, "你是助手。\n"),
        (setup, "#!/bin/sh\necho ok\n"),
        (metadata_file, json.dumps({"label": "妊娠", "inputs": {"age": 30, "amh": 2.0, "gn_days": 10}})),
    ):
        path.write_text(text, encoding="utf-8")
    sample = {
        "prompt": "病例文本",
        "label": "",
        "reward_fn": "claw_grader",
        "metadata": {
            "instance_id": instance_id,
            "task_id": "task-1",
            "sandbox_image": "e2b-dressage-claw-blackbox",
            "sandbox_timeout_sec": 1200,
            "system_prompt_file": "/tmp/omni_task/runtime/system_prompt_prefix.txt",
            "before_agent_files": [
                {"path": "/tmp/omni_task/runtime/system_prompt_prefix.txt", "local_path": str(system_prompt)},
                {"path": "/tmp/omni_task/scripts/setup.sh", "local_path": str(setup)},
            ],
            "after_agent_files": [
                {"path": "/tmp/omni_grader/metadata.json", "local_path": str(metadata_file)},
            ],
            "blackbox_execute_cmds": {
                "before_agent": [
                    {"name": "setup", "cmd": "bash /tmp/omni_task/scripts/setup.sh", "required": True, "timeout": 10}
                ],
                "after_agent": [
                    {"name": "grader", "cmd": "bash /tmp/omni_grader/run.sh", "required": True, "timeout": 10}
                ],
            },
        },
    }
    sample.update(overrides)
    return sample


def write_jsonl(path: Path, samples: list[dict]) -> Path:
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    return path


class LoadSamplesTests(unittest.TestCase):
    def test_loads_and_matches_case_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_jsonl(root / "samples.jsonl", [make_sample(root)])
            samples = load_samples(path)
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.instance_id, "claw-test-0001")
            self.assertEqual(sample.sandbox_timeout_sec, 1200)
            self.assertEqual(len(sample.files("before_agent")), 2)
            self.assertEqual(sample.commands("before_agent")[0]["name"], "setup")

            selected = load_samples(path, only=["0001"])
            self.assertEqual(len(selected), 1)
            with self.assertRaises(SampleError):
                load_samples(path, only=["nope"])

    def test_rejects_newline_in_execute_cmd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = make_sample(root)
            sample["metadata"]["blackbox_execute_cmds"]["before_agent"][0]["cmd"] = "echo a\necho b"
            path = write_jsonl(root / "samples.jsonl", [sample])
            with self.assertRaisesRegex(SampleError, "single line"):
                load_samples(path)

    def test_rejects_duplicate_instance_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_jsonl(root / "samples.jsonl", [make_sample(root), make_sample(root)])
            with self.assertRaisesRegex(SampleError, "duplicate case key"):
                load_samples(path)

    def test_invalid_json_reports_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaisesRegex(SampleError, "invalid JSON"):
                load_samples(path)

    def test_rejects_unknown_command_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = make_sample(root)
            sample["metadata"]["blackbox_execute_cmds"]["during_agent"] = []
            path = write_jsonl(root / "samples.jsonl", [sample])
            with self.assertRaisesRegex(SampleError, "unsupported"):
                load_samples(path)

    def test_rejects_non_openclaw_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = make_sample(root)
            sample["metadata"]["blackbox_type"] = "opencode"
            path = write_jsonl(root / "samples.jsonl", [sample])
            with self.assertRaisesRegex(SampleError, "blackbox_type=openclaw"):
                load_samples(path)


class UploadPlanTests(unittest.TestCase):
    def test_before_upload_has_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_jsonl(root / "samples.jsonl", [make_sample(root)])
            sample = load_samples(path)[0]
            before = resolve_uploads(sample, "before_agent", samples_path=path, task_root=root)
            after = resolve_uploads(sample, "after_agent", samples_path=path, task_root=root)
            self.assertEqual([u.sandbox_path for u in before][0], "/tmp/omni_task/runtime/system_prompt_prefix.txt")
            validate_upload_plan(sample, before, after)

    def test_rejects_grader_file_before_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = make_sample(root)
            sample["metadata"]["before_agent_files"].append(
                {"path": "/tmp/omni_grader/grader.py", "local_path": str(root / "setup.sh")}
            )
            path = write_jsonl(root / "samples.jsonl", [sample])
            loaded = load_samples(path)[0]
            before = resolve_uploads(loaded, "before_agent", samples_path=path, task_root=root)
            after = resolve_uploads(loaded, "after_agent", samples_path=path, task_root=root)
            with self.assertRaisesRegex(SampleError, "grader paths"):
                validate_upload_plan(loaded, before, after)

    def test_rejects_missing_system_prompt_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = make_sample(root)
            sample["metadata"]["before_agent_files"] = sample["metadata"]["before_agent_files"][1:]
            sample["metadata"]["system_prompt_file"] = "/tmp/omni_task/runtime/missing.txt"
            path = write_jsonl(root / "samples.jsonl", [sample])
            loaded = load_samples(path)[0]
            before = resolve_uploads(loaded, "before_agent", samples_path=path, task_root=root)
            after = resolve_uploads(loaded, "after_agent", samples_path=path, task_root=root)
            with self.assertRaisesRegex(SampleError, "system_prompt_file"):
                validate_upload_plan(loaded, before, after)


class ResolveLocalPathTests(unittest.TestCase):
    def test_direct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("x", encoding="utf-8")
            self.assertEqual(
                resolve_local_path(str(target), samples_path=root / "samples.jsonl"),
                target.resolve(),
            )

    def test_explicit_prefix_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "task" / "setup.sh"
            target.parent.mkdir()
            target.write_text("x", encoding="utf-8")
            resolved = resolve_local_path(
                "/old/root/task/setup.sh",
                samples_path=root / "samples.jsonl",
                path_maps=(("/old/root", str(root)),),
            )
            self.assertEqual(resolved, target.resolve())

    def test_task_root_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "task" / "scripts" / "setup.sh"
            target.parent.mkdir(parents=True)
            target.write_text("x", encoding="utf-8")
            resolved = resolve_local_path(
                "/home/someone/repo/task/scripts/setup.sh",
                samples_path=root / "samples.jsonl",
                task_root=root,
            )
            self.assertEqual(resolved, target.resolve())

    def test_cases_directory_next_to_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "run" / "samples.jsonl"
            metadata = root / "run" / "cases" / "abc" / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("{}", encoding="utf-8")
            resolved = resolve_local_path(
                "/home/someone/repo/runs/dev/cases/abc/metadata.json",
                samples_path=samples,
            )
            self.assertEqual(resolved, metadata.resolve())

    def test_missing_file_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SampleError, "--path-map"):
                resolve_local_path(
                    "/not/here/task/a.txt",
                    samples_path=Path(tmp) / "samples.jsonl",
                    task_root=Path(tmp),
                )


if __name__ == "__main__":
    unittest.main()
