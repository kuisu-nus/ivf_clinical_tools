"""Command line entry point: ``python3 -m e2b_eval``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .config import (
    AgentLLMConfig,
    ConfigError,
    DEFAULT_TEMPLATE,
    E2BConfig,
)
from .runner import RunOptions, run_all
from .samples import SampleError
from .util import load_env_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m e2b_eval",
        description=(
            "Run each Dressage-claw sample in its own E2B sandbox: "
            "before_agent -> agent -> after_agent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="samples.jsonl produced by dataset/build_samples.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/e2b_eval"),
        help="output directory (per-case artifacts and summary.json)",
    )
    parser.add_argument("--limit", type=int, help="run only the first N selected cases")
    parser.add_argument(
        "--case",
        dest="only",
        action="append",
        default=[],
        metavar="ID",
        help="run only this instance_id / case suffix (repeatable; e.g. --case 0974)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of sandboxes to run concurrently",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="rerun cases even when score.json already exists",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="do not kill the sandbox at the end (debugging; it stays until its timeout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate samples and --path-map mappings without creating a sandbox",
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=REPO_ROOT,
        help="repo checkout used to remap absolute local_path values from another machine",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="explicit local_path prefix remap (repeatable)",
    )
    parser.add_argument(
        "--template",
        help=(
            "override the sample's sandbox_image "
            f"(default: sample metadata, then $E2B_TEMPLATE or {DEFAULT_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        help=(
            "override the sandbox lifetime in seconds "
            "(default: sample metadata, then $E2B_SANDBOX_TIMEOUT_SEC)"
        ),
    )

    agent = parser.add_argument_group("agent LLM (used by OpenClaw inside the sandbox)")
    agent.add_argument(
        "--agent-mode",
        choices=("openclaw", "mock"),
        default=None,
        help=(
            "openclaw = real model through BlackboxServer (default, or $AGENT_MODE); "
            "mock = deterministic tool calls, no LLM"
        ),
    )
    agent.add_argument("--agent-base-url", help="OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1")
    agent.add_argument("--agent-model", help="model id, e.g. deepseek-flash")
    agent.add_argument("--agent-api-key", help="model API key (prefer AGENT_LLM_API_KEY)")
    agent.add_argument("--agent-provider-id", help="OpenClaw provider id, e.g. deepseek")
    agent.add_argument("--agent-context-window", type=int, help="OpenClaw context window")
    agent.add_argument("--agent-max-tokens", type=int, help="max output tokens per model call")
    agent.add_argument(
        "--agent-max-steps",
        type=int,
        help="max agent tool-call steps (0 disables the limit)",
    )
    agent.add_argument("--agent-register-timeout", type=float, help="seconds to wait for OpenClaw startup")
    agent.add_argument("--agent-turn-timeout", type=float, help="seconds for the agent turn")

    parser.add_argument(
        "--env-file",
        type=Path,
        help="load KEY=VALUE exports before reading the environment (does not override existing vars)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"e2b_eval {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.env_file:
            load_env_file(args.env_file)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("e2b").setLevel(logging.WARNING)

        if args.workers < 1:
            raise ConfigError("--workers must be >= 1")
        if args.sandbox_timeout is not None and args.sandbox_timeout <= 0:
            raise ConfigError("--sandbox-timeout must be positive")
        path_maps = tuple(_parse_path_map(item) for item in args.path_map)
        samples_path = args.samples.expanduser().resolve()
        out_dir = args.out.expanduser().resolve()
        task_root = args.task_root.expanduser().resolve() if args.task_root else None

        if args.dry_run:
            template = args.template or os.environ.get("E2B_TEMPLATE") or DEFAULT_TEMPLATE
            timeout = args.sandbox_timeout or 0
            e2b = E2BConfig(
                api_key="dry-run",
                api_url=None,
                sandbox_url=None,
                template=template,
                timeout_sec=timeout or 3600,
            )
        else:
            e2b = E2BConfig.from_env(
                template=args.template,
                timeout_sec=args.sandbox_timeout,
            )

        agent_mode = args.agent_mode or os.environ.get("AGENT_MODE", "openclaw")
        agent = AgentLLMConfig.from_env(
            mode=agent_mode,
            base_url=args.agent_base_url,
            model=args.agent_model,
            api_key=args.agent_api_key,
            provider_id=args.agent_provider_id,
            context_window=args.agent_context_window,
            max_tokens=args.agent_max_tokens,
            max_steps=args.agent_max_steps,
            register_timeout_sec=args.agent_register_timeout,
            turn_timeout_sec=args.agent_turn_timeout,
            require_key=not args.dry_run,
        )

        options = RunOptions(
            samples_path=samples_path,
            out_dir=out_dir,
            limit=args.limit,
            only=tuple(args.only),
            workers=args.workers,
            resume=args.resume,
            keep_sandbox=args.keep_sandbox,
            dry_run=args.dry_run,
            task_root=task_root,
            path_maps=path_maps,
            template_override=args.template,
            sandbox_timeout_override=args.sandbox_timeout,
            e2b=e2b,
            agent=agent,
        )
        if not args.dry_run:
            print(
                f"samples={samples_path}\n"
                f"out={out_dir}\n"
                f"agent={agent.mode} model={agent.model} base_url={agent.base_url}\n"
                f"template={e2b.template} sandbox_timeout={args.sandbox_timeout or '(sample metadata)'} "
                f"workers={args.workers}",
                flush=True,
            )
        return run_all(options)
    except (ConfigError, SampleError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _parse_path_map(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ConfigError(f"--path-map must look like OLD=NEW, got {value!r}")
    old, new = value.split("=", 1)
    old = old.strip()
    new = new.strip()
    if not old or not new:
        raise ConfigError(f"--path-map must look like OLD=NEW, got {value!r}")
    return old, new
