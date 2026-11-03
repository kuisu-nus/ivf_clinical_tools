"""Point an OpenClaw config (openclaw.json) at a model served locally (SGLang, OpenAI-compatible).

  python ckpt_eval/oc_config.py <openclaw.json> --base-url http://127.0.0.1:30001/v1 --model ivf-step100 \
      [--provider ivf] [--workspace <dir>]

Adds (or replaces) provider <provider> with that one model and makes it the default model; every other
provider is kept, so the change is easy to undo. The provider gets a placeholder API key: OpenClaw keeps
real keys in its own auth store, which a new provider has no entry in, and a local SGLang checks none.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

PLACEHOLDER_KEY = "sk-local-sglang"
DEFAULT_MODEL = {"reasoning": False, "input": ["text"],
                 "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                 "contextWindow": 128000, "maxTokens": 8192}


def point_to_model(config: dict[str, Any], base_url: str, model: str, provider: str = "ivf",
                   workspace: str | None = None) -> dict[str, Any]:
    config = copy.deepcopy(config)
    providers = config.setdefault("models", {}).setdefault("providers", {})
    # Model settings (context window, token limit, ...) follow an existing provider when there is one.
    template = next((p for k, p in providers.items() if k != provider and p.get("models")), None)
    entry = copy.deepcopy(template["models"][0]) if template else copy.deepcopy(DEFAULT_MODEL)
    entry["id"] = entry["name"] = model
    providers[provider] = {"baseUrl": base_url, "api": (template or {}).get("api", "openai-completions"),
                           "apiKey": PLACEHOLDER_KEY, "models": [entry]}
    defaults = config.setdefault("agents", {}).setdefault("defaults", {})
    defaults["model"] = {"primary": f"{provider}/{model}"}
    if workspace:
        defaults["workspace"] = workspace
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="ivf")
    parser.add_argument("--workspace")
    args = parser.parse_args(argv)
    config = point_to_model(json.loads(args.config.read_text(encoding="utf-8")), args.base_url, args.model,
                            args.provider, args.workspace)
    args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, p in config["models"]["providers"].items():
        print(f"  provider {name}: {p.get('baseUrl')} {[m.get('id') for m in p.get('models', [])]}")
    print(f"  default model: {config['agents']['defaults']['model']['primary']}")


if __name__ == "__main__":
    main()
