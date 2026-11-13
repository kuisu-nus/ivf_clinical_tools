"""Write the MCP tool specs of the oocyte-yield tools from the service's request models.

  OVU_DATA_DIR=outputs/ovu PYTHONPATH=task/services python -m ovu_services.specs --tools task/ovu_tools

The schemas follow the model file and the retrieval spec, so run this again after a rebuild.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ivf_services.specs import input_schema
from ovu_services import server


def tool_specs(base_url: str | None = None) -> list[dict[str, Any]]:
    base = base_url or f"http://127.0.0.1:{server.DEFAULT_PORT}"
    return [{"name": t.TOOL, "description": t.TOOL_DESCRIPTION, "input_schema": input_schema(t.REQUEST_MODEL),
             "backend": {"type": "http", "service": server.SERVICE, "method": "POST", "url": base + server.tool_path(t)}}
            for t in server.TOOLS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tools", type=Path, required=True, help="directory for the tool spec JSON files")
    args = parser.parse_args(argv)
    args.tools.mkdir(parents=True, exist_ok=True)
    for spec in tool_specs():
        path = args.tools / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
