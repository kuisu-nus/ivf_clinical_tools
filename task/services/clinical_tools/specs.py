"""Write the MCP tool spec files (one JSON per tool) for the Claw task format.

  python -m clinical_tools.specs --out task/tools [--similar-url URL] [--predict-url URL]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_tools import predict, similar

SIMILAR_URL = "http://127.0.0.1:9201"
PREDICT_URL = "http://127.0.0.1:9202"


def tool_specs(similar_url: str = SIMILAR_URL, predict_url: str = PREDICT_URL) -> list[dict]:
    return [similar.tool_spec(similar_url), predict.tool_spec(predict_url)]


def write_specs(directory: Path, similar_url: str = SIMILAR_URL, predict_url: str = PREDICT_URL) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in tool_specs(similar_url, predict_url):
        path = directory / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--similar-url", default=SIMILAR_URL)
    parser.add_argument("--predict-url", default=PREDICT_URL)
    args = parser.parse_args(argv)
    for path in write_specs(args.out, args.similar_url, args.predict_url):
        print(path)


if __name__ == "__main__":
    main()
