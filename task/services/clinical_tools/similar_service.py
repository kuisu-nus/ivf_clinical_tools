"""Similar-patient retrieval service (default port 9201).

  POST /ivf/similar_patients   {"age": 38, "amh": 0.9, "age_window": 1}

A cohort below the minimum size is a normal 200 answer with 拒绝统计 = true.

  python -m clinical_tools.similar_service --csv train_clean.csv [--port 9201] [--exclude-case ID]
"""

from __future__ import annotations

import argparse
import os

from clinical_tools.cleaning import load_training
from clinical_tools.http_service import make_server, serve
from clinical_tools.similar import SimilarCohort

DEFAULT_PORT = 9201


def build(cohort: SimilarCohort, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
    health = {"tool": "ivf_similar_patients", "scope": cohort.scope, "searchable_cases": int(cohort.searchable.sum())}
    return make_server({"/ivf/similar_patients": cohort.query}, health, host, port), health


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=os.environ.get("IVF_TRAIN_CSV"), help="training CSV (never the sealed test set)")
    parser.add_argument("--host", default=os.environ.get("IVF_TOOLS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("IVF_SIMILAR_PORT", DEFAULT_PORT)))
    parser.add_argument("--exclude-case", default=os.environ.get("IVF_EXCLUDE_CASE") or None,
                        help="training case under evaluation; its fold is hidden from the tool")
    args = parser.parse_args(argv)
    if not args.csv:
        parser.error("--csv or IVF_TRAIN_CSV is required")
    server, health = build(SimilarCohort(load_training(args.csv), exclude_case=args.exclude_case), args.host, args.port)
    serve(server, health)


if __name__ == "__main__":
    main()
