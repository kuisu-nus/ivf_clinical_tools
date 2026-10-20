"""Train the pregnancy prediction model and save it as a JSON artifact.

  python -m clinical_tools.train_predictor --csv train_clean.csv --out models/lr_35_indicators.json

Reads the training CSV (1132 cases; never the sealed test set), applies the
same outcome-blind cleaning as the tools, fits the logistic regression on all
35 indicators (mean imputation, standardised, L2 C=1) and writes the artifact.
The prediction service loads only this file. An existing output file is
refused so trained models are never overwritten. --exclude-case hides that training case's fold (agentic RL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from clinical_tools.cleaning import load_training
from clinical_tools.predict import train_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="training CSV (golden, v2 or cleaned)")
    parser.add_argument("--out", type=Path, required=True, help="new model artifact path (.json)")
    parser.add_argument("--exclude-case", help="training case under evaluation; its fold is left out")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; write the new model to a new file")
    training, scope = load_training(args.csv).visible(args.exclude_case)
    artifact = train_model(training, scope=scope)
    artifact["training"]["data_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    keys = ("n", "cv_auc_all_indicators", "cv_brier_all_indicators", "cv_auc_three_inputs",
            "cv_brier_three_inputs", "base_rate_brier")
    print(json.dumps({"model": str(args.out), **{k: artifact["training"][k] for k in keys}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
