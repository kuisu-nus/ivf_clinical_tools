"""Pregnancy prediction service (default port 9202); loads a trained model artifact only.

  POST /ivf/predict_outcome    {"age": 38, "amh": 0.9, "gn_days": 10}
  ->                           {"妊娠概率": 0.2, "95%置信区间": [0.15, 0.26]}

  python -m clinical_tools.predict_service --model models/lr_age_amh_gn.json [--port 9202]
"""

from __future__ import annotations

import argparse
import os

from clinical_tools.http_service import make_server, serve
from clinical_tools.predict import Predictor

DEFAULT_PORT = 9202


def build(predictor: Predictor, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
    health = {"tool": "ivf_predict_outcome", "model_training_n": predictor.training["n"],
              "model_cv_auc": predictor.training["cv_auc"]}
    return make_server({"/ivf/predict_outcome": predictor.query}, health, host, port), health


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("IVF_MODEL"), help="model artifact from train_predictor")
    parser.add_argument("--host", default=os.environ.get("IVF_TOOLS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("IVF_PREDICT_PORT", DEFAULT_PORT)))
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or IVF_MODEL is required")
    server, health = build(Predictor.from_file(args.model), args.host, args.port)
    serve(server, health)


if __name__ == "__main__":
    main()
