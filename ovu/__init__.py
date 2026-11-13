"""Lets ``python -m ovu.pipeline`` (and ``import ovu``) work from the repository root:
the package itself is task/services/ovu, where the tool service finds it inside a sandbox."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1] / "task" / "services" / "ovu"))
