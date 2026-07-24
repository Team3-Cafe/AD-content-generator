from __future__ import annotations

import shutil
from pathlib import Path


def remove_pipeline_stage(output_dir: str | Path, stage_name: str) -> None:
    """Remove one generated stage directory without escaping the output root."""
    root = Path(output_dir).expanduser().resolve()
    target = (root / stage_name).resolve()
    if target.parent != root:
        raise ValueError(f"Invalid pipeline stage path: {target}")
    if target.is_dir():
        shutil.rmtree(target)


def keep_only(directory: str | Path, keep: tuple[str | Path, ...]) -> None:
    """Delete generated artifacts in a directory except explicitly kept paths."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return

    kept = {Path(path).expanduser().resolve() for path in keep}
    for child in root.iterdir():
        resolved = child.resolve()
        if resolved in kept:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


__all__ = ["keep_only", "remove_pipeline_stage"]
