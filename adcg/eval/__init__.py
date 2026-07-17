"""Evaluation helpers for generated advertisement images."""


def run_evaluation(*args, **kwargs):
    from .runner import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


__all__ = ["run_evaluation"]