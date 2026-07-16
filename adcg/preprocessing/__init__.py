def run_preprocess(*args, **kwargs):
    """Load optional preprocessing dependencies only when preprocessing starts."""
    from .product import run_preprocess as _run_preprocess

    return _run_preprocess(*args, **kwargs)

__all__ = ["run_preprocess"]
