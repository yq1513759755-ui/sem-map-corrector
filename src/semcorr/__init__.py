"""SE2 SEM mark localization and distortion correction."""

__version__ = "0.1.0"
__all__ = ["correct_image"]


def correct_image(*args, **kwargs):
    """Load the image stack lazily so metadata commands stay lightweight."""
    from .pipeline import correct_image as _correct_image

    return _correct_image(*args, **kwargs)
