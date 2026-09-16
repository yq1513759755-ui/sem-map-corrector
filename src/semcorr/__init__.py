"""SE2 SEM mark localization and distortion correction."""

__version__ = "0.4.0"
__all__ = ["correct_image", "make_demo_image"]


def correct_image(*args, **kwargs):
    """Load the image stack lazily so metadata commands stay lightweight."""
    from .pipeline import correct_image as _correct_image

    return _correct_image(*args, **kwargs)


def make_demo_image(*args, **kwargs):
    """Synthetic SE2 demo image generator (see semcorr.demo)."""
    from .demo import make_demo_image as _make_demo_image

    return _make_demo_image(*args, **kwargs)
