"""Image format utilities for BitTrainer."""

from __future__ import annotations

from pathlib import Path

# PIL plugin registration must happen in every process that decodes images,
# including DataLoader workers spawned on Windows (no fork inheritance).
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import pillow_avif_plugin  # noqa: F401
except ImportError:
    pass

SUPPORTED_EXTENSIONS: set[str] = {
    ".webp", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif", ".avif",
}


def is_supported_image(filename: str | Path) -> bool:
    """Return True if *filename* has a supported image extension (case-insensitive).

    Files containing ``-masklabel`` in their stem are always excluded.
    """
    p = Path(filename)
    if "-masklabel" in p.stem.lower():
        return False
    return p.suffix.lower() in SUPPORTED_EXTENSIONS
