import io
import numpy as np
from PIL import Image

# pyright: reportMissingImports=false
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print('HEIC/HEIF support registered (pillow-heif found).')
except ImportError:
    print('pillow-heif NOT installed - HEIC/HEIF uploads will fail. '
          'Run: pip install pillow-heif')


def load_as_rgb(path: str):
    """Load any image (including HEIC) from disk as an RGB numpy array."""
    try:
        return np.array(Image.open(path).convert('RGB'))
    except Exception as e:
        print(f'Failed to load {path}: {e}')
        return None


def load_bytes_as_rgb(image_bytes: bytes):
    """Load an image from raw bytes (e.g. an UploadFile's contents) as RGB numpy array."""
    if not image_bytes:
        print('load_bytes_as_rgb received empty bytes (0-byte upload).')
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()   

        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f'Failed to load image ({len(image_bytes)} bytes, '
              f'header={image_bytes[:12]!r}): {e}')
        return None


def load_and_resize(path: str):
    """Kept for backward compatibility with existing router.py imports."""
    return load_as_rgb(path)
