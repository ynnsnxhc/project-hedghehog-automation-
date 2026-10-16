"""Image loading, resizing, encoding and cropping."""
import base64
import io

from PIL import Image, ImageOps

import config


def load(path):
    img = Image.open(path)
    img.load()
    img = ImageOps.exif_transpose(img)  # photos can carry a rotation flag the pixels don't show
    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > config.MAX_IMAGE_EDGE:
        scale = config.MAX_IMAGE_EDGE / long_edge
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    return img


def encode(img):
    """Return an image part for providers: JPEG, base64, small enough for the API.

    Starts at high quality and only steps down if the file would be too large.
    """
    data = b""
    for quality in config.JPEG_QUALITIES:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= config.MAX_IMAGE_BYTES:
            break
    return {"type": "image", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(data).decode("ascii")}


def bbox_ok(box):
    """False when a box can't be in normalized 0-1 units (pixels or percentages by mistake).

    A bad box would crop the wrong place and the Verifier would "reject" a true finding,
    so callers fall back to the full image instead.
    """
    values = (box.x, box.y, box.w, box.h)
    if any(v != v for v in values):  # NaN
        return False
    if min(values) < -0.01 or max(values) > 1.01:
        return False
    return box.x + box.w <= 1.05 and box.y + box.h <= 1.05


def crop(img, box):
    """Crop a normalized box with padding, clamped to the image, upscaled if tiny.

    Boxes from the Inspector are approximate, so the crop is centred on the box, padded,
    and never narrower or shorter than config.MIN_CROP_FRACTION of the frame.
    """
    x, y = _clamp(box.x), _clamp(box.y)
    w, h = _clamp(box.w), _clamp(box.h)
    left, right = _span(x + w / 2, w * (1 + 2 * config.CROP_PADDING) / 2)
    top, bottom = _span(y + h / 2, h * (1 + 2 * config.CROP_PADDING) / 2)
    left, right = left * img.width, right * img.width
    top, bottom = top * img.height, bottom * img.height
    if right - left < 2 or bottom - top < 2:
        return img.copy()
    out = img.crop((round(left), round(top), round(right), round(bottom)))
    short = min(out.size)
    if short < config.MIN_CROP_EDGE:
        scale = config.MIN_CROP_EDGE / short
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.Resampling.LANCZOS)
    return out


def _span(center, half):
    """Interval around `center` at least MIN_CROP_FRACTION wide, shifted to stay inside 0-1."""
    half = max(half, config.MIN_CROP_FRACTION / 2)
    lo, hi = center - half, center + half
    if lo < 0:
        lo, hi = 0.0, hi - lo
    if hi > 1:
        lo, hi = lo - (hi - 1), 1.0
    return _clamp(lo), _clamp(hi)


def _clamp(v):
    return min(max(float(v), 0.0), 1.0)
