"""Tests for loading, encoding, box validation and cropping."""
import base64
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

import config
import imaging
import schemas as s


def box(x, y, w, h):
    return s.BBox(x=x, y=y, w=w, h=h)


# ---------------------------------------------------------------- bbox_ok
def test_bbox_ok_accepts_normalized_boxes():
    assert imaging.bbox_ok(box(0.1, 0.2, 0.3, 0.4))
    assert imaging.bbox_ok(box(0, 0, 1, 1))
    assert imaging.bbox_ok(box(0.5, 0.5, 0, 0))  # a point: the crop floor gives it context


def test_bbox_ok_rejects_pixels_percentages_and_boxes_off_the_frame():
    assert not imaging.bbox_ok(box(120, 80, 60, 40))      # pixels
    assert not imaging.bbox_ok(box(10, 10, 20, 20))       # percentages
    assert not imaging.bbox_ok(box(0.9, 0.1, 0.5, 0.1))   # runs off the right edge
    assert not imaging.bbox_ok(box(-0.5, 0.1, 0.2, 0.1))
    assert not imaging.bbox_ok(box(float("nan"), 0.1, 0.2, 0.1))


# ---------------------------------------------------------------- crop
def test_crop_is_never_smaller_than_the_floor(monkeypatch):
    monkeypatch.setattr(config, "MIN_CROP_EDGE", 1)  # keep the upscale out of this test
    out = imaging.crop(Image.new("RGB", (2000, 1000), "white"), box(0.5, 0.5, 0.01, 0.01))
    assert out.size == (round(0.15 * 2000), round(0.15 * 1000))


def test_crop_at_the_edge_shifts_inward_instead_of_shrinking(monkeypatch):
    monkeypatch.setattr(config, "MIN_CROP_EDGE", 1)
    out = imaging.crop(Image.new("RGB", (2000, 1000), "white"), box(0.0, 0.0, 0.02, 0.02))
    assert out.size == (round(0.15 * 2000), round(0.15 * 1000))


def test_crop_pads_a_large_box(monkeypatch):
    monkeypatch.setattr(config, "MIN_CROP_EDGE", 1)
    out = imaging.crop(Image.new("RGB", (2000, 1000), "white"), box(0.3, 0.3, 0.4, 0.4))
    assert abs(out.width - 0.4 * (1 + 2 * config.CROP_PADDING) * 2000) <= 2


def test_crop_of_the_whole_frame_is_the_whole_frame():
    out = imaging.crop(Image.new("RGB", (800, 600), "white"), box(0, 0, 1, 1))
    assert out.size == (800, 600)


def test_small_crops_are_upscaled_so_details_resolve():
    out = imaging.crop(Image.new("RGB", (1000, 1000), "white"), box(0.5, 0.5, 0.01, 0.01))
    assert min(out.size) >= config.MIN_CROP_EDGE


# ---------------------------------------------------------------- encode
def _noise(size, seed=1):
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size * size)])
    return img


def _jpeg_size(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return len(buf.getvalue())


def _decoded_size(part):
    return len(base64.b64decode(part["data"]))


def test_encode_keeps_top_quality_when_the_image_fits():
    img = Image.new("RGB", (64, 64), "white")
    assert _decoded_size(imaging.encode(img)) == _jpeg_size(img, config.JPEG_QUALITIES[0])


def test_encode_steps_quality_down_until_it_fits(monkeypatch):
    img = _noise(300)
    top, bottom = _jpeg_size(img, config.JPEG_QUALITIES[0]), _jpeg_size(img, config.JPEG_QUALITIES[-1])
    assert top > bottom
    cap = (top + bottom) // 2
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", cap)
    part = imaging.encode(img)
    assert _decoded_size(part) <= cap
    assert part["media_type"] == "image/jpeg"


# ---------------------------------------------------------------- load
def test_load_applies_the_exif_rotation(tmp_path):
    img = Image.new("RGB", (40, 20), "red")
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90 clockwise to display"
    path = tmp_path / "photo.jpg"
    img.save(path, "JPEG", exif=exif)
    assert imaging.load(path).size == (20, 40)


def test_load_downscales_the_long_edge(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MAX_IMAGE_EDGE", 100)
    path = tmp_path / "big.png"
    Image.new("RGB", (300, 200), "blue").save(path)
    assert imaging.load(path).size == (100, 67)


def test_load_flattens_to_rgb(tmp_path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(path)
    assert imaging.load(path).mode == "RGB"
