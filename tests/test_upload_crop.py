import io

import numpy as np
import pytest
import tifffile
from PIL import Image

from demo_core.upload_crop import ImageValidationError, prepare_upload


def png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG", dpi=(96, 96))
    return buffer.getvalue()


def tiff_bytes(array: np.ndarray, **kwargs) -> bytes:
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, **kwargs)
    return buffer.getvalue()


def test_png_preserves_nonmultiple_dimensions_without_resampling():
    array = np.arange(517 * 300 * 3, dtype=np.uint32).reshape(517, 300, 3).astype(np.uint8)
    content = png_bytes(array)
    result = prepare_upload(content, "cells.png")
    assert result.status == "unchanged" and result.data == content
    assert (result.output_width, result.output_height) == (300, 517)
    assert (result.removed_right, result.removed_bottom) == (0, 0)
    assert result.filename == "cells.png" and result.verified
    with Image.open(io.BytesIO(result.data)) as output:
        assert output.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(output), array)


def test_compliant_png_uses_original_verified_bytes():
    content = png_bytes(np.zeros((256, 512), dtype=np.uint8))
    result = prepare_upload(content, "ready.png")
    assert result.status == "unchanged" and result.data == content
    assert (result.output_width, result.output_height) == (512, 256)


def test_uint16_png_preserves_bit_depth_and_pixels():
    array = np.arange(300 * 300, dtype=np.uint16).reshape(300, 300)
    result = prepare_upload(png_bytes(array), "depth16.png")
    with Image.open(io.BytesIO(result.data)) as output:
        assert output.mode == "I;16"
        np.testing.assert_array_equal(np.asarray(output), array)


def test_multichannel_tiff_preserves_dtype_channels_resolution_and_pixels():
    array = np.arange(300 * 300 * 3, dtype=np.uint16).reshape(300, 300, 3)
    content = tiff_bytes(
        array, photometric="rgb", resolution=(300, 300), resolutionunit="INCH", metadata={"axes": "YXS"},
    )
    result = prepare_upload(content, "rgb16.tif")
    assert result.channels == 3 and result.dtype == np.dtype(np.uint16).str
    with tifffile.TiffFile(io.BytesIO(result.data)) as tif:
        output = tif.pages[0].asarray()
        assert output.shape == (300, 300, 3) and output.dtype == np.uint16
        assert tif.pages[0].tags["XResolution"].value == (300, 1)
        np.testing.assert_array_equal(output, array)


def test_cyx_tiff_is_not_reinterpreted_or_dropped():
    array = np.arange(2 * 300 * 300, dtype=np.uint16).reshape(2, 300, 300)
    content = tiff_bytes(
        array, photometric="minisblack", planarconfig="separate", metadata={"axes": "CYX"},
    )
    result = prepare_upload(content, "channels.tiff")
    with tifffile.TiffFile(io.BytesIO(result.data)) as tif:
        output = tif.pages[0].asarray()
        assert tif.series[0].axes == "CYX"
        np.testing.assert_array_equal(output, array)


def test_multipage_tiff_is_rejected_without_silent_page_loss():
    buffer = io.BytesIO()
    with tifffile.TiffWriter(buffer) as writer:
        writer.write(np.zeros((300, 300), dtype=np.uint8))
        writer.write(np.ones((300, 300), dtype=np.uint8))
    with pytest.raises(ImageValidationError, match="2 页"):
        prepare_upload(buffer.getvalue(), "stack.tif")


def test_small_images_are_retained_and_fake_files_are_rejected():
    content = png_bytes(np.zeros((255, 256), dtype=np.uint8))
    result = prepare_upload(content, "small.png")
    assert result.status == "unchanged" and result.data == content
    assert (result.output_width, result.output_height) == (256, 255)
    with pytest.raises(ImageValidationError, match="真实格式不一致"):
        prepare_upload(png_bytes(np.zeros((256, 256), dtype=np.uint8)), "fake.tif")
    with pytest.raises(ImageValidationError, match="不能仅通过修改后缀"):
        prepare_upload(b"not a real image", "fake.png")


def test_decoded_pixel_limit_is_enforced(monkeypatch):
    monkeypatch.setattr("demo_core.upload_crop.MAX_PIXELS", 256 * 256)
    with pytest.raises(ImageValidationError, match="安全限制"):
        prepare_upload(png_bytes(np.zeros((256, 512), dtype=np.uint8)), "large.png")
