"""Deterministic PNG/TIFF upload validation and top-left cropping to multiples of 256."""
from __future__ import annotations

import io
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

TILE = 256
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_PIXELS = 16_000_000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class ImageValidationError(ValueError):
    """An upload cannot be safely used by the analysis pipeline."""


@dataclass
class CropResult:
    data: bytes
    filename: str
    media_type: str
    source_format: str
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    removed_right: int
    removed_bottom: int
    status: str
    dtype: str
    channels: int
    pages: int
    verified: bool

    @property
    def cropped(self) -> bool:
        return self.status == "cropped"

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("data")
        return result

    def metadata_header(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _safe_output_name(filename: str, suffix: str) -> str:
    clean = Path(filename or "image").name
    stem = Path(clean).stem or "image"
    return f"{stem}_crop256{suffix}"


def _detect_format(data: bytes, filename: str) -> tuple[str, str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ImageValidationError("仅支持扩展名为 .png、.tif、.tiff 的文件。")
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError("文件超过 200 MB 上传限制。")
    if data.startswith(PNG_MAGIC):
        actual = "PNG"
    elif data.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        actual = "TIFF"
    else:
        raise ImageValidationError("文件内容不是有效的 PNG 或 TIFF；不能仅通过修改后缀绕过检查。")
    expected = "PNG" if suffix == ".png" else "TIFF"
    if actual != expected:
        raise ImageValidationError(f"文件扩展名与真实格式不一致：后缀表示 {expected}，内容实际为 {actual}。")
    media_type = "image/png" if actual == "PNG" else "image/tiff"
    return actual, media_type, suffix


def _target_size(width: int, height: int) -> tuple[int, int]:
    if width < TILE or height < TILE:
        raise ImageValidationError(
            f"图片为 {width}×{height}，宽和高都必须至少为 {TILE} 像素；不会放大或生成零尺寸图片。"
        )
    if width * height > MAX_PIXELS:
        raise ImageValidationError(f"解码后共有 {width * height:,} 个像素，超过 {MAX_PIXELS:,} 像素安全限制。")
    return width // TILE * TILE, height // TILE * TILE


def _result(
    *, data: bytes, filename: str, media_type: str, source_format: str,
    width: int, height: int, out_width: int, out_height: int,
    dtype: str, channels: int, pages: int,
) -> CropResult:
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError("处理后的文件仍超过目标站点的 200 MB 限制。")
    return CropResult(
        data=data,
        filename=filename,
        media_type=media_type,
        source_format=source_format,
        original_width=width,
        original_height=height,
        output_width=out_width,
        output_height=out_height,
        removed_right=width - out_width,
        removed_bottom=height - out_height,
        status="unchanged" if (width, height) == (out_width, out_height) else "cropped",
        dtype=dtype,
        channels=channels,
        pages=pages,
        verified=True,
    )


def _process_png(data: bytes, filename: str, media_type: str, suffix: str) -> CropResult:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise ImageValidationError("暂不支持多帧 PNG/APNG，以免静默丢弃页面。")
                opened.load()
                image = opened.copy()
                info = dict(opened.info)
                mode = opened.mode
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"PNG 解码失败：{exc}") from exc

    width, height = image.size
    out_width, out_height = _target_size(width, height)
    channels = len(image.getbands())
    dtype = np.asarray(image).dtype.str
    output_name = _safe_output_name(filename, suffix)
    if (width, height) == (out_width, out_height):
        return _result(
            data=data, filename=output_name, media_type=media_type, source_format="PNG",
            width=width, height=height, out_width=out_width, out_height=out_height,
            dtype=dtype, channels=channels, pages=1,
        )

    supported_modes = {"1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I;16B", "I;16L"}
    if mode not in supported_modes:
        raise ImageValidationError(f"PNG 模式 {mode} 暂不支持无损另存；不会自动转换颜色或位深。")
    cropped = image.crop((0, 0, out_width, out_height))
    save_kwargs: dict = {}
    if "dpi" in info:
        save_kwargs["dpi"] = info["dpi"]
    if "icc_profile" in info:
        save_kwargs["icc_profile"] = info["icc_profile"]
    buffer = io.BytesIO()
    try:
        cropped.save(buffer, format="PNG", **save_kwargs)
        output = buffer.getvalue()
        with Image.open(io.BytesIO(output)) as check:
            check.load()
            if check.size != (out_width, out_height) or check.mode != mode:
                raise ImageValidationError("输出 PNG 的尺寸或颜色模式验证失败。")
            if not np.array_equal(np.asarray(check), np.asarray(cropped)):
                raise ImageValidationError("输出 PNG 未通过逐像素一致性验证。")
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"PNG 无损另存失败：{exc}") from exc
    return _result(
        data=output, filename=output_name, media_type=media_type, source_format="PNG",
        width=width, height=height, out_width=out_width, out_height=out_height,
        dtype=dtype, channels=channels, pages=1,
    )


def _resolution_tag(page, name: str) -> float | None:
    tag = page.tags.get(name)
    if tag is None:
        return None
    value = tag.value
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return value[0] / value[1]
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _process_tiff(data: bytes, filename: str, media_type: str, suffix: str) -> CropResult:
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            if len(tif.pages) != 1:
                raise ImageValidationError(
                    f"检测到 {len(tif.pages)} 页 TIFF；为避免静默丢页，暂不支持自动裁剪，请先拆分页面。"
                )
            page = tif.pages[0]
            series = tif.series[0]
            array = series.asarray()
            axes = series.axes
            if len(array.shape) not in {2, 3} or any(axis not in "YXSC" for axis in axes) \
                    or "Y" not in axes or "X" not in axes:
                raise ImageValidationError(f"暂不支持轴结构为 {axes}、形状为 {array.shape} 的特殊 TIFF。")
            height = array.shape[axes.index("Y")]
            width = array.shape[axes.index("X")]
            channel_axis = "S" if "S" in axes else "C" if "C" in axes else None
            channels = array.shape[axes.index(channel_axis)] if channel_axis else 1
            out_width, out_height = _target_size(width, height)
            dtype = array.dtype.str
            output_name = _safe_output_name(filename, suffix)
            if (width, height) == (out_width, out_height):
                return _result(
                    data=data, filename=output_name, media_type=media_type, source_format="TIFF",
                    width=width, height=height, out_width=out_width, out_height=out_height,
                    dtype=dtype, channels=channels, pages=1,
                )

            slices = [slice(None)] * array.ndim
            slices[axes.index("Y")] = slice(0, out_height)
            slices[axes.index("X")] = slice(0, out_width)
            cropped = array[tuple(slices)]
            kwargs: dict = {
                "photometric": page.photometric,
                "compression": "deflate",
                "metadata": {"axes": axes},
            }
            if page.planarconfig is not None:
                kwargs["planarconfig"] = page.planarconfig
            if page.extrasamples:
                kwargs["extrasamples"] = page.extrasamples
            if page.colormap is not None:
                kwargs["colormap"] = page.colormap
            x_resolution = _resolution_tag(page, "XResolution")
            y_resolution = _resolution_tag(page, "YResolution")
            if x_resolution and y_resolution:
                kwargs["resolution"] = (x_resolution, y_resolution)
                unit = page.tags.get("ResolutionUnit")
                if unit is not None:
                    kwargs["resolutionunit"] = unit.value
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, cropped, bigtiff=array.nbytes >= 2**32, **kwargs)
            output = buffer.getvalue()
            with tifffile.TiffFile(io.BytesIO(output)) as check_tif:
                if len(check_tif.pages) != 1:
                    raise ImageValidationError("输出 TIFF 页数验证失败。")
                check = check_tif.pages[0].asarray()
            if check.shape != cropped.shape or check.dtype != cropped.dtype:
                raise ImageValidationError("输出 TIFF 的形状或数据类型发生变化。")
            if not np.array_equal(check, cropped):
                raise ImageValidationError("输出 TIFF 未通过逐像素一致性验证。")
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"TIFF 解码或无损另存失败：{exc}") from exc

    return _result(
        data=output, filename=output_name, media_type=media_type, source_format="TIFF",
        width=width, height=height, out_width=out_width, out_height=out_height,
        dtype=dtype, channels=channels, pages=1,
    )


def prepare_upload(data: bytes, filename: str) -> CropResult:
    """Validate and crop one upload, returning only verified bytes for later analysis."""
    if not data:
        raise ImageValidationError("文件为空。")
    source_format, media_type, suffix = _detect_format(data, filename)
    if source_format == "PNG":
        return _process_png(data, filename, media_type, suffix)
    return _process_tiff(data, filename, media_type, suffix)
