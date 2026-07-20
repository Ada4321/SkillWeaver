import numpy as np
from PIL import Image
import base64
from io import BytesIO
import math
import copy


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def _encode_image(image: Image.Image) -> str:
    """
    Converts a PIL Image to base64-encoded PNG for VLM input.
    """
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    return base64_bytes.decode("utf-8")


def _encode_pcd(pcd: np.array) -> str:
    buffer = BytesIO()
    np.savez_compressed(buffer, arr=pcd)
    base64_bytes = base64.b64encode(buffer.getvalue())
    return base64_bytes.decode("utf-8")


def _decode_base64(base64_str, to_image=False):
    if "base64," in base64_str:
        base64_data = base64_str.split("base64,", 1)[1]
    else:
        base64_data = base64_str
    # _, base64_data = base64_str.split("base64,", 1)
    data = base64.b64decode(base64_data)
    # fix memory leak issue while using BytesIO
    with BytesIO(data) as bio:
        if to_image:
            obj = copy.deepcopy(Image.open(bio))
        else:
            obj = copy.deepcopy(np.load(bio)["arr"])
    return obj


def smart_new_hw(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def take_even_n_frames(frames, n=3):
    total = len(frames)
    if n <= 0:
        return []
    if n >= total:
        return frames[:]
    if n == 1:
        return [frames[0]]

    idx = [(i * (total - 1)) // (n - 1) for i in range(n)]
    return [frames[i] for i in idx]
