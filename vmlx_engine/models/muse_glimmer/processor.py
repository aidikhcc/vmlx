# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer image/video preprocessing.

The shipped bundles name ``MuseGlimmerProcessor`` / ``MuseGlimmerImageProcessor``
/ ``MuseGlimmerVideoProcessor`` in ``processor_config.json``. Those classes are
newer than the pinned runtime's transformers/mlx-vlm packages, so
``AutoProcessor`` degrades to a text-only processor unless vMLX registers this
source-owned implementation.

Ground truth is the exact bundle sidecar plus the upstream Transformers Muse
implementation at revision ``c7e57f79348480f73d3ef0ad8c47f807ef1378c8``.
In particular, native image spans carry image-start/end delimiters and native
video spans carry timestamps and one placeholder run per temporal group.
"""

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Patch grids are snapped to this multiple so a merge block is never split.
_MERGE_ALIGNED = "patch_size * merge_size"


class MuseGlimmerImageProcessor:
    """Resize -> normalize -> patchify, returning ``(patches, grid_thw)``."""

    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        max_image_tokens: int = 4096,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        rescale_factor: float = 1.0 / 255.0,
        **_: Any,
    ) -> None:
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.max_image_tokens = int(max_image_tokens)
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        self.rescale_factor = float(rescale_factor)

    # ---- geometry ---------------------------------------------------------

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def smart_resize(
        self, height: int, width: int, max_tokens: Optional[int] = None
    ) -> Tuple[int, int]:
        """Match the native aspect-ratio search under the merged-token budget."""
        factor = self.factor
        budget = self.max_image_tokens if max_tokens is None else int(max_tokens)
        if height <= 0 or width <= 0 or budget <= 0:
            raise ValueError(
                "Muse Glimmer: height, width and token budget must be positive."
            )

        ideal_h = height / factor
        ideal_w = width / factor
        ratio = ideal_w / ideal_h if ideal_h > 0 else 1.0
        if ideal_h * ideal_w > budget:
            ideal_h = math.sqrt(budget / ratio)
            ideal_w = ideal_h * ratio
        candidates = set(
            itertools.product(
                (math.floor(ideal_h), math.ceil(ideal_h)),
                (math.floor(ideal_w), math.ceil(ideal_w)),
            )
        )
        candidates = {
            (grid_h, grid_w)
            for grid_h, grid_w in candidates
            if grid_h >= 1 and grid_w >= 1 and grid_h * grid_w <= budget
        }
        if not candidates:
            candidates = {(max(1, round(ideal_h)), max(1, round(ideal_w)))}
        grid_h, grid_w = min(
            candidates,
            key=lambda grid: abs(grid[0] / grid[1] - height / width),
        )
        return grid_h * factor, grid_w * factor

    # ---- pixels -----------------------------------------------------------

    @staticmethod
    def _as_pil(image):
        """Accept what callers actually hand over, not just PIL images.

        Depending on the surface, an image arrives as a PIL image, a numpy
        array, a filesystem path, or a data/HTTP URL. Feeding a path straight
        to ``Image.fromarray`` produces a 0-d unicode array and the opaque
        "Cannot handle this data type: (1, 1), <U64".
        """
        import base64
        import io
        from pathlib import Path

        from PIL import Image

        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)):
            text = str(image)
            if text.startswith("data:"):
                payload = text.split(",", 1)[1] if "," in text else ""
                return Image.open(io.BytesIO(base64.b64decode(payload)))
            if text.startswith(("http://", "https://")):
                import urllib.request

                from vmlx_engine.http_security import MediaUrlError, validate_outbound_media_url

                try:
                    validate_outbound_media_url(text)
                except MediaUrlError as exc:
                    raise ValueError(str(exc)) from exc
                with urllib.request.urlopen(text, timeout=60) as response:
                    return Image.open(io.BytesIO(response.read()))
            return Image.open(text)
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image))

        # Arrays arrive in several shapes. The engine's video path in
        # particular hands over frames still carrying leading singleton axes
        # and float pixel data — "(1, 1, 448, 448), <f4" straight into
        # Image.fromarray is the opaque failure that made video unusable.
        array = np.asarray(image)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.dtype != np.uint8:
            peak = float(array.max()) if array.size else 0.0
            if peak <= 1.0 + 1e-6:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def _to_chw(self, image, size: Tuple[int, int]) -> np.ndarray:
        """RGB -> resize -> rescale -> normalize, as ``[3, H, W]`` float32."""
        from PIL import Image

        image = self._as_pil(image)
        image = image.convert("RGB").resize(
            (size[1], size[0]), resample=Image.Resampling.LANCZOS
        )
        arr = np.asarray(image, dtype=np.float32) * self.rescale_factor
        arr = arr.transpose(2, 0, 1)
        return (arr - self.image_mean) / self.image_std

    def patchify(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """``[3,H,W]`` frames -> ``(L, C*T*p*p)`` rows, merge-block-major.

        A still image is DUPLICATED along time to fill the temporal patch; a
        video consumes two real frames per patch.
        """
        temporal = self.temporal_patch_size
        if len(frames) % temporal:
            frames = list(frames) + [frames[-1]] * (temporal - len(frames) % temporal)
        stack = np.stack(frames, axis=0)  # (N, 3, H, W)

        n, channels, height, width = stack.shape
        p, m = self.patch_size, self.merge_size
        grid_t, grid_h, grid_w = n // temporal, height // p, width // p

        patches = stack.reshape(
            grid_t, temporal, channels, grid_h // m, m, p, grid_w // m, m, p
        )
        # -> (grid_t, gh/m, gw/m, m, m, C, T, p, p): each merge block contiguous
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(
            grid_t * grid_h * grid_w, channels * temporal * p * p
        )
        return np.ascontiguousarray(flat, dtype=np.float32), (grid_t, grid_h, grid_w)

    def __call__(self, images: Sequence[Any]) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        all_patches, grids = [], []
        for image in images:
            pil = self._as_pil(image)
            size = self.smart_resize(pil.height, pil.width)
            patches, grid = self.patchify([self._to_chw(pil, size)])
            all_patches.append(patches)
            grids.append(grid)
        if not all_patches:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.concatenate(all_patches, axis=0), grids


class MuseGlimmerVideoProcessor(MuseGlimmerImageProcessor):
    """Frame sampling plus the per-frame token budget the bundle declares."""

    def __init__(
        self,
        fps: float = 2.0,
        num_frames: int = 96,
        max_video_frame_tokens: int = 144,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fps = float(fps)
        self.num_frames = int(num_frames)
        self.max_video_frame_tokens = int(max_video_frame_tokens)

    def __call__(  # type: ignore[override]
        self, frames: Sequence[Any]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        frames = list(frames)[: self.num_frames]
        if not frames:
            return np.zeros((0, 0), dtype=np.float32), []

        first = self._as_pil(frames[0])
        # Sized once, from the first frame, under the per-FRAME budget.
        size = self.smart_resize(
            first.height, first.width, max_tokens=self.max_video_frame_tokens
        )
        chw = [self._to_chw(f, size) for f in frames]
        patches, grid = self.patchify(chw)
        return patches, [grid]


def _as_frame_sequence(videos: Any) -> List[Any]:
    """Normalize whatever the caller calls "videos" into a list of frames.

    The engine's video fetch hands back a single 4-D ``(frames, H, W, C)``
    ndarray, not a list. Only unwrapping list/tuple meant that array reached
    ``_as_pil`` whole and PIL raised on it — so the video path could not run at
    all, while the panel advertised video.

    Accepts: a 4-D array (frames stacked), a list of frames, or a list
    containing one of those.
    """
    if videos is None:
        return []
    candidate = videos
    # a list/tuple wrapping the real payload (the usual single-video case)
    if isinstance(candidate, (list, tuple)) and len(candidate) == 1:
        candidate = candidate[0]

    array = getattr(candidate, "ndim", None)
    if array == 4:
        return [np.asarray(frame) for frame in candidate]
    if array == 3:
        # one frame handed over bare
        return [np.asarray(candidate)]
    if isinstance(candidate, (list, tuple)):
        return list(candidate)
    return [candidate]


def _as_video_sequences(videos: Any) -> List[List[Any]]:
    """Normalize processor input into one frame sequence per video item.

    The engine hands one video over as ``[ndarray[T,H,W,C]]`` and multiple
    videos as ``[ndarray[T,H,W,C], ...]``.  Treating the outer list as one
    frame sequence works for the first shape but feeds whole 4-D clips to PIL
    for the second.  Keep media-item boundaries so repeated video turns remain
    one-to-one with their prompt placeholders and cache side keys.
    """
    if videos is None:
        return []
    ndim = getattr(videos, "ndim", None)
    if ndim in (3, 4):
        return [_as_frame_sequence(videos)]
    if not isinstance(videos, (list, tuple)):
        return [_as_frame_sequence(videos)]
    if not videos:
        return []

    # A flat list of 3-D arrays/PIL images is one video's frame sequence.
    # A list containing 4-D arrays or nested frame lists is multiple videos.
    if any(
        getattr(item, "ndim", None) == 4 or isinstance(item, (list, tuple))
        for item in videos
    ):
        return [_as_frame_sequence(item) for item in videos]
    return [list(videos)]


def _split_patches_by_grid(
    patches: np.ndarray,
    grids: Sequence[Tuple[int, int, int]],
) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Split a concatenated raw-patch buffer back into media-owned spans."""
    items: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    cursor = 0
    for raw_grid in grids:
        grid = tuple(int(v) for v in raw_grid)
        span = grid[0] * grid[1] * grid[2]
        items.append((patches[cursor : cursor + span], grid))
        cursor += span
    if cursor != int(patches.shape[0]):
        raise ValueError(
            "Muse Glimmer: media grids account for "
            f"{cursor} raw patches but the processor produced {patches.shape[0]}."
        )
    return items


def expand_media_placeholders(
    input_ids: List[int],
    grids: Sequence[Tuple[int, int, int]],
    token_id: int,
    merge_size: int,
) -> List[int]:
    """Replace each single placeholder with one id per MERGED cell.

    The count must equal the tower's output row count. If it does not, the
    scatter that writes vision features into the text stream misaligns silently
    and the model reads shifted features as if they were real.
    """
    positions = [i for i, t in enumerate(input_ids) if t == token_id]
    if not positions:
        return list(input_ids)
    if len(positions) != len(grids):
        raise ValueError(
            f"Muse Glimmer: {len(positions)} placeholder(s) for token {token_id} but "
            f"{len(grids)} media grid(s); the prompt and the media disagree."
        )

    out: List[int] = []
    cursor = 0
    for position, (grid_t, grid_h, grid_w) in zip(positions, grids):
        out.extend(input_ids[cursor:position])
        cells = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
        out.extend([token_id] * cells)
        cursor = position + 1
    out.extend(input_ids[cursor:])
    return out


def merged_token_count(grid: Tuple[int, int, int], merge_size: int) -> int:
    grid_t, grid_h, grid_w = (int(v) for v in grid)
    return grid_t * (grid_h // merge_size) * (grid_w // merge_size)


class MuseGlimmerProcessor:
    """Tokenizer + image/video processor, in the shape mlx-vlm calls.

    ``AutoProcessor`` cannot build this — the bundle names classes that exist
    nowhere in transformers — and mlx-vlm's ``prepare_inputs`` only consults
    ``processor.image_processor`` for resizing before calling
    ``processor(text=..., images=...)``. So attaching an image processor is not
    enough: the processor itself has to turn images into ``pixel_values`` and
    expand the placeholder. Without that the call silently returns text-only
    inputs and the model confabulates a description of an image it never saw.
    """

    def __init__(self, tokenizer, image_processor=None, video_processor=None,
                 image_token_id: int = 200092, video_token_id: int = 200091):
        self.tokenizer = tokenizer
        self.image_processor = image_processor or MuseGlimmerImageProcessor()
        self.video_processor = video_processor or MuseGlimmerVideoProcessor()
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.image_start_token_id = self._single_token_id(
            "<|image_start|>", 200080
        )
        self.image_end_token_id = self._single_token_id("<|image_end|>", 200081)
        self.video_start_token_id = self._single_token_id("<|vid_start|>", 200082)
        self.video_end_token_id = self._single_token_id("<|vid_end|>", 200083)
        self.video_separator_token_id = self._single_token_id(
            "<|vid_frame_separator|>", 200087
        )

    def _single_token_id(self, token: str, fallback: int) -> int:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            try:
                value = convert(token)
                if isinstance(value, int) and value >= 0:
                    return value
            except Exception:
                pass
        try:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
                return int(encoded[0])
        except Exception:
            pass
        return int(fallback)

    def _encode_fragment(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if not isinstance(encoded, (list, tuple)):
            raise TypeError(
                "Muse Glimmer tokenizer returned a non-sequence for video timestamp text."
            )
        return [int(value) for value in encoded]

    @staticmethod
    def _media_value(values: Any, index: int, default: Any = None) -> Any:
        if values is None:
            return default
        if isinstance(values, (list, tuple)):
            if not values:
                return default
            # A flat numeric list is one video's per-frame timestamp sequence.
            if index == 0 and all(isinstance(value, (int, float)) for value in values):
                return values
            return values[index] if index < len(values) else default
        return values if index == 0 else default

    def _video_group_timestamps(
        self,
        *,
        video_index: int,
        frame_count: int,
        grid_t: int,
        fps: Any,
        video_timestamps: Any,
    ) -> List[float]:
        raw = self._media_value(video_timestamps, video_index)
        if raw is not None:
            try:
                timestamps = [float(value) for value in raw]
            except (TypeError, ValueError):
                timestamps = []
        else:
            timestamps = []
        if not timestamps:
            rate = self._media_value(fps, video_index, self.video_processor.fps)
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                rate = float(self.video_processor.fps)
            rate = rate if rate > 0 else float(self.video_processor.fps)
            timestamps = [index / rate for index in range(frame_count)]

        grouped = timestamps[:: self.video_processor.temporal_patch_size][:grid_t]
        while len(grouped) < grid_t:
            grouped.append(grouped[-1] if grouped else 0.0)
        return grouped

    def _image_replacement(self, grid: Tuple[int, int, int]) -> List[int]:
        return (
            [self.image_start_token_id]
            + [self.image_token_id]
            * merged_token_count(grid, self.image_processor.merge_size)
            + [self.image_end_token_id]
        )

    def _video_replacement(
        self,
        grid: Tuple[int, int, int],
        timestamps: Sequence[float],
    ) -> List[int]:
        grid_t, grid_h, grid_w = (int(value) for value in grid)
        tokens_per_group = (
            (grid_h // self.video_processor.merge_size)
            * (grid_w // self.video_processor.merge_size)
        )
        output = [self.video_start_token_id]
        for group in range(grid_t):
            timestamp = timestamps[group] if group < len(timestamps) else 0.0
            output.extend(self._encode_fragment(f"Time: {timestamp:.1f}s"))
            output.extend([self.video_token_id] * tokens_per_group)
            output.append(
                self.video_separator_token_id
                if group < grid_t - 1
                else self.video_end_token_id
            )
        return output

    # ---- delegation -------------------------------------------------------

    def __getattr__(self, name):
        # Anything not defined here (detokenizer, eos ids, chat template, the
        # stopping criteria vMLX attaches) falls through to the tokenizer.
        return getattr(self.__dict__["tokenizer"], name)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # ---- the call mlx-vlm makes ------------------------------------------

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        return_tensors=None,
        fps=None,
        video_timestamps=None,
        **kwargs,
    ):
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError("Muse Glimmer processor handles one prompt at a time.")
            text = text[0]

        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        out = {}

        # Images and videos are NOT mutually exclusive, and their raw patches
        # must follow PLACEHOLDER order.  The previous implementation always
        # concatenated video first and image second, then exposed only separate
        # grids.  The model chose image_grid_thw when both were present, encoded
        # a wrongly sliced prefix of that buffer, and live image->video history
        # failed with 765 placeholders vs 345 feature rows.
        original_ids = list(ids)
        image_items: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        video_items: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        video_group_timestamps: List[List[float]] = []
        image_grids: List[Tuple[int, int, int]] = []
        video_grids: List[Tuple[int, int, int]] = []
        if images:
            patches, image_grids = self.image_processor(images)
            image_items = _split_patches_by_grid(patches, image_grids)
        if videos:
            for video_index, frames in enumerate(_as_video_sequences(videos)):
                patches, grids = self.video_processor(frames)
                video_grids.extend(grids)
                video_items.extend(_split_patches_by_grid(patches, grids))
                if len(grids) != 1:
                    raise ValueError(
                        "Muse Glimmer: one video payload must produce exactly one grid."
                    )
                video_group_timestamps.append(
                    self._video_group_timestamps(
                        video_index=video_index,
                        frame_count=len(frames),
                        grid_t=int(grids[0][0]),
                        fps=fps,
                        video_timestamps=video_timestamps,
                    )
                )

        image_queue = list(image_items)
        video_queue = list(video_items)
        video_time_queue = list(video_group_timestamps)
        ordered_media: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        expanded_ids: List[int] = []
        for token_id in original_ids:
            if token_id == self.image_token_id:
                if not image_queue:
                    raise ValueError(
                        "Muse Glimmer: image placeholder has no matching image payload."
                    )
                item = image_queue.pop(0)
                ordered_media.append(item)
                expanded_ids.extend(self._image_replacement(item[1]))
            elif token_id == self.video_token_id:
                if not video_queue or not video_time_queue:
                    raise ValueError(
                        "Muse Glimmer: video placeholder has no matching video payload."
                    )
                item = video_queue.pop(0)
                ordered_media.append(item)
                expanded_ids.extend(
                    self._video_replacement(item[1], video_time_queue.pop(0))
                )
            else:
                expanded_ids.append(int(token_id))
        if image_queue or video_queue or video_time_queue:
            raise ValueError(
                "Muse Glimmer: media payload count exceeds prompt placeholders "
                f"(images={len(image_queue)}, videos={len(video_queue)} unmatched)."
            )
        ids = expanded_ids
        if ordered_media:
            ordered_patches = [patches for patches, _grid in ordered_media]
            out["pixel_values"] = (
                ordered_patches[0]
                if len(ordered_patches) == 1
                else np.concatenate(ordered_patches, axis=0)
            )
            # This unified grid is the authoritative contract for the combined
            # pixel buffer.  Separate grids remain available for diagnostics.
            out["grid_thw"] = [grid for _patches, grid in ordered_media]
        if image_grids:
            out["image_grid_thw"] = image_grids
        if video_grids:
            out["video_grid_thw"] = video_grids

        out["input_ids"] = [ids]
        out["attention_mask"] = [[1] * len(ids)]

        if return_tensors == "mlx":
            import mlx.core as mx

            for key in ("input_ids", "attention_mask"):
                out[key] = mx.array(out[key])
            if "pixel_values" in out:
                out["pixel_values"] = mx.array(out["pixel_values"])
        return out
