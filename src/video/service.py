"""Discover videos and sample RGB frames sequentially with OpenCV."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def resolve_source(workspace: Path, relative_path: str) -> Path:
    """Resolve an existing file inside the mounted workspace.

    Args:
        workspace: Volume mount, possibly a symlink.
        relative_path: Source path relative to the mount.

    Returns:
        Resolved file path confined to the Volume.

    Raises:
        ValueError: The path resolves outside the workspace.
        FileNotFoundError: The resolved path is not an existing file.
    """
    # Modal mounts can be symlinks into /__modal/volumes/... .
    workspace = workspace.resolve()
    source = (workspace / relative_path).resolve()
    source.relative_to(workspace)
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def list_videos(workspace: Path, video_dir: str = "data") -> list[str]:
    """Find MP4 videos recursively and reject colliding video IDs.

    Args:
        workspace: Mounted Volume root.
        video_dir: Directory within the Volume to search.

    Returns:
        Sorted source paths relative to the resolved Volume root.

    Raises:
        ValueError: The search directory resolves outside the workspace, or
            multiple videos have the same filename stem.
        NotADirectoryError: The search path is not an existing directory.
        FileNotFoundError: The directory contains no MP4 videos.
    """
    workspace = workspace.resolve()
    source = (workspace / video_dir).resolve()
    source.relative_to(workspace)
    if not source.is_dir():
        raise NotADirectoryError(source)
    paths = sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"
    )
    if not paths:
        raise FileNotFoundError(f"No MP4 videos found in {source}")
    seen = {}
    for path in paths:
        if path.stem in seen:
            raise ValueError(
                f"Duplicate video ID {path.stem}: {seen[path.stem]} and {path}"
            )
        seen[path.stem] = path
    return [str(path.relative_to(workspace)) for path in paths]


def load_media_info(workspace, video_id):
    """Read organizer metadata and require a usable watch URL.

    Args:
        workspace: Mounted Volume root.
        video_id: Source video's filename stem.

    Returns:
        Organizer metadata including the original watch URL.

    Raises:
        FileNotFoundError: The video's metadata file does not exist.
        ValueError: The metadata path escapes the workspace, the JSON is
            invalid, or watch_url is missing or not an HTTP(S) URL.
    """
    from src.config import MEDIA_INFO_DIR

    relative = MEDIA_INFO_DIR / f"{video_id}.json"
    path = resolve_source(Path(workspace), str(relative))
    info = json.loads(path.read_text())
    url = urlsplit(info.get("watch_url", ""))
    if url.scheme not in ("https", "http") or not url.netloc:
        raise ValueError(f"Missing or invalid watch_url in {relative}")
    return info


def read_candidates(path, shots):
    """Read video once in presentation order, retaining sampled RGB CPU frames.

    Advance with grab() and retrieve pixels only for candidates at roughly one
    frame per second within each shot, always including the shot's first frame.
    Read through the end of the video to build the complete timeline. Normalize
    OpenCV timestamps to the first frame and estimate the exclusive endpoint
    from the final timestamp and reported average FPS. Release the capture on
    success and failure; returned frames remain in CPU memory for embedding
    and export without further decoding.

    Args:
        path: Filesystem path to the source video. OpenCV selects its default
            first video stream.
        shots: Nonempty sequence of ordered, disjoint (start, end) pairs with
            inclusive, zero-based source frame indices.

    Returns:
        A tuple containing:
            source: Metadata dictionary with the source SHA-256, video-stream
                ordinal (stream_index=0), and reported FPS as a string.
            candidates: Metadata dictionaries in presentation order, with
                candidate_idx, shot_id, frame_idx and timestamp_us fields.
            frames: List of RGB uint8 CPU tensors with shape (3, H, W), aligned
                with candidates and retained independently of the capture.
            timeline: Integer presentation timestamps in microseconds for all
                decoded frames, followed by an estimated exclusive endpoint.

    Raises:
        OSError: The source file cannot be opened or read for hashing.
        ValueError: Shot ranges are empty, invalid or out of bounds; the video
            cannot be opened; FPS or timestamps are invalid; a candidate cannot
            be retrieved; or decoded and reported positive frame counts differ.
        MemoryError: Retained candidate RGB bytes exceed MAX_FRAME_CACHE_BYTES.
        cv2.error: The OpenCV backend fails while decoding or converting pixels.
    """
    import math

    import cv2
    import torch

    from src.config import MAX_FRAME_CACHE_BYTES

    previous_end = -1
    for start, end in shots:
        if start <= previous_end or end < start:
            raise ValueError("Shot ranges must be ordered and disjoint")
        previous_end = end
    if not shots:
        raise ValueError("Expected at least one shot")
    with Path(path).open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames, candidates, timeline = [], [], []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Missing video FPS")
        shot_id, frame_idx, next_sample = 0, 0, None
        origin, cache_bytes = None, 0
        while capture.grab():
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) * 1000
            if not math.isfinite(timestamp):
                raise ValueError("Invalid OpenCV presentation timestamp")
            if origin is None:
                origin = timestamp
            timestamp = round(timestamp - origin)
            if timeline and timestamp <= timeline[-1]:
                raise ValueError("Expected increasing OpenCV presentation timestamps")
            timeline.append(timestamp)
            while shot_id < len(shots) and frame_idx > shots[shot_id][1]:
                shot_id += 1
                next_sample = None
            if shot_id < len(shots):
                start, end = shots[shot_id]
                if start <= frame_idx <= end:
                    if next_sample is None:
                        next_sample = timestamp
                    if timestamp >= next_sample:
                        ok, image = capture.retrieve()
                        if not ok or image is None:
                            raise ValueError(f"Cannot retrieve frame {frame_idx}")
                        required = image.size
                        if cache_bytes + required > MAX_FRAME_CACHE_BYTES:
                            raise MemoryError(
                                "Candidate RGB cache exceeds configured limit"
                            )
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
                        cache_bytes += rgb.nbytes
                        candidates.append(
                            {
                                "candidate_idx": len(candidates),
                                "shot_id": shot_id,
                                "frame_idx": frame_idx,
                                "timestamp_us": timestamp,
                            }
                        )
                        next_sample += (
                            (timestamp - next_sample) // 1_000_000 + 1
                        ) * 1_000_000
            frame_idx += 1
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not timeline or shots[-1][1] >= frame_idx:
            raise ValueError("Shot range exceeds decoded video")
        if (
            math.isfinite(reported_count)
            and reported_count > 0
            and round(reported_count) != frame_idx
        ):
            raise ValueError(
                "Decoded frame count differs from reported video frame count"
            )
        timeline.append(timeline[-1] + max(1, round(1_000_000 / fps)))
        source = {
            "sha256": fingerprint,
            # OpenCV's default first video stream, indexed among video streams.
            "stream_index": 0,
            "fps": str(fps),
        }
        return source, candidates, frames, timeline
    finally:
        capture.release()
