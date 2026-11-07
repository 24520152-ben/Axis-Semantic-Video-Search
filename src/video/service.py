"""Source metadata, discovery and Volume path boundaries."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit


def resolve_source(workspace: Path, relative_path: str) -> Path:
    """Resolve an existing file inside the mounted workspace.

    Args:
        workspace: Volume mount, possibly a symlink.
        relative_path: Source path relative to the mount.

    Returns:
        Resolved file path confined to the Volume.
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
    """
    from src.config import MEDIA_INFO_DIR

    relative = MEDIA_INFO_DIR / f"{video_id}.json"
    path = resolve_source(Path(workspace), str(relative))
    info = json.loads(path.read_text())
    url = urlsplit(info.get("watch_url", ""))
    if url.scheme not in ("https", "http") or not url.netloc:
        raise ValueError(f"Missing or invalid watch_url in {relative}")
    return info


def presentation_timeline(pts, info):
    """Normalize native PTS into microseconds and append the video endpoint.

    Args:
        pts: Strictly increasing decoded presentation timestamps.
        info: Stream metadata and end_pts collected during decoding.

    Returns:
        Frame timestamps followed by the exclusive end time, in microseconds.
    """
    if not len(pts) or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Expected increasing presentation timestamps")
    origin, time_base = int(pts[0]), Fraction(info["time_base"])
    times = [round((int(p) - origin) * time_base * 1_000_000) for p in pts]
    end = times[-1] + round(1_000_000 / Fraction(info["fps"]))
    if "end_pts" in info:
        end = round((info["end_pts"] - origin) * time_base * 1_000_000)
    elif info["duration_pts"] is not None and info["start_pts"] is not None:
        end = round(
            (info["start_pts"] + info["duration_pts"] - origin) * time_base * 1_000_000
        )
    if any(b <= a for a, b in zip(times, times[1:])) or end <= times[-1]:
        raise ValueError("Invalid presentation timeline or stream duration")
    return times + [end]


class DecordFrames:
    """Decode candidate frames once and retain their CPU tensors for export.

    Args:
        reader: CPU Decord VideoReader.
        indices: Source frame indices in candidate order.
    """

    def __init__(self, reader, indices):
        self.reader = reader
        self.indices = indices
        self.cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        import torch

        from src.config import MAX_FRAME_CACHE_BYTES

        single = isinstance(index, int)
        if isinstance(index, slice):
            indices = self.indices[index]
        elif single:
            indices = [self.indices[index]]
        else:
            indices = [self.indices[i] for i in index]
        if not indices:
            return []
        missing = list(dict.fromkeys(i for i in indices if i not in self.cache))
        if missing:
            batch = self.reader.get_batch(missing)
            # Retain the original DLPack storage; export never decodes these again.
            tensor = (
                batch
                if isinstance(batch, torch.Tensor)
                else torch.from_dlpack(batch.to_dlpack())
            )
            required = len(self.indices) * tensor[0].numel() * tensor.element_size()
            if required > MAX_FRAME_CACHE_BYTES:
                raise MemoryError(
                    f"Candidate RGB cache needs {required / 1024**3:.1f} GiB; "
                    f"limit is {MAX_FRAME_CACHE_BYTES / 1024**3:.1f} GiB"
                )
            self.cache.update(
                zip(missing, tensor.permute(0, 3, 1, 2).unbind(0), strict=True)
            )
        frames = [self.cache[i] for i in indices]
        return frames[0] if single else frames


def _read_source(path):
    """Read exact packet timestamps and stream metadata without decoding pixels.

    Args:
        path: Source video path.

    Returns:
        Source metadata and presentation timestamps in microseconds.
    """
    import av

    with Path(path).open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Expected exactly one video stream for Decord indexing")
        stream = container.streams.video[0]
        if not stream.average_rate or stream.average_rate <= 0 or not stream.time_base:
            raise ValueError("Missing video FPS or time base")
        source = {
            "sha256": fingerprint,
            "stream_index": stream.index,
            "fps": str(stream.average_rate),
            "time_base": str(stream.time_base),
            "start_pts": stream.start_time,
            "duration_pts": stream.duration,
        }
        packets = []
        for packet in container.demux(stream):
            if not packet.size:
                continue
            if packet.pts is None:
                raise ValueError("Missing packet presentation timestamp")
            packets.append((packet.pts, packet.duration))
        packets.sort()
        ends = [pts + duration for pts, duration in packets if duration > 0]
        if ends:
            source["end_pts"] = max(ends)
        pts = [pts for pts, _ in packets]
        timeline = presentation_timeline(pts, source)
        # Decord reports seconds relative to the stream's start time, in FP32.
        origin = source["start_pts"]
        if origin is None:
            raise ValueError("Missing stream start PTS for Decord timestamp validation")
        source["first_timestamp_seconds"] = float((pts[0] - origin) * stream.time_base)
    return source, timeline


def read_candidates(path, shots):
    """Index shot candidates and decode their pixels lazily on CPU.

    Args:
        path: Source video path.
        shots: Ordered, disjoint, inclusive frame ranges.

    Returns:
        Source metadata, candidate rows, lazy CPU frames and exact PTS timeline.
    """
    import decord
    import numpy as np

    previous_end = -1
    for start, end in shots:
        if start <= previous_end or end < start:
            raise ValueError("Shot ranges must be ordered and disjoint")
        previous_end = end
    if not shots:
        raise ValueError("Expected at least one shot")
    source, timeline = _read_source(path)
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    if len(reader) != len(timeline) - 1:
        raise ValueError("Decord frame count differs from source packet timestamps")
    if shots[-1][1] >= len(reader):
        raise ValueError("Shot range exceeds the indexed video")
    reported = np.asarray(
        reader.get_frame_timestamp(range(len(reader))), dtype=np.float64
    )
    expected = np.asarray(timeline[:-1], dtype=np.float64) / 1_000_000
    expected += source.pop("first_timestamp_seconds")
    tolerance = np.maximum(
        np.abs(np.spacing(expected.astype(np.float32))) * 2, 0.000002
    )
    if (
        reported.shape != (len(reader), 2)
        or not np.isfinite(reported).all()
        or np.any(np.abs(reported[:, 0] - expected) > tolerance)
    ):
        raise ValueError(
            "Decord frame timestamps differ from source presentation timestamps"
        )
    candidates = []
    for shot_id, (start, end) in enumerate(shots):
        next_sample = timeline[start]
        for frame_idx in range(start, end + 1):
            timestamp = timeline[frame_idx]
            if timestamp < next_sample:
                continue
            candidates.append(
                {
                    "candidate_idx": len(candidates),
                    "shot_id": shot_id,
                    "frame_idx": frame_idx,
                    "timestamp_us": timestamp,
                }
            )
            next_sample += ((timestamp - next_sample) // 1_000_000 + 1) * 1_000_000
    frames = DecordFrames(reader, [row["frame_idx"] for row in candidates])
    return source, candidates, frames, timeline
