"""Write retained keyframe images and seven-column Parquet records."""

import tempfile
from pathlib import Path

from src.config import EMBEDDING_DIM, JPEG_QUALITY


def output_paths(root, video_id):
    """Return the image directory and Parquet shard paths."""
    root = Path(root)
    return (
        root / "images" / video_id,
        root / "records" / f"{video_id}.parquet",
    )


def check_output(root, video_id, overwrite=False):
    """Reject existing artifacts unless replacement was requested."""
    if not overwrite and any(path.exists() for path in output_paths(root, video_id)):
        raise FileExistsError(
            f"Output exists for {video_id}; use --overwrite to replace it"
        )


def _publish(sources, targets, staging):
    """Publish staged artifacts and restore previous files on rename failure."""
    # Keep old artifacts until the new ones are complete; restore on rename failure.
    moved, backups = [], []
    try:
        for index, (source, target) in enumerate(zip(sources, targets)):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = staging / f"backup-{index}"
                target.rename(backup)
                backups.append((backup, target))
            source.rename(target)
            moved.append((target, source))
    except Exception:
        for target, source in reversed(moved):
            target.rename(source)
        for backup, target in reversed(backups):
            backup.rename(target)
        raise


def _write_images(directory, frames, candidates, selected):
    """Write only retained JPEGs, named by their source frame indices."""
    from PIL import Image

    for index in selected:
        frame_idx = candidates[index]["frame_idx"]
        # Only retained pixels leave the GPU, when a JPEG is actually needed.
        pixels = frames[index].permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(pixels).save(
            directory / f"{frame_idx}.jpg", format="JPEG", quality=JPEG_QUALITY
        )


def _validate_vectors(vectors, candidates, frames, selected):
    """Validate aligned rows without changing the supplied BF16 embeddings."""
    import torch

    if (
        not isinstance(vectors, torch.Tensor)
        or vectors.dtype != torch.bfloat16
        or vectors.shape != (len(candidates), EMBEDDING_DIM)
        or len(frames) != len(candidates)
    ):
        raise ValueError("Expected one bfloat16[2048] vector and frame per candidate")
    if (
        not selected
        or sorted(set(selected)) != selected
        or not 0 <= selected[0] <= selected[-1] < len(candidates)
    ):
        raise ValueError("Invalid selected frame indices")
    if not torch.isfinite(vectors).all() or not torch.any(vectors != 0, dim=1).all():
        raise ValueError("Invalid embedding vectors")
    indices = [row["frame_idx"] for row in candidates]
    times = [row["timestamp_us"] for row in candidates]
    if (
        indices != sorted(set(indices))
        or indices[0] < 0
        or times[0] < 0
        or any(b <= a for a, b in zip(times, times[1:]))
    ):
        raise ValueError("Candidate indices and timestamps must be strictly ordered")
    return vectors


def _write_records(path, video_id, source, watch_url, candidates, selected, vectors):
    """Write only retained frames with their aligned embeddings and image URIs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    schema = pa.schema(
        [
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("video_id", pa.string(), nullable=False),
            pa.field("frame_idx", pa.int64(), nullable=False),
            pa.field("timestamp_us", pa.int64(), nullable=False),
            pa.field("image_uri", pa.string(), nullable=False),
            pa.field("watch_url", pa.string(), nullable=False),
            pa.field("embedding", pa.binary(EMBEDDING_DIM * 2), nullable=False),
        ]
    )
    candidates = [candidates[index] for index in selected]
    vectors = vectors[selected].detach().cpu().contiguous().view(torch.uint8)
    data = {
        "frame_id": [
            f"{source['sha256']}:{source['stream_index']}:{c['frame_idx']}"
            for c in candidates
        ],
        "video_id": [video_id] * len(candidates),
        "frame_idx": [c["frame_idx"] for c in candidates],
        "timestamp_us": [c["timestamp_us"] for c in candidates],
        "image_uri": [f"images/{video_id}/{c['frame_idx']}.jpg" for c in candidates],
        "watch_url": [watch_url] * len(candidates),
        "embedding": [row.numpy().tobytes() for row in vectors],
    }
    pq.write_table(pa.Table.from_pydict(data, schema=schema), path, compression="zstd")


def export_selected(
    root,
    path,
    frames,
    candidates,
    selection,
    vectors,
    *,
    source,
    watch_url,
    overwrite=False,
):
    """Stage and publish retained JPEGs and their matching Parquet records.

    Args:
        root: Output root directory.
        path: Source video path, used for its ID.
        frames: Lazy sequence of candidate RGB uint8 CHW tensors.
        candidates: Candidate metadata in image order.
        selection: Selected candidate indices.
        vectors: BF16 candidate tensors in the same order.
        source: Source fingerprint, stream identity and timing metadata.
        watch_url: Original video URL from organizer media-info.
        overwrite: Replace an existing video output when true.

    Returns:
        Output paths and the number of retained frames.
    """
    root, video_id = Path(root), Path(path).stem
    check_output(root, video_id, overwrite)
    selected = selection["selected"]
    vectors = _validate_vectors(vectors, candidates, frames, selected)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".keyframes-", dir=root) as temporary:
        staging = Path(temporary)
        images, records = output_paths(staging, video_id)
        for directory in (images, records.parent):
            directory.mkdir(parents=True, exist_ok=True)
        _write_records(
            records, video_id, source, watch_url, candidates, selected, vectors
        )
        _write_images(images, frames, candidates, selected)
        check_output(root, video_id, overwrite)
        _publish((images, records), output_paths(root, video_id), staging)
    images, records = output_paths(root, video_id)
    return {
        "output_dir": str(images),
        "records_path": str(records),
        "kept": len(selected),
    }
