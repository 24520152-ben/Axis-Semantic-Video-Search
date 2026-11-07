"""Run the video-to-keyframes workflow using small domain functions."""

from pathlib import Path

from src.config import BATCH_SIZE, DATASET_VERSION, WORKSPACE
from src.embedding import service as embedding
from src.keyframes import export
from src.keyframes import service as keyframes
from src.shot_detection.service import detect_shots
from src.video import service as video


def run_pipeline(video_path, *, workspace=WORKSPACE, overwrite=False):
    """Run sampling, embedding, selection and export for one video.

    Args:
        video_path: Source path relative to the workspace.
        workspace: Mounted Volume root.
        overwrite: Replace an existing output when true.

    Returns:
        Output paths and selection counts.
    """
    path = video.resolve_source(Path(workspace), video_path)
    root = Path(workspace).resolve() / "data" / "retrieval" / DATASET_VERSION
    export.check_output(root, path.stem, overwrite)
    media = video.load_media_info(workspace, path.stem)
    shots = detect_shots(path)
    source, candidates, frames, timeline = video.read_candidates(path, shots)
    model = embedding.load_model()
    vectors = embedding.embed_candidates(model, frames, BATCH_SIZE)
    selection = keyframes.select_frames(
        vectors, candidates, shots, timeline_us=timeline
    )
    output = export.export_selected(
        root,
        path,
        frames,
        candidates,
        selection,
        vectors,
        source=source,
        watch_url=media["watch_url"],
        overwrite=overwrite,
    )
    return {
        "video": video_path,
        "shots": len(shots),
        "candidates": len(candidates),
        "kept": selection["kept"],
        "max_gap_seconds": selection["max_gap_seconds"],
        "output": output,
    }
