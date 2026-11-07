"""Modal entrypoint for offline video processing."""

import shlex

import modal

from src import config
from src.pipeline.service import run_pipeline
from src.video.service import list_videos as find_videos

app = modal.App("aic-keyframes")
volume = modal.Volume.from_name(config.VOLUME_NAME)
cache_code = (
    "from huggingface_hub import snapshot_download; "
    f"snapshot_download({config.MODEL_ID!r}, revision={config.MODEL_REVISION!r}, "
    "allow_patterns=['*.json','*.txt','*.safetensors','*.jinja','tokenizer.model'])"
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-base-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "cmake",
        "build-essential",
        "pkg-config",
        "ffmpeg",
        "libavcodec-dev",
        "libavformat-dev",
        "libavutil-dev",
        "libavdevice-dev",
        "libavfilter-dev",
        "libswresample-dev",
        "libswscale-dev",
        "libgl1",
        "libglib2.0-0",
        "cuda-nvcc-12-6",
        "cuda-cudart-dev-12-6",
        "cuda-driver-dev-12-6",
        "cuda-nvrtc-dev-12-6",
        "cuda-nvml-dev-12-6",
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video"})
    .uv_sync(extras=["pipeline"], frozen=True, extra_options="--no-dev")
    .add_local_file("scripts/build_decord.sh", "/opt/build_decord.sh", copy=True)
    .run_commands("sh /opt/build_decord.sh")
    .workdir("/root")  # Volume reload requires cwd outside its mount.
    .run_commands(f"/.uv/.venv/bin/python -c {shlex.quote(cache_code)}")
    .add_local_python_source("src")
)


@app.function(
    image=image,
    gpu=config.PIPELINE_GPU,
    cpu=(config.CPU_CORES, config.CPU_CORES),
    memory=(config.MEMORY_MIB, config.MEMORY_MIB),
    timeout=1800,
    max_containers=config.MAX_CONTAINERS,
    scaledown_window=config.SCALEDOWN_WINDOW,
    volumes={str(config.WORKSPACE): volume},
)
def process_video(video_path: str, overwrite: bool = False):
    """Process one video and persist its artifacts on the Volume.

    Args:
        video_path: Source path relative to the Volume root.
        overwrite: Replace existing artifacts when true.

    Returns:
        Pipeline report with output paths and selection counts.
    """
    volume.reload()
    result = run_pipeline(video_path, overwrite=overwrite)
    volume.commit()
    return result


@app.function(image=image, volumes={str(config.WORKSPACE): volume})
def list_videos(video_dir: str = "data"):
    """Return paths of all MP4 videos below a Volume directory."""
    return find_videos(config.WORKSPACE, video_dir)


def _process_videos(paths, overwrite=False):
    """Map videos across workers and collect failures without dropping successes."""
    results = process_video.map(
        paths,
        kwargs={"overwrite": overwrite},
        order_outputs=True,
        return_exceptions=True,
    )
    failures = []
    for path, result in zip(paths, results, strict=True):
        if isinstance(result, Exception):
            failures.append(f"{path}: {result}")
            print(failures[-1])
        else:
            print(f"{path}: {result['kept']} keyframes")
    if failures:
        raise RuntimeError("\n".join(failures))


@app.local_entrypoint()
def main(
    video_path: str = "",
    video_dir: str = "data",
    overwrite: bool = False,
):
    """Run one video or discover all videos on the Volume.

    Args:
        video_path: Optional single video, relative to the Volume root.
        video_dir: Directory searched recursively when no video is given.
        overwrite: Replace existing artifacts when true.
    """
    paths = [video_path] if video_path else list_videos.remote(video_dir)
    _process_videos(paths, overwrite)
