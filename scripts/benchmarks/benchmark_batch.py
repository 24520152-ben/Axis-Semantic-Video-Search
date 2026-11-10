"""Measure candidate embedding batches on all source videos using a Modal A10G worker by default."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

from scripts.process_videos import image as pipeline_image
from scripts.process_videos import volume
from src import config

image = pipeline_image.add_local_python_source("scripts.process_videos")

app = modal.App(os.environ.get("BENCHMARK_APP_NAME", "aic-benchmark-batch"))
_MODEL = None
BENCHMARK_GPU = os.environ.get("BENCHMARK_GPU", "A10G")
BENCHMARK_WORKERS = 5


def _rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


@app.function(
    image=image,
    gpu=BENCHMARK_GPU,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=10800,
    max_containers=BENCHMARK_WORKERS,
    scaledown_window=300,
    volumes={str(config.WORKSPACE): volume},
)
def benchmark_video(
    video_path: str,
    run_id: str,
    repeats: int = 3,
    reference_from_measured_batch16: bool = False,
):
    """Prepare candidates once, then measure each batch size in rotated order."""
    import gc
    import hashlib
    import importlib.metadata
    import statistics
    import threading
    import time
    import traceback

    import torch
    import torch.nn.functional as functional

    from src.embedding.service import embed_candidates, load_model
    from src.shot_detection.service import detect_shots
    from src.video.service import read_candidates

    global _MODEL
    volume.reload()
    started_setup = time.monotonic()
    path = config.WORKSPACE / video_path
    destination = config.WORKSPACE / "benchmarks" / run_id / "batch" / (path.stem + ".json")
    if destination.exists():
        existing = json.loads(destination.read_text())
        with path.open("rb") as file:
            current_sha = hashlib.file_digest(file, "sha256").hexdigest()
        if (
            existing.get("dataset_version") == config.DATASET_VERSION
            and existing.get("reader") == "production_opencv_sequential"
            and existing.get("sha256") == current_sha
            and existing.get("gpu") == torch.cuda.get_device_name()
            and existing.get("method")
            == "cached CPU RGB candidates; wall time includes H2D, preprocessing, encode and CPU embeddings; CUDA synchronized; no shot/decode/model-load time"
            and existing.get("reference_from_measured_batch16", False)
            == reference_from_measured_batch16
            and len(existing.get("summary", {})) == 5
            and all(
                len(
                    [
                        row
                        for row in existing["runs"]
                        if row["batch_size"] == size and row["status"] == "ok"
                    ]
                )
                >= repeats
                or any(
                    row["batch_size"] == size and row["status"] == "oom" for row in existing["runs"]
                )
                for size in (1, 2, 4, 8, 16)
            )
        ):
            print(f"RESUME completed {video_path}", flush=True)
            return existing
    shots = detect_shots(path)
    shot_source = "production_detect_shots"
    source, candidates, frames, _ = read_candidates(path, shots)
    setup_seconds = time.monotonic() - started_setup
    print(
        json.dumps(
            {
                "stage": "setup_complete",
                "video": video_path,
                "candidates": len(candidates),
                "setup_seconds": setup_seconds,
            }
        ),
        flush=True,
    )
    started_model = time.monotonic()
    loaded_model = _MODEL is None
    if loaded_model:
        _MODEL = load_model()
    model_load_seconds = time.monotonic() - started_model
    # Single-video mode uses the measured batch-16 pass as its reference.
    reference = None
    reference_seconds = 0.0
    if not reference_from_measured_batch16:
        torch.cuda.synchronize()
        started_reference = time.monotonic()
        reference = embed_candidates(_MODEL, frames, config.BATCH_SIZE)
        torch.cuda.synchronize()
        reference_seconds = time.monotonic() - started_reference
        print(
            json.dumps(
                {
                    "stage": "reference_complete",
                    "video": video_path,
                    "candidates": len(candidates),
                    "seconds": reference_seconds,
                }
            ),
            flush=True,
        )
    result = {
        "video": video_path,
        "dataset_version": config.DATASET_VERSION,
        "reader": "production_opencv_sequential",
        "sampling_policy": "all uniform 1-second candidates within clean shots; no selection",
        "reference_batch_size": config.BATCH_SIZE,
        "sha256": source["sha256"],
        "candidates": len(candidates),
        "shots": len(shots),
        "shot_source": shot_source,
        "setup_seconds_excluded": setup_seconds,
        "model_loaded_this_call": loaded_model,
        "model_load_seconds_excluded": model_load_seconds,
        "batch16_reference_seconds_excluded": reference_seconds,
        "reference_dtype": (str(reference.dtype) if reference is not None else "torch.bfloat16"),
        "reference_from_measured_batch16": reference_from_measured_batch16,
        "gpu": torch.cuda.get_device_name(),
        "resources": {
            "cpu_cores": 4,
            "memory_mib": 16384,
            "gpu": torch.cuda.get_device_name(),
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "sentence-transformers",
                "transformers",
                "opencv-python",
            )
        },
        "method": "cached CPU RGB candidates; wall time includes H2D, preprocessing, encode and CPU embeddings; CUDA synchronized; no shot/decode/model-load time",
        "runs": [],
    }
    destination = config.WORKSPACE / "benchmarks" / run_id / "batch" / (path.stem + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(result, indent=2))
        volume.commit()

    save()
    failed_sizes = set()
    sizes = [1, 2, 4, 8, 16]
    # Rotate by video identity and repetition to reduce fixed execution-order bias.
    offset = sum(path.stem.encode()) % len(sizes)
    for repeat in range(repeats):
        order = (
            [16, 1, 2, 4, 8] if reference_from_measured_batch16 else sizes[offset:] + sizes[:offset]
        )
        if repeat % 2:
            order = list(reversed(order))
        offset = (offset + 1) % len(sizes)
        for size in order:
            if size in failed_sizes:
                continue
            gc.collect()
            torch.cuda.empty_cache()
            row = {"repeat": repeat + 1, "batch_size": size}
            try:
                # Warm the actual batch shape; excluded from measurement.
                warmup = embed_candidates(_MODEL, frames[:size], size)
                del warmup
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                row["gpu_allocated_start_bytes"] = torch.cuda.memory_allocated()
                rss = [_rss_bytes()]
                stop = threading.Event()

                def sample_rss():
                    while not stop.wait(0.05):
                        rss.append(_rss_bytes())

                watcher = threading.Thread(target=sample_rss, daemon=True)
                watcher.start()
                started = time.perf_counter()
                try:
                    vectors = embed_candidates(_MODEL, frames, size)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                finally:
                    stop.set()
                    watcher.join()
                    rss.append(_rss_bytes())
                row.update(
                    status="ok",
                    seconds=elapsed,
                    candidates_per_second=len(frames) / elapsed,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_process_rss_bytes=max(rss),
                )
                if reference is None:
                    assert size == config.BATCH_SIZE
                    reference = vectors.clone()
                assert vectors.shape == reference.shape and vectors.dtype == torch.bfloat16
                cosine = functional.cosine_similarity(vectors.float(), reference.float(), dim=1)
                row["reference_batch16_cosine_min"] = float(cosine.min())
                row["reference_batch16_cosine_mean"] = float(cosine.mean())
                row["reference_batch16_max_abs_difference"] = float(
                    (vectors.float() - reference.float()).abs().max()
                )
                row["reference_batch16_agrees"] = bool(
                    torch.isfinite(vectors).all() and cosine.min() >= 0.995
                )
                del vectors
            except torch.OutOfMemoryError as error:
                row.update(status="oom", error=str(error))
                failed_sizes.add(size)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as error:
                row.update(status="error", error=str(error), traceback=traceback.format_exc())
                failed_sizes.add(size)
            result["runs"].append(row)
            save()
            print(json.dumps({"video": path.stem, **row}), flush=True)
    result["summary"] = {}
    for size in sizes:
        successful = [
            row for row in result["runs"] if row["batch_size"] == size and row["status"] == "ok"
        ]
        result["summary"][str(size)] = {
            "successful_repeats": len(successful),
            "median_seconds": (
                statistics.median(row["seconds"] for row in successful) if successful else None
            ),
            "max_peak_gpu_allocated_bytes": max(
                (row["peak_gpu_allocated_bytes"] for row in successful), default=None
            ),
            "all_reference_agree": (
                all(row["reference_batch16_agrees"] for row in successful) if successful else None
            ),
        }
    save()
    del frames, reference
    gc.collect()
    return result


@app.function(image=image, volumes={str(config.WORKSPACE): volume})
def list_videos(video_dir: str = "data"):
    from src.video.service import list_videos as find_videos

    volume.reload()
    return find_videos(config.WORKSPACE, video_dir)


@app.local_entrypoint()
def main(
    repeats: int = 1,
    run_id: str = "",
    video_path: str = "",
    start_index: int = 0,
    end_index: int = 0,
    result_suffix: str = "",
    single_run: bool = False,
):
    """Map videos across up to five workers and retain results locally."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    run_id = run_id or "batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    paths = list_videos.remote()
    if video_path:
        paths = [path for path in paths if path == video_path]
    paths = paths[start_index : end_index or None]
    assert paths
    if single_run:
        assert len(paths) == 1 and repeats == 1, (
            "single_run requires one explicit video and repeats=1"
        )
    output = Path(
        "benchmarks/results/uniform1fps/single" if single_run else "benchmarks/results/uniform1fps"
    )
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "run_id": run_id,
        "repeats": repeats,
        "source_videos": paths,
        "videos": [],
        "failures": [],
    }

    aggregate_path = output / (
        "batch-all" + ("-" + result_suffix if result_suffix else "") + ".json"
    )
    mapped = benchmark_video.map(
        paths,
        kwargs={
            "run_id": run_id,
            "repeats": repeats,
            "reference_from_measured_batch16": True,
        },
        order_outputs=True,
        return_exceptions=True,
    )
    for path, result in zip(paths, mapped, strict=True):
        if isinstance(result, Exception):
            results["failures"].append({"video": path, "error": str(result)})
            print(f"FAILED {path}: {result}", flush=True)
        else:
            results["videos"].append(result)
            (output / f"batch-{Path(path).stem}.json").write_text(json.dumps(result, indent=2))
        aggregate_path.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "completed": len(results["videos"]),
                "failures": results["failures"],
                "run_id": run_id,
            }
        ),
        flush=True,
    )
    failed_trials = any(
        row["status"] == "error" or (row["status"] == "ok" and not row["reference_batch16_agrees"])
        for video in results["videos"]
        for row in video["runs"]
    )
    if results["failures"] or failed_trials:
        raise RuntimeError(f"Benchmark failures; see {aggregate_path}")
