"""Compare production DALI shot detection with OmniShotCut's file-path API.

Run all videos: uv run modal run scripts/benchmarks/benchmark_dali.py
Results are isolated from production outputs and written locally and to the Volume.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

from scripts.process_videos import image, volume
from src import config

app = modal.App("aic-benchmark-dali")


@app.function(
    image=image,
    serialized=True,
    gpu=os.environ.get("BENCHMARK_GPU", "A10G"),
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=14400,
    max_containers=5,
    volumes={str(config.WORKSPACE): volume},
)
def benchmark(repeats: int, run_id: str, video_path: str, video_index: int = 0):
    import gc
    import hashlib
    import importlib.metadata
    import platform
    import statistics
    import threading
    import time
    from unittest.mock import patch

    import numpy as np
    import omnishotcut
    import torch
    from omnishotcut.label_correspondence import unique_intra_label_mapping

    from src.shot_detection.service import _decode_video, _infer_ranges
    from src.video.service import list_videos

    if repeats < 1:
        raise ValueError("repeats must be positive")
    volume.reload()
    torch.set_num_threads(4)
    if not video_path:
        raise ValueError("Each worker requires exactly one video")
    if video_path not in list_videos(config.WORKSPACE):
        raise ValueError(f"Video not found: {video_path}")
    paths = [video_path]
    gpu_name = torch.cuda.get_device_name()
    started = time.monotonic()
    wrapper = omnishotcut.load(
        "uva-cv-lab/OmniShotCut_v1.5", filename="OmniShotCut_ckpt.pth"
    )
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    args = wrapper._model_args
    general = unique_intra_label_mapping["general"]
    original_decode = omnishotcut._decode_video

    def rss_tree(pid, include_children=True):
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            rss = next(
                int(line.split()[1]) * 1024
                for line in status.splitlines()
                if line.startswith("VmRSS:")
            )
        except (OSError, StopIteration, ValueError):
            return 0
        if include_children:
            try:
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            except OSError:
                children = []
            rss += sum(rss_tree(int(child)) for child in children)
        return rss

    class MemoryMonitor:
        def __enter__(self):
            self.stop = threading.Event()
            self.peak_rss = 0
            self.baseline_rss = rss_tree(os.getpid(), include_children=False)

            def sample():
                while not self.stop.is_set():
                    rss = rss_tree(os.getpid())
                    self.peak_rss = max(self.peak_rss, rss)
                    self.stop.wait(0.05)

            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
            return self

        def __exit__(self, *_):
            self.stop.set()
            self.thread.join()

    def measure(path, variant):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = {}
        with MemoryMonitor() as memory:
            begin = time.monotonic()
            if variant == "dali":
                frames = _decode_video(path, args.process_width, args.process_height)
                torch.cuda.synchronize()
                prepared = time.monotonic()
                stats["frame_count"] = len(frames)
                ranges, labels, _ = _infer_ranges(
                    frames, wrapper._model, args, overlap=10
                )
                shots = [
                    [int(start), int(end)]
                    for (start, end), label in zip(ranges, labels, strict=True)
                    if label == general
                ]
                torch.cuda.synchronize()
                end = time.monotonic()
                stats["prepare_seconds"] = prepared - begin
                del frames
            else:

                def timed_decode(*decode_args, **kwargs):
                    decode_start = time.monotonic()
                    frames = original_decode(*decode_args, **kwargs)
                    stats["prepare_seconds"] = time.monotonic() - decode_start
                    stats["frame_count"] = len(frames)
                    return frames

                with patch.object(omnishotcut, "_decode_video", timed_decode):
                    shots = wrapper.inference(str(path), mode="clean_shot", overlap=10)
                torch.cuda.synchronize()
                end = time.monotonic()
            stats.update(
                variant=variant,
                total_seconds=end - begin,
                inference_and_window_preprocess_seconds=end
                - begin
                - stats["prepare_seconds"],
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        stats["peak_process_tree_rss_bytes"] = memory.peak_rss
        stats["baseline_process_rss_bytes"] = memory.baseline_rss
        stats["shot_count"] = len(shots)
        stats["ranges_sha256"] = hashlib.sha256(json.dumps(shots).encode()).hexdigest()
        return stats, shots

    manifest_path = (
        config.WORKSPACE
        / "benchmarks"
        / run_id
        / "shots"
        / f"{Path(video_path).stem}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shot_manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "schema": "inclusive clean-shot frame ranges by Volume-relative source path",
            "shots": {},
        }
    )

    def save_shots(relative, exclusive_shots):
        shot_manifest["shots"][relative] = [
            [start, end - 1] for start, end in exclusive_shots
        ]
        manifest_path.write_text(json.dumps(shot_manifest, indent=2))
        volume.commit()

    # Warm only one synthetic inference window per path. No extra full-video pass.
    from omnishotcut.engine import _run_on_numpy

    warm_start = time.monotonic()
    warm_frames = np.zeros(
        (args.max_process_window_length, args.process_height, args.process_width, 3),
        dtype=np.uint8,
    )
    _run_on_numpy(warm_frames, wrapper._model, args, 10)
    gpu_warm_frames = torch.from_numpy(warm_frames).cuda()
    _infer_ranges(gpu_warm_frames, wrapper._model, args, overlap=10)
    torch.cuda.synchronize()
    del warm_frames, gpu_warm_frames
    warmup_seconds = time.monotonic() - warm_start
    output = {
        "benchmark": "DALI vs official OmniShotCut file-path API",
        "run_id": run_id,
        "gpu": gpu_name,
        "cpu_cores": 4,
        "memory_mib": 16384,
        "torch_num_threads": 4,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "nvidia-dali-cuda130",
                "omnishotcut",
                "numpy",
                "opencv-python",
            )
        },
        "repeats": repeats,
        "model_load_seconds_excluded": load_seconds,
        "warmup_seconds_excluded": warmup_seconds,
        "notes": [
            "Both variants use the same loaded GPU model and overlap=10.",
            "Official baseline calls wrapper.inference(path, mode=clean_shot) unchanged; only decode timing is intercepted.",
            "DALI variant uses production _decode_video and _infer_ranges.",
            "Preparation includes decoder construction/build and decode+resize; model load is excluded.",
            "CUDA is synchronized before/after timers; variant order alternates per video/repetition.",
            "RAM is sampled process-tree RSS every 50ms, including baseline FFmpeg subprocess.",
            "Torch VRAM peaks omit unmanaged DALI/driver allocations; do not use as total GPU memory.",
            "One synthetic inference window per path warms the model; no full-video warmup or filesystem cache flush.",
        ],
        "videos": [],
    }
    result_dir = (
        config.WORKSPACE / "benchmarks" / run_id / "videos" / Path(video_path).stem
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for index, relative in enumerate(paths, start=video_index):
        path = config.WORKSPACE / relative
        row = {"video": relative, "trials": [], "errors": []}
        reference_shots = {}
        for repeat in range(repeats):
            variants = (
                ("dali", "official")
                if (index + repeat) % 2 == 0
                else ("official", "dali")
            )
            for variant in variants:
                try:
                    metrics, shots = measure(path, variant)
                    metrics["repeat"] = repeat
                    row["trials"].append(metrics)
                    reference_shots.setdefault(variant, shots)
                    if variant == "dali" and relative not in shot_manifest["shots"]:
                        save_shots(relative, shots)
                    print(
                        f"{relative} {variant} repeat={repeat}: {metrics['total_seconds']:.3f}s prepare={metrics['prepare_seconds']:.3f}s shots={len(shots)}",
                        flush=True,
                    )
                except Exception as error:
                    row["errors"].append(
                        {"variant": variant, "repeat": repeat, "error": repr(error)}
                    )
                    print(f"ERROR {relative} {variant}: {error!r}", flush=True)
        for variant in ("dali", "official"):
            trials = [trial for trial in row["trials"] if trial["variant"] == variant]
            row[variant] = {
                "successful_trials": len(trials),
                "median_total_seconds": (
                    statistics.median(t["total_seconds"] for t in trials)
                    if trials
                    else None
                ),
                "median_prepare_seconds": (
                    statistics.median(t["prepare_seconds"] for t in trials)
                    if trials
                    else None
                ),
            }
        if len(reference_shots) == 2:
            dali_ranges = {tuple(s) for s in reference_shots["dali"]}
            official_ranges = {tuple(s) for s in reference_shots["official"]}
            row["correctness"] = {
                "exact_clean_ranges_equal": reference_shots["dali"]
                == reference_shots["official"],
                "shared_exact_clean_ranges": len(dali_ranges & official_ranges),
                "dali_only_ranges": sorted(dali_ranges - official_ranges),
                "official_only_ranges": sorted(official_ranges - dali_ranges),
                "dali_clean_ranges": len(dali_ranges),
                "official_clean_ranges": len(official_ranges),
                "frame_counts_equal": len({t["frame_count"] for t in row["trials"]})
                == 1,
            }
        output["videos"].append(row)
        (result_dir / "dali.json").write_text(json.dumps(output, indent=2))
        volume.commit()
    output["aggregate"] = {
        "videos": len(output["videos"]),
        "errors": sum(len(row["errors"]) for row in output["videos"]),
        "dali_sum_video_median_seconds": sum(
            row["dali"]["median_total_seconds"] or 0 for row in output["videos"]
        ),
        "official_sum_video_median_seconds": sum(
            row["official"]["median_total_seconds"] or 0 for row in output["videos"]
        ),
    }
    (result_dir / "dali.json").write_text(json.dumps(output, indent=2))
    volume.commit()
    return output


@app.function(image=image, serialized=True, volumes={str(config.WORKSPACE): volume})
def list_videos(video_dir: str = "data"):
    from src.video.service import list_videos as find_videos

    volume.reload()
    return find_videos(config.WORKSPACE, video_dir)


@app.local_entrypoint()
def main(video_path: str = "", repeats: int = 1):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    paths = list_videos.remote()
    if video_path:
        if video_path not in paths:
            raise ValueError(f"Video not found: {video_path}")
        paths = [video_path]
    prefix = "single-dali-" if video_path else "dali-"
    run_id = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = {
        "benchmark": "DALI vs official OmniShotCut file-path API",
        "run_id": run_id,
        "repeats": repeats,
        "max_containers": 5,
        "source_videos": paths,
        "videos": [],
        "failures": [],
        "worker_metadata": [],
    }
    destination = Path(
        "benchmarks/results/single/dali.json"
        if video_path
        else "benchmarks/results/dali.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = benchmark.map(
        [repeats] * len(paths),
        [run_id] * len(paths),
        paths,
        range(len(paths)),
        order_outputs=True,
        return_exceptions=True,
    )
    for path, result in zip(paths, results, strict=True):
        if isinstance(result, Exception):
            output["failures"].append({"video": path, "error": repr(result)})
            print(f"FAILED {path}: {result!r}", flush=True)
        else:
            output["videos"].extend(result["videos"])
            output["worker_metadata"].append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("videos", "aggregate")
                }
            )
            print(f"COMPLETE {path}", flush=True)
        destination.write_text(json.dumps(output, indent=2))
    output["aggregate"] = {
        "videos": len(output["videos"]),
        "errors": len(output["failures"])
        + sum(len(row["errors"]) for row in output["videos"]),
        "dali_sum_video_median_seconds": sum(
            row["dali"]["median_total_seconds"] or 0 for row in output["videos"]
        ),
        "official_sum_video_median_seconds": sum(
            row["official"]["median_total_seconds"] or 0 for row in output["videos"]
        ),
    }
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["aggregate"], indent=2))
    print(
        f"Results: {destination}; per-video Volume results: benchmarks/{run_id}/videos/"
    )
    if output["aggregate"]["errors"]:
        raise RuntimeError(f"Benchmark trials failed; see {destination}")
