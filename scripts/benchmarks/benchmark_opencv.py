"""Benchmark full candidate reading against two OpenCV sampling strategies on Modal.

Run from repo root: uv run modal run scripts/benchmarks/benchmark_opencv.py
All timings are warm-source, fresh-reader timings. Production datasets are read-only.
"""

import gc
import hashlib
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import modal

from scripts.process_videos import image as pipeline_image
from src import config

app = modal.App("aic-benchmark-opencv")
volume = modal.Volume.from_name(config.VOLUME_NAME)
image = pipeline_image.uv_pip_install(
    "av>=19.0.2", "decord==0.6.0"
).add_local_python_source("scripts.process_videos", "scripts.benchmarks.legacy_video")
REMOTE_ROOT = "benchmarks/decord-cpu-20261107"
BENCHMARK_GPU = os.environ.get("BENCHMARK_GPU", "A10G")
VARIANTS = ("pyav_decord", "opencv_sequential", "opencv_seek")


def _rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


class _Memory:
    def __init__(self):
        self.baseline = _rss()
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, _rss())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, _rss())
        self.stop.set()
        self.thread.join()


def _clean():
    gc.collect()
    # Return freed CPU image allocations when glibc supports this.
    import ctypes

    try:
        ctypes.CDLL(None).malloc_trim(0)
    except AttributeError:
        pass


def _hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _opencv_open(path):
    import cv2

    fingerprint = _hash(path)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = {
        "sha256": fingerprint,
        "frame_count": round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "backend": capture.getBackendName(),
        "decoder_threads": capture.get(cv2.CAP_PROP_N_THREADS),
    }
    return info, capture


def _opencv_sample(capture, indices, strategy):
    import cv2
    import torch

    frames, times, positions = [], [], []
    next_index = 0
    try:
        for index in indices:
            if strategy == "opencv_seek":
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError(f"OpenCV seek failed at {index}")
                ok, image = capture.read()
            else:
                while next_index <= index:
                    if not capture.grab():
                        raise RuntimeError(f"OpenCV grab failed at {next_index}")
                    next_index += 1
                ok, image = capture.retrieve()
            if not ok:
                raise RuntimeError(f"OpenCV read failed at {index}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) * 1000)
            positions.append(capture.get(cv2.CAP_PROP_POS_FRAMES))
    finally:
        capture.release()
    return frames, times, positions


def _pixel_probes(frames):
    """Return sparse RGB grids from 9 candidates; diagnostic work is untimed."""
    import numpy as np

    positions = sorted(set(np.linspace(0, len(frames) - 1, 9).astype(int).tolist()))
    return {
        str(position): frames[position].permute(1, 2, 0).numpy()[::32, ::32].copy()
        for position in positions
    }


def _measure(path, shots, indices, expected_timestamps, variant):
    from scripts.benchmarks.legacy_video import read_candidates

    _clean()
    frames = None
    with _Memory() as memory:
        start = time.perf_counter()
        if variant == "pyav_decord":
            source, candidates, reader, timeline = read_candidates(path, shots)
            found_indices = [row["frame_idx"] for row in candidates]
            if found_indices != indices:
                raise AssertionError("Candidate indices changed")
            metadata_end = time.perf_counter()
            frames = []
            for position in range(0, len(reader), 4):
                frames.extend(reader[position : position + 4])
            sampling_end = time.perf_counter()
            metadata = {
                "sha256": source["sha256"],
                "frame_count": len(timeline) - 1,
                "fps": float(__import__("fractions").Fraction(source["fps"])),
                "width": int(frames[0].shape[2]),
                "height": int(frames[0].shape[1]),
                "backend": "Decord CPU + PyAV",
                "decoder_threads": "default",
            }
            timestamps = [row["timestamp_us"] for row in candidates]
            positions = [index + 1 for index in indices]
        else:
            metadata, capture = _opencv_open(path)
            metadata_end = time.perf_counter()
            frames, timestamps, positions = _opencv_sample(capture, indices, variant)
            sampling_end = time.perf_counter()
    probes = _pixel_probes(frames)
    timestamp_errors = [
        abs(float(a) - float(b))
        for a, b in zip(timestamps, expected_timestamps, strict=True)
    ]
    measurement = {
        "variant": variant,
        "metadata_seconds": metadata_end - start,
        "sampling_seconds": sampling_end - metadata_end,
        "combined_seconds": sampling_end - start,
        "candidate_count": len(frames),
        "peak_rss_bytes": memory.peak,
        "baseline_rss_bytes": memory.baseline,
        "incremental_peak_rss_bytes": max(0, memory.peak - memory.baseline),
        "metadata": metadata,
        "timestamp_max_abs_error_us": max(timestamp_errors),
        "timestamp_mean_abs_error_us": statistics.mean(timestamp_errors),
        "position_mismatch_count": sum(
            abs(pos - index - 1) > 0.01
            for pos, index in zip(positions, indices, strict=True)
        ),
        "image_shapes": sorted({tuple(frame.shape) for frame in frames}),
    }
    del frames
    return measurement, probes


def _probe_difference(actual, reference):
    import numpy as np

    errors, max_error, matching = [], 0, 0
    for key, value in actual.items():
        if value.shape != reference[key].shape:
            return {"shape_mismatch": True}
        difference = np.abs(value.astype(np.int16) - reference[key].astype(np.int16))
        errors.append(float(difference.mean()))
        max_error = max(max_error, int(difference.max()))
        matching += bool(np.array_equal(value, reference[key]))
    return {
        "probe_candidates": len(errors),
        "exact_matching_probe_candidates": matching,
        "mean_abs_rgb_error": statistics.mean(errors),
        "max_abs_rgb_error": max_error,
        "grid_stride_pixels": 32,
    }


@app.function(image=image, volumes={str(config.WORKSPACE): volume}, timeout=600)
def list_videos(video_dir: str = "data"):
    from src.video.service import list_videos as find_videos

    volume.reload()
    return find_videos(config.WORKSPACE, video_dir)


@app.function(
    image=image,
    gpu=BENCHMARK_GPU,
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=7200,
    max_containers=5,
    scaledown_window=60,
    volumes={str(config.WORKSPACE): volume},
)
def benchmark_video(video_path: str, repetitions: int = 3, single: bool = False):
    import cv2
    import decord
    import numpy as np
    import torch

    from scripts.benchmarks.legacy_video import read_candidates
    from src.shot_detection.service import detect_shots
    from src.video.service import resolve_source

    volume.reload()
    path = resolve_source(config.WORKSPACE, video_path)
    shared_path = config.WORKSPACE / REMOTE_ROOT / "shot_ranges.json"
    shots = None
    setup_source = "local_detection_outside_timing"
    if shared_path.exists():
        shared = json.loads(shared_path.read_text())
        # Shared file maps source-relative paths to inclusive clean-shot ranges.
        shots = shared.get("shots", shared).get(video_path)
        if isinstance(shots, dict):
            shots = shots.get("shots") or shots.get("shot_ranges")
        if shots:
            setup_source = "shared_dali_shot_ranges"
    if not shots:
        shots = detect_shots(path)
    # Exact candidates and timestamp reference, prepared once outside measured runs.
    source, candidates, initial_reader, timeline = read_candidates(path, shots)
    indices = [row["frame_idx"] for row in candidates]
    timestamps = [row["timestamp_us"] for row in candidates]
    # Prime decoder/library initialization with one frame only, not a full-video pass.
    initial_reader.reader.get_batch([0])
    del initial_reader
    warm_capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    ok, warm_frame = warm_capture.read()
    if not ok:
        raise RuntimeError("OpenCV one-frame warmup failed")
    cv2.cvtColor(warm_frame, cv2.COLOR_BGR2RGB)
    warm_capture.release()
    del warm_capture, warm_frame
    reference_probes = None
    measurements = []
    for repeat in range(repetitions):
        # Rotate order to counterbalance warming and transient CPU effects.
        order = VARIANTS[repeat % 3 :] + VARIANTS[: repeat % 3]
        for variant in order:
            measurement, probes = _measure(path, shots, indices, timestamps, variant)
            measurement["repeat"] = repeat + 1
            if reference_probes is None:
                assert variant == "pyav_decord"
                reference_probes = probes
            measurement["probe_comparison_to_decord"] = _probe_difference(
                probes, reference_probes
            )
            measurements.append(measurement)
            print(
                f"{path.stem} repeat={repeat + 1} {variant}: {measurement['combined_seconds']:.3f}s",
                flush=True,
            )
    summary = {
        variant: {
            metric: statistics.median(
                row[metric] for row in measurements if row["variant"] == variant
            )
            for metric in (
                "metadata_seconds",
                "sampling_seconds",
                "combined_seconds",
                "peak_rss_bytes",
            )
        }
        for variant in VARIANTS
    }
    result = {
        "video": video_path,
        "source_sha256": source["sha256"],
        "frame_count": len(timeline) - 1,
        "candidate_count": len(indices),
        "shot_count": len(shots),
        "setup_shot_source": setup_source,
        "repetitions": repetitions,
        "summary_medians": summary,
        "measurements": measurements,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "cpu": platform.processor(),
            "cpu_allocation": 4,
            "memory_mib": 16384,
        },
        "versions": {
            "opencv": cv2.__version__,
            "decord": decord.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "methodology": {
            "source_cache": "warm source from untimed metadata/candidate setup; fresh readers each trial, one-frame decoder initialization only",
            "model_setup": "Shot detection/reference candidate construction outside measurement",
            "metadata_scope": "SHA-256 in all branches; PyAV exact timeline+Decord reader indexing/validation vs OpenCV basic properties",
            "sampling_scope": "All identical candidate indices retained as RGB uint8 CHW tensors; historical Decord get_batch batches of4",
            "timestamps": "OpenCV POS_MSEC sampled after each decoded candidate, compared to PyAV normalized native PTS",
            "gpu_timing": "Decode benchmarks are CPU-only; GPU is used for untimed setup only",
            "order": "Rotate three variants each repetition; 20ms RSS sampling includes retained RGB cache",
            "probe_scope": "Sparse RGB grids every32pixels from9 evenly spaced candidates, not full-video pixel equality",
        },
    }
    destination = (
        config.WORKSPACE
        / REMOTE_ROOT
        / ("single/opencv" if single else "opencv")
        / f"{path.stem}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(repetitions: int = 1, video_path: str = "", single: bool = False):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if single and not video_path:
        raise ValueError("--single requires --video-path")
    paths = [video_path] if video_path else list_videos.remote()
    destination = Path(
        "benchmarks/results/single/opencv.json"
        if single
        else "benchmarks/results/opencv.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": "pyav_decord_vs_opencv", "videos": [], "errors": []}
    for path, result in zip(
        paths,
        benchmark_video.map(
            paths,
            kwargs={"repetitions": repetitions, "single": single},
            order_outputs=True,
            return_exceptions=True,
        ),
        strict=True,
    ):
        if isinstance(result, Exception):
            payload["errors"].append({"video": path, "error": str(result)})
            print(f"ERROR {path}: {result}", flush=True)
        else:
            payload["videos"].append(result)
            print(f"COMPLETE {path}: {result['summary_medians']}", flush=True)
        destination.write_text(json.dumps(payload, indent=2))
    if payload["errors"]:
        raise RuntimeError(f"{len(payload['errors'])} videos failed; see {destination}")
