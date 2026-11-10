"""Exercise the benchmark's production reader locally without Modal execution."""

from fractions import Fraction
from unittest.mock import Mock

import av
import numpy as np

from scripts.benchmarks import benchmark_opencv
from src.video import service


def test_sequential_benchmark_calls_production_reader_and_keeps_every_sample(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate([0, 1000, 2200, 6600]):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), index * 50, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    shots = [(0, 3)]
    _, rows, _, timeline = service.read_candidates(path, shots)
    spy = Mock(wraps=service.read_candidates)
    monkeypatch.setattr(service, "read_candidates", spy)
    measurement, probes = benchmark_opencv._measure(
        path,
        shots,
        [r["frame_idx"] for r in rows],
        [r["timestamp_us"] for r in rows],
        "opencv_sequential",
    )
    spy.assert_called_once_with(path, shots)
    assert measurement["candidate_count"] == 4
    assert measurement["metadata"]["frame_count"] == len(timeline) - 1
    assert measurement["metadata_seconds"] is None
    assert measurement["combined_seconds"] == measurement["sampling_seconds"]
    assert measurement["timestamp_max_abs_error_us"] == 0
    assert len(probes) == 4
