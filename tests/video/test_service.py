import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from src.video import service


def test_volume_symlink_and_recursive_case_insensitive_discovery(tmp_path):
    real = tmp_path / "volume"
    source = real / "data" / "nested" / "L26_V035.MP4"
    source.parent.mkdir(parents=True)
    source.touch()
    mount = tmp_path / "workspace"
    mount.symlink_to(real, target_is_directory=True)
    assert service.list_videos(mount) == ["data/nested/L26_V035.MP4"]
    assert service.resolve_source(mount, "data/nested/L26_V035.MP4") == source


def test_duplicate_ids_are_rejected_before_output_can_collide(tmp_path):
    for directory in ("a", "b"):
        source = tmp_path / "data" / directory / "same.mp4"
        source.parent.mkdir(parents=True)
        source.touch()
    with pytest.raises(ValueError, match="Duplicate video ID"):
        service.list_videos(tmp_path)


def test_path_and_symlink_escape_are_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.touch()
    (workspace / "escape.mp4").symlink_to(outside)
    for path in ("../outside.mp4", "escape.mp4", str(outside)):
        with pytest.raises(ValueError):
            service.resolve_source(workspace, path)


def test_empty_discovery_reports_error(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError, match="No MP4"):
        service.list_videos(Path(tmp_path))


def test_source_fingerprint_depends_on_file_contents(tmp_path):
    path = tmp_path / "video.mp4"
    write_video(path, [0, 40, 80])
    first, *_ = service.read_candidates(path, [(0, 2)])
    assert first["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    write_video(path, [0, 80, 160])
    second, *_ = service.read_candidates(path, [(0, 2)])
    assert first["sha256"] != second["sha256"]


def test_media_info_joins_by_video_id_and_requires_watch_url(tmp_path):
    path = tmp_path / "data/media-info-aic25-b1/media-info/L21_V001.json"
    path.parent.mkdir(parents=True)
    info = {"title": "Tin tức", "watch_url": "https://youtube.com/watch?v=Rzpw5WR7nAY"}
    path.write_text(json.dumps(info))
    actual = service.load_media_info(tmp_path, "L21_V001")
    assert actual == info
    path.write_text(json.dumps({"title": "Missing URL"}))
    with pytest.raises(ValueError, match="watch_url"):
        service.load_media_info(tmp_path, "L21_V001")


def write_video(path, times_ms):
    """Create a small VFR video with nonzero timestamps and changing image values."""
    import av
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(times_ms):
            pixels = np.full((16, 16, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts + 9000, Fraction(1, 1000)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_vfr_sampling_retains_rgb_cpu_frames_with_normalized_timestamps(tmp_path):
    path = tmp_path / "vfr.mp4"
    times = [0, 400, 1100, 1400, 2700, 3100, 3500, 4600, 5100, 6200, 6400]
    write_video(path, times)
    source, candidates, frames, timeline = service.read_candidates(
        path, [(0, 4), (6, 10)]
    )
    assert timeline[:-1] == [t * 1000 for t in times]
    assert timeline[-1] > timeline[-2]
    indices = [row["frame_idx"] for row in candidates]
    assert indices == [0, 2, 4, 6, 7, 9]
    assert [row["timestamp_us"] for row in candidates] == [
        times[i] * 1000 for i in indices
    ]
    assert all(
        frame.device.type == "cpu" and frame.shape == (3, 16, 16) for frame in frames
    )
    assert [round(float(frame.float().mean()) / 10) for frame in frames] == indices
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sampling_skips_missed_targets_without_duplicate_frames(tmp_path):
    path = tmp_path / "gaps.mp4"
    write_video(path, [0, 2500, 2600, 3000, 4500])
    _, candidates, frames, _ = service.read_candidates(path, [(0, 4)])
    assert [row["frame_idx"] for row in candidates] == [0, 1, 3, 4]
    assert len(frames) == len(candidates)


@pytest.mark.parametrize("shots", [[], [(-1, 0)], [(1, 0)], [(0, 1), (1, 2)], [(0, 5)]])
def test_invalid_shots_are_rejected(tmp_path, shots):
    path = tmp_path / "video.mp4"
    write_video(path, [0, 40, 80])
    with pytest.raises(ValueError, match="shot|Shot"):
        service.read_candidates(path, shots)


def test_sequential_capture_never_seeks_and_retrieves_only_candidates(
    tmp_path, monkeypatch
):
    from unittest.mock import Mock

    import cv2
    import numpy as np

    path = tmp_path / "video.mp4"
    path.touch()
    capture = Mock()
    position = [-1]

    def grab():
        position[0] += 1
        return position[0] < 61

    capture.grab.side_effect = grab
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FRAME_COUNT: 61,
        cv2.CAP_PROP_POS_MSEC: position[0] * 1000 / 30,
    }[prop]
    capture.retrieve.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    _, rows, _, _ = service.read_candidates(path, [(0, 60)])
    assert [r["frame_idx"] for r in rows] == [0, 30, 60]
    assert capture.retrieve.call_count == 3
    capture.set.assert_not_called()
    capture.release.assert_called_once()


@pytest.mark.parametrize(
    "failure", ["timestamps", "retrieve", "count", "fps", "open", "memory"]
)
def test_capture_is_released_on_invalid_input(tmp_path, monkeypatch, failure):
    from unittest.mock import Mock

    import cv2
    import numpy as np

    from src import config

    path = tmp_path / "video.mp4"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = failure != "open"
    capture.grab.side_effect = [True, True, False]
    capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 0 if failure == "fps" else 25,
        cv2.CAP_PROP_FRAME_COUNT: 3 if failure == "count" else 2,
        cv2.CAP_PROP_POS_MSEC: (
            0 if failure == "timestamps" else 40 * capture.grab.call_count
        ),
    }[prop]
    capture.retrieve.return_value = (
        failure != "retrieve",
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    if failure == "memory":
        monkeypatch.setattr(config, "MAX_FRAME_CACHE_BYTES", 1)
    with pytest.raises((ValueError, MemoryError)):
        service.read_candidates(path, [(0, 1)])
    capture.release.assert_called_once()
