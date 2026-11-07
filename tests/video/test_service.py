import hashlib
import json
import sys
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

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


def test_vfr_timeline_uses_pts_and_normalizes_nonzero_origin():
    info = {"fps": "25", "time_base": "1/1000", "start_pts": 9000, "duration_pts": 240}
    assert service.presentation_timeline([9000, 9040, 9100, 9200], info) == [
        0,
        40000,
        100000,
        200000,
        240000,
    ]


def test_fractional_time_base_is_converted_without_float_drift():
    info = {"fps": "30000/1001", "time_base": "1/90000", "start_pts": 9000, "duration_pts": 9009}
    assert service.presentation_timeline([9000, 12003, 15006], info) == [0, 33367, 66733, 100100]


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


@pytest.fixture(autouse=True)
def fake_decord(monkeypatch):
    """Exercise real packet indexing with a CPU stand-in for NVDEC/DLPack."""
    import av
    import numpy as np
    import torch

    readers = []

    class Batch:
        def __init__(self, tensor):
            self.tensor = tensor

        def to_dlpack(self):
            return torch.utils.dlpack.to_dlpack(self.tensor)

    class Reader:
        def __init__(self, path, *, ctx):
            assert ctx == ("gpu", 0)
            self.calls = []
            with av.open(path) as container:
                stream = container.streams.video[0]
                self.width, self.height = stream.width, stream.height
                pairs = sorted((p.pts, p.duration) for p in container.demux(stream) if p.size)
                self.timestamps = np.asarray([
                    [(pts - stream.start_time) * stream.time_base,
                     (pts + duration - stream.start_time) * stream.time_base]
                    for pts, duration in pairs
                ], dtype=np.float32)
            readers.append(self)

        def __len__(self):
            return len(self.timestamps)

        def get_frame_timestamp(self, indices):
            return self.timestamps[list(indices)]

        def get_batch(self, indices):
            self.calls.append(list(indices))
            return Batch(torch.stack([
                torch.full((self.height, self.width, 3), index, dtype=torch.uint8)
                for index in indices
            ]))

    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(
        VideoReader=Reader, gpu=lambda index: ("gpu", index)
    ))
    return readers


def test_vfr_sampling_keeps_exact_pts_and_decodes_only_requested_batches(tmp_path, fake_decord):
    path = tmp_path / "vfr.mp4"
    times_ms = [0, 400, 1100, 1400, 2700, 3100, 3500, 4600, 5100, 6200, 6400]
    write_video(path, times_ms)
    source, candidates, frames, timeline = service.read_candidates(path, [(0, 4), (6, 10)])
    assert timeline[:-1] == [value * 1000 for value in times_ms]
    assert timeline[-1] > timeline[-2]
    assert "end_pts" in source
    indices = [row["frame_idx"] for row in candidates]
    assert indices == [0, 2, 4, 6, 7, 9]
    assert [row["timestamp_us"] for row in candidates] == [times_ms[i] * 1000 for i in indices]
    assert [row["shot_id"] for row in candidates] == [0, 0, 0, 1, 1, 1]
    assert fake_decord[0].calls == []
    images = frames[1:3]
    assert fake_decord[0].calls == [[2, 4]]
    assert [int(image[0, 0, 0]) for image in images] == [2, 4]
    assert all(image.shape == (3, 16, 16) for image in images)
    assert int(frames[-1][0, 0, 0]) == 9
    assert [int(image[0, 0, 0]) for image in frames[[3, 0]]] == [6, 0]
    assert frames[0:0] == []
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sampling_skips_missed_targets_without_duplicating_a_frame(tmp_path):
    path = tmp_path / "gaps.mp4"
    write_video(path, [0, 2500, 2600, 3000, 4500])
    _, candidates, images, _ = service.read_candidates(path, [(0, 4)])
    assert [row["frame_idx"] for row in candidates] == [0, 1, 3, 4]
    assert len(images) == len(candidates)


@pytest.mark.parametrize("shots", [[], [(-1, 0)], [(1, 0)], [(0, 1), (1, 2)], [(0, 5)]])
def test_invalid_or_out_of_bounds_shots_are_rejected(tmp_path, shots):
    path = tmp_path / "video.mp4"
    write_video(path, [0, 40, 80])
    with pytest.raises(ValueError, match="shot|Shot"):
        service.read_candidates(path, shots)


@pytest.mark.parametrize("timestamps", [[None], [0, 0], [40, 0]])
def test_reader_rejects_missing_or_duplicate_timestamps(tmp_path, monkeypatch, timestamps):
    import av

    path = tmp_path / "video.mp4"
    path.touch()
    stream = SimpleNamespace(
        index=0, average_rate=Fraction(25), time_base=Fraction(1, 1000),
        start_time=0, duration=None,
    )
    @contextmanager
    def open_video(*_):
        yield SimpleNamespace(
            streams=SimpleNamespace(video=[stream]),
            demux=lambda *_: (SimpleNamespace(pts=pts, duration=40, size=1) for pts in timestamps),
        )

    monkeypatch.setattr(av, "open", open_video)
    if timestamps == [40, 0]:
        # B-frame packet decode order is allowed; presentation order comes from PTS.
        _, timeline = service._read_source(path)
        assert timeline == [0, 40000, 80000]
    else:
        with pytest.raises(ValueError, match="presentation timestamp"):
            service.read_candidates(path, [(10, 10)])


def test_metadata_index_never_decodes_pixels(tmp_path, monkeypatch):
    import av

    path = tmp_path / "video.mp4"
    write_video(path, [0, 40, 80])
    original_open = av.open

    @contextmanager
    def open_video(*args, **kwargs):
        with original_open(*args, **kwargs) as container:
            def forbidden_decode(*_, **__):
                pytest.fail("Metadata indexing must not decode video pixels")
            yield SimpleNamespace(
                streams=container.streams, demux=container.demux, decode=forbidden_decode
            )

    monkeypatch.setattr(av, "open", open_video)
    service.read_candidates(path, [(0, 2)])


def test_decord_index_mismatch_is_rejected(tmp_path, monkeypatch):
    import decord
    import numpy as np

    path = tmp_path / "video.mp4"
    write_video(path, [0, 40, 80])
    reader_type = decord.VideoReader

    class WrongCount(reader_type):
        def __len__(self):
            return super().__len__() - 1

    monkeypatch.setattr(decord, "VideoReader", WrongCount)
    with pytest.raises(ValueError, match="frame count"):
        service.read_candidates(path, [(0, 1)])

    class WrongTimestamps(reader_type):
        def get_frame_timestamp(self, indices):
            return super().get_frame_timestamp(indices) + np.float32(0.01)

    monkeypatch.setattr(decord, "VideoReader", WrongTimestamps)
    with pytest.raises(ValueError, match="frame timestamps"):
        service.read_candidates(path, [(0, 1)])


def test_cuda_initialization_failure_does_not_fall_back_to_cpu(tmp_path, monkeypatch):
    import decord

    path = tmp_path / "video.mp4"
    write_video(path, [0, 40, 80])

    def unavailable(*_, **__):
        raise RuntimeError("CUDA not enabled")

    monkeypatch.setattr(decord, "VideoReader", unavailable)
    with pytest.raises(RuntimeError, match="CUDA not enabled"):
        service.read_candidates(path, [(0, 2)])


def test_cached_frames_are_reused_after_decoder_reaches_eof(tmp_path, fake_decord, monkeypatch):
    path = tmp_path / 'cache.mp4'
    write_video(path, [0, 1100, 2200, 3300])
    _, _, frames, _ = service.read_candidates(path, [(0, 3)])
    original = frames[:]
    reader = fake_decord[0]
    assert reader.calls == [[0, 1, 2, 3]]

    def fail(*_):
        raise RuntimeError('EOF when decoding pixels twice')

    monkeypatch.setattr(reader, 'get_batch', fail)
    assert frames[0] is original[0]
    assert frames[-1] is original[-1]
    assert frames[[3, 1, 3]] == [original[3], original[1], original[3]]


def test_cache_limit_rejects_oversized_candidates(tmp_path, monkeypatch):
    from src import config

    path = tmp_path / 'cache.mp4'
    write_video(path, [0, 1100, 2200])
    _, _, frames, _ = service.read_candidates(path, [(0, 2)])
    monkeypatch.setattr(config, 'MAX_FRAME_CACHE_BYTES', 3 * 16 * 16 * 3 - 1)
    with pytest.raises(MemoryError, match='Candidate RGB cache'):
        frames[:1]
    assert frames.cache == {}
