"""Exercise the pipeline with a real video and mocked model inference."""

import json
from unittest.mock import Mock

import av
import decord
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from src.pipeline import service


def setup_adapters(tmp_path, monkeypatch):
    """Create a six-second video; substitute GPU decoding and the two models."""
    path = tmp_path / "data/video.mp4"
    path.parent.mkdir()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for _ in range(150):
            frame = av.VideoFrame.from_image(Image.new("RGB", (16, 16), "red"))
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    metadata = tmp_path / "data/media-info-aic25-b1/media-info/video.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"watch_url": "https://youtube.com/watch?v=example"}))

    class Reader:
        def __init__(self, path, *, ctx):
            self.calls = []

        def __len__(self):
            return 150

        def get_frame_timestamp(self, indices):
            return np.asarray([[i / 25, (i + 1) / 25] for i in indices], dtype=np.float32)

        def get_batch(self, indices):
            self.calls.append(list(indices))
            return torch.stack([
                torch.full((16, 16, 3), index, dtype=torch.uint8) for index in indices
            ])

    monkeypatch.setattr(decord, "VideoReader", Reader)
    monkeypatch.setattr(service, "detect_shots", lambda _: [(0, 149)])
    monkeypatch.setattr(service.embedding, "load_model", lambda: object())
    embed = Mock(return_value=torch.eye(1, 2048, dtype=torch.bfloat16).repeat(6, 1))
    monkeypatch.setattr(service.embedding, "embed_candidates", embed)
    return embed


def test_pipeline_decodes_samples_and_exports_aligned_records(tmp_path, monkeypatch):
    embed = setup_adapters(tmp_path, monkeypatch)
    result = service.run_pipeline("data/video.mp4", workspace=tmp_path)
    embed.assert_called_once()
    assert embed.call_args.args[2] == 4
    assert isinstance(embed.call_args.args[1], service.video.DecordFrames)
    assert embed.call_args.args[1].reader.calls == [[0], [125]]
    assert result["candidates"] == 6 and result["kept"] == 2
    assert result["max_gap_seconds"] <= 5
    root = tmp_path / "data/retrieval" / service.DATASET_VERSION
    table = pq.read_table(root / "records/video.parquet")
    assert table["frame_idx"].to_pylist() == [0, 125]
    assert table["timestamp_us"].to_pylist() == [0, 5_000_000]
    payloads = table["embedding"].to_pylist()
    assert all(len(payload) == 2048 * 2 for payload in payloads)
    first_vector = torch.frombuffer(bytearray(payloads[0]), dtype=torch.bfloat16)
    assert torch.equal(first_vector, embed.return_value[0])
    assert table.num_rows == result["kept"] == len(list((root / "images/video").glob("*.jpg")))
    for uri in table["image_uri"].to_pylist():
        with Image.open(root / uri) as image:
            image.verify()
        with Image.open(root / uri) as image:
            assert image.getpixel((0, 0)) == (int(uri.rsplit("/", 1)[1][:-4]),) * 3
    assert not list(root.rglob("*.json"))


def test_missing_media_info_fails_before_embedding(tmp_path, monkeypatch):
    embed = setup_adapters(tmp_path, monkeypatch)
    (tmp_path / "data/media-info-aic25-b1/media-info/video.json").unlink()
    with pytest.raises(FileNotFoundError):
        service.run_pipeline("data/video.mp4", workspace=tmp_path)
    embed.assert_not_called()


def test_export_reuses_embedded_pixels_without_rewinding_decoder(tmp_path, monkeypatch):
    """Export must use the same pixels embedded before the decoder reached EOF."""
    embed = setup_adapters(tmp_path, monkeypatch)
    reader_type = decord.VideoReader
    readers = []

    class Reader(reader_type):
        def __init__(self, path, *, ctx):
            super().__init__(path, ctx=ctx)
            self.exhausted = False
            readers.append(self)

        def get_batch(self, indices):
            if self.exhausted:
                raise RuntimeError('Decord EOF while rewinding an exhausted reader')
            return super().get_batch(indices)

    monkeypatch.setattr(decord, 'VideoReader', Reader)

    def encode(model, frames, batch_size):
        assert len(readers) == 1
        for start in range(0, len(frames), batch_size):
            frames[start:start + batch_size]
        readers[0].exhausted = True
        return embed.return_value

    embed.side_effect = encode
    report = service.run_pipeline('data/video.mp4', workspace=tmp_path)
    assert report['kept'] == 2
    assert len(readers) == 1
    assert readers[0].calls == [[0, 25, 50, 75], [100, 125]]
    table = pq.read_table(report['output']['records_path'])
    assert table['frame_idx'].to_pylist() == [0, 125]
    assert table['timestamp_us'].to_pylist() == [0, 5_000_000]
