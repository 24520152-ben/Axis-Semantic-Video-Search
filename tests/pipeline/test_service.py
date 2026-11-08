"""Exercise the pipeline with a real video and mocked model inference."""

import json
from unittest.mock import Mock

import av
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from src.pipeline import service


def setup_adapters(tmp_path, monkeypatch, size=16):
    """Create a six-second video; substitute only shot detection and model inference."""
    path = tmp_path / "data/video.mp4"
    path.parent.mkdir()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = size, size, "yuv420p"
        for _ in range(150):
            frame = av.VideoFrame.from_image(Image.new("RGB", (size, size), "red"))
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    metadata = tmp_path / "data/media-info-aic25-b1/media-info/video.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"watch_url": "https://youtube.com/watch?v=example"})
    )

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
    assert len(embed.call_args.args[1]) == 6
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
    assert (
        table.num_rows
        == result["kept"]
        == len(list((root / "images/video").glob("*.jpg")))
    )
    for uri in table["image_uri"].to_pylist():
        with Image.open(root / uri) as image:
            image.verify()
        with Image.open(root / uri) as image:
            red, green, blue = image.getpixel((0, 0))
            assert red > 240 and green < 10 and blue < 10
    assert not list(root.rglob("*.json"))


def test_missing_media_info_fails_before_embedding(tmp_path, monkeypatch):
    embed = setup_adapters(tmp_path, monkeypatch)
    (tmp_path / "data/media-info-aic25-b1/media-info/video.json").unlink()
    with pytest.raises(FileNotFoundError):
        service.run_pipeline("data/video.mp4", workspace=tmp_path)
    embed.assert_not_called()


def test_export_uses_cached_pixels_after_capture_is_closed(tmp_path, monkeypatch):
    import cv2

    embed = setup_adapters(tmp_path, monkeypatch)

    def encode(model, frames, batch_size):
        assert batch_size == 4
        assert len(frames) == 6
        assert all(frame.device.type == "cpu" for frame in frames)
        monkeypatch.setattr(
            cv2, "VideoCapture", Mock(side_effect=AssertionError("Repeated decode"))
        )
        return embed.return_value

    embed.side_effect = encode
    result = service.run_pipeline("data/video.mp4", workspace=tmp_path)
    assert result["kept"] == 2
    table = pq.read_table(result["output"]["records_path"])
    assert table["frame_idx"].to_pylist() == [0, 125]
    assert table["timestamp_us"].to_pylist() == [0, 5_000_000]
