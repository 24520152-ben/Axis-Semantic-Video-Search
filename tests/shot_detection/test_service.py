"""Check GPU decoder wiring and parity with OmniShotCut's inference logic."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.shot_detection import service


def test_window_preprocessing_matches_original_transform():
    from omnishotcut.engine import video_transform

    frames = torch.randint(0, 256, (7, 3, 4, 3), dtype=torch.uint8)
    expected = video_transform(
        np.concatenate((frames.numpy(), np.zeros((3, 3, 4, 3), dtype=np.uint8)))
    ).unsqueeze(0)
    actual = service._prepare_window(frames, 10)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device == frames.device


@pytest.mark.parametrize("frame_count,overlap", [(7, 2), (20, 2), (21, 3), (35, 0)])
def test_window_ranges_match_original_assembly(monkeypatch, frame_count, overlap):
    from omnishotcut import engine

    original_to = torch.Tensor.to

    def keep_on_cpu(tensor, *args, **kwargs):
        if args == ("cuda",) and not kwargs:
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", keep_on_cpu)
    captured = []

    def model(batch):
        captured.append(batch.clone())
        class_logits = torch.tensor(
            [[[4.0, 0.0, -1.0], [4.0, 0.0, -1.0], [4.0, 0.0, -1.0]]]
        )
        range_logits = torch.full((1, 3, 12), -10.0)
        range_logits[0, torch.arange(3), torch.tensor([3, 7, 10])] = 10
        return {
            "intra_clip_logits": class_logits,
            "inter_clip_logits": class_logits,
            "pred_shot_logits": range_logits,
        }

    frames = torch.randint(0, 256, (frame_count, 3, 4, 3), dtype=torch.uint8)
    args = SimpleNamespace(max_process_window_length=10)
    expected = engine._windows_to_ranges(frames.numpy(), model, args, overlap)
    original_batches = captured[:]
    captured.clear()
    actual = service._infer_ranges(frames, model, args, overlap)
    assert actual == expected
    assert len(captured) == len(original_batches)
    for actual_batch, expected_batch in zip(captured, original_batches, strict=True):
        torch.testing.assert_close(actual_batch, expected_batch, rtol=0, atol=0)


def test_decode_uses_gpu_reader_and_removes_padding(monkeypatch):
    class FakeGPU(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    reader = Mock(return_value=("video_node", "index_node"))
    resize = Mock(return_value="resized_node")
    pipe = Mock()
    pipe.epoch_size.return_value = 2
    indices = [np.arange(64), np.concatenate((np.arange(64, 67), np.full(61, -1)))]
    batches = []
    for idx in indices:
        pixels = torch.zeros((64, 2, 3, 3), dtype=torch.uint8).as_subclass(FakeGPU)
        pixels[:, :, :, :] = (
            torch.tensor(idx).clamp_min(0).to(torch.uint8)[:, None, None, None]
        )
        metadata = SimpleNamespace(
            as_cpu=lambda idx=idx: SimpleNamespace(as_array=lambda: idx[None])
        )
        batches.append(([pixels], metadata))
    pipe.run.side_effect = batches

    def pipeline_def(**kwargs):
        def decorator(func):
            def factory():
                assert func() == ("resized_node", "index_node")
                return pipe

            return factory

        return decorator

    fake_dali = SimpleNamespace(
        fn=SimpleNamespace(
            experimental=SimpleNamespace(readers=SimpleNamespace(video=reader)),
            resize=resize,
        ),
        pipeline_def=pipeline_def,
    )
    monkeypatch.setitem(sys.modules, "nvidia.dali", fake_dali)
    monkeypatch.setattr(torch, "from_dlpack", lambda tensor: tensor)
    result = service._decode_video("example.mp4", 3, 2)
    assert result.shape == (67, 2, 3, 3)
    assert result[:, 0, 0, 0].tolist() == list(range(67))
    assert reader.call_args.kwargs["device"] == "gpu"
    assert reader.call_args.kwargs["pad_mode"] == "constant"
    resize.assert_called_once_with("video_node", device="gpu", resize_x=3, resize_y=2)


def mock_detector(monkeypatch, ranges, labels):
    import omnishotcut

    wrapper = SimpleNamespace(
        _model=object(),
        _model_args=SimpleNamespace(process_width=128, process_height=96),
        inference=Mock(
            side_effect=AssertionError("Public API would copy video to CPU")
        ),
    )
    loader = Mock(return_value=wrapper)
    decoder = Mock(return_value=object())
    infer = Mock(return_value=(ranges, labels, [0] * len(ranges)))
    monkeypatch.setattr(omnishotcut, "load", loader)
    monkeypatch.setattr(service, "_decode_video", decoder)
    monkeypatch.setattr(service, "_infer_ranges", infer)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    return wrapper, loader, decoder, infer


def test_detector_uses_tensor_adapter_and_filters_clean_shots(monkeypatch):
    from omnishotcut.label_correspondence import unique_intra_label_mapping

    general = unique_intra_label_mapping["general"]
    wrapper, loader, decoder, infer = mock_detector(
        monkeypatch, [(0, 50), (50, 60), (60, 70)], [general, general + 1, general]
    )
    assert service.detect_shots("video.mp4") == [(0, 49), (60, 69)]
    decoder.assert_called_once_with("video.mp4", 128, 96)
    infer.assert_called_once_with(
        decoder.return_value, wrapper._model, wrapper._model_args
    )
    wrapper.inference.assert_not_called()
    loader.assert_called_once()


def test_detector_rejects_video_without_clean_shots(monkeypatch):
    mock_detector(monkeypatch, [], [])
    with pytest.raises(ValueError, match="No clean shots"):
        service.detect_shots("empty.mp4")
