"""Verify Qwen checkpoint weights survive the feature-extraction loader."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sentence_transformers.base.modality import InputFormatter
from sentence_transformers.sentence_transformer.modules.normalize import Normalize
from transformers import AutoModel, AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import MAX_PIXELS, MODEL_ID, MODEL_REVISION
from src.embedding.service import embed_candidates, load_model


def test_bf16_checkpoint_output_is_returned_without_casting_or_renormalizing():
    values = torch.arange(1, 2049, dtype=torch.float32)
    raw = torch.stack([values, values.flip(0)]).to(torch.bfloat16)
    checkpoint_output = Normalize()({"sentence_embedding": raw})["sentence_embedding"]
    assert torch.max(torch.abs(checkpoint_output.float().norm(dim=1) - 1)) > 1e-4
    model = SimpleNamespace(encode=Mock(return_value=checkpoint_output))
    frames = [torch.zeros((3, 28, 28), dtype=torch.uint8) for _ in range(2)]

    vectors = embed_candidates(model, frames, batch_size=2)

    assert vectors.shape == (2, 2048) and vectors.dtype == torch.bfloat16
    assert torch.equal(vectors.view(torch.uint8), checkpoint_output.view(torch.uint8))
    model.encode.assert_called_once_with(
        frames, batch_size=2, convert_to_tensor=True, show_progress_bar=False
    )


def test_lazy_frames_are_decoded_and_encoded_once_in_bounded_batches():
    class Frames:
        def __init__(self):
            self.slices = []

        def __len__(self):
            return 5

        def __getitem__(self, index):
            assert isinstance(index, slice), "Never materialize the entire frame reader"
            self.slices.append(index)
            return [
                torch.full((3, 28, 28), i, dtype=torch.uint8)
                for i in range(*index.indices(len(self)))
            ]

    def encode(batch, **kwargs):
        return torch.tensor(
            [[frame[0, 0, 0].item()] * 2048 for frame in batch], dtype=torch.bfloat16
        )

    frames = Frames()
    model = SimpleNamespace(encode=Mock(side_effect=encode))
    vectors = embed_candidates(model, frames, batch_size=2)

    assert frames.slices == [slice(0, 2), slice(2, 4), slice(4, 6)]
    assert vectors.dtype == torch.bfloat16
    torch.testing.assert_close(vectors[:, 0], torch.arange(5, dtype=torch.bfloat16))
    assert [len(call.args[0]) for call in model.encode.call_args_list] == [2, 2, 1]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_does_not_decode_frames(batch_size):
    with pytest.raises(ValueError, match="positive"):
        embed_candidates(Mock(), [torch.empty(3, 28, 28)], batch_size=batch_size)


def test_qwen_image_preprocessing_keeps_tensor_input_without_pil_or_cpu_copy():
    frame = torch.arange(3 * 40 * 56, dtype=torch.int32).reshape(3, 40, 56).to(torch.uint8)
    formatter = InputFormatter("qwen3_vl", message_format="structured")
    modality, inputs, _ = formatter.parse_inputs([frame])
    _, messages = formatter.batch_to_message(modality, inputs)
    image = messages["message"][0][0]["content"][0]["image"]
    assert image is frame

    processor = Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=56 * 56)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Image CPU copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("Image NumPy conversion")),
    ):
        output = processor(images=[image], return_tensors="pt")

    assert processor.backend == "torchvision"
    assert output["pixel_values"].device == frame.device
    assert output["pixel_values"].dtype == torch.float32
    assert output["image_grid_thw"].shape == (1, 3)


def test_model_loader_rejects_cpu_only_image_processor(monkeypatch):
    first_module = SimpleNamespace(
        processor=SimpleNamespace(image_processor=SimpleNamespace(backend="pil"))
    )
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", Mock(return_value=[first_module])
    )

    with pytest.raises(RuntimeError, match="torchvision"):
        load_model()


def _cached_processor(**kwargs):
    """Load checkpoint preprocessing files without weights or network requests."""
    try:
        return AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True, **kwargs
        )
    except OSError:
        pytest.skip("Pinned Qwen processor files are not cached locally")


def test_model_loader_kwargs_load_actual_cached_qwen_processor(monkeypatch):
    def load_without_weights(*args, processor_kwargs, **kwargs):
        processor = _cached_processor(**processor_kwargs)
        return [SimpleNamespace(processor=processor)]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", load_without_weights)
    model = load_model(device="cpu")

    assert model[0].processor.image_processor.backend == "torchvision"
    assert model[0].processor.image_processor.size.longest_edge == MAX_PIXELS


def test_actual_qwen_chat_template_processes_tensor_without_image_cpu_copy():
    processor = _cached_processor(min_pixels=4096, max_pixels=MAX_PIXELS)
    frame = torch.arange(3 * 40 * 56, dtype=torch.int32).reshape(3, 40, 56).to(torch.uint8)
    messages = [[{"role": "user", "content": [{"type": "image", "image": frame}]}]]
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Image CPU copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("Image NumPy conversion")),
    ):
        output = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "text_kwargs": {"padding": True},
                "images_kwargs": {"input_data_format": "channels_first"},
                "common_kwargs": {"return_tensors": "pt"},
            },
        )

    assert output["pixel_values"].device == frame.device
    assert output["image_grid_thw"].shape == (1, 3)
    assert output["input_ids"].shape[0] == 1
    assert "Represent the user's input." in processor.tokenizer.decode(output["input_ids"][0])


def test_qwen_feature_extraction_loads_prefixed_checkpoint_weights(tmp_path):
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 1,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    )
    original = Qwen3VLForConditionalGeneration(config)
    # Production checkpoints use model.* keys; saving the bare base model
    # would miss the Transformers 4.57.x prefix-loading regression.
    with torch.no_grad():
        for index, parameter in enumerate(original.parameters(), 1):
            parameter.fill_(index / 100)
    original.save_pretrained(tmp_path)
    loaded, info = AutoModel.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"]
    expected = original.model.state_dict()
    actual = loaded.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
