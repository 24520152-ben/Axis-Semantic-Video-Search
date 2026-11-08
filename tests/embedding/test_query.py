"""Verify query payloads without loading Qwen or creating GPU containers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.embedding.query import encode_query


def test_query_keeps_native_bf16_bits_and_checkpoint_prompt():
    vector = torch.nn.functional.normalize(
        torch.arange(1, 2049, dtype=torch.bfloat16), dim=0
    )
    model = SimpleNamespace(encode=Mock(return_value=vector.unsqueeze(0)))
    payload = encode_query(model, "  người phụ nữ mặc áo đỏ  ")

    model.encode.assert_called_once_with(
        ["người phụ nữ mặc áo đỏ"], convert_to_tensor=True, show_progress_bar=False
    )
    assert len(payload) == 4096
    assert payload == vector.view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("query", ["", " \n ", None])
def test_empty_query_never_runs_model(query):
    model = Mock()
    with pytest.raises(ValueError, match="empty"):
        encode_query(model, query)
    model.encode.assert_not_called()


@pytest.mark.parametrize("invalid", ["dtype", "dimension", "zero", "nan"])
def test_invalid_query_embedding_is_rejected(invalid):
    vector = torch.ones((1, 2048), dtype=torch.bfloat16)
    if invalid == "dtype":
        vector = vector.float()
    elif invalid == "dimension":
        vector = vector[:, :1024]
    elif invalid == "zero":
        vector.zero_()
    else:
        vector[0, 0] = float("nan")
    with pytest.raises(ValueError):
        encode_query(
            SimpleNamespace(encode=Mock(return_value=vector)), "người mặc áo đỏ"
        )
