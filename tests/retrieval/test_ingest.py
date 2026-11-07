"""Exercise ingestion with real Parquet files and mocked cloud clients."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("pymilvus")
pytest.importorskip("dotenv")

from src.retrieval import ingest as cli  # noqa: E402
from src.retrieval import service  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Keep tests independent of the user's real connection settings."""
    for name in ("ZILLIZ_URI", "ZILLIZ_TOKEN", "ZILLIZ_COLLECTION"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def client():
    fields = [
        {"name": name, "type": dtype, "is_primary": name == "frame_id", "params": {}}
        for name, dtype in service.FIELDS.items()
    ]
    fields[-1]["params"] = {"dim": "2048"}
    result = Mock()
    result.describe_collection.return_value = {"auto_id": False, "fields": fields}
    result.query.return_value = [{"count(*)": 5}]
    return result


def table(video_id="a", count=3, *, invalid=""):
    """Build a seven-column export with native BF16 payloads."""
    vectors = torch.eye(max(count, 1), 2048, dtype=torch.bfloat16)[:count]
    if invalid == "zero":
        vectors[0] = 0
    elif invalid == "nan":
        vectors[0, 0] = float("nan")
    data = {
        "frame_id": [f"sha:0:{video_id}:{index}" for index in range(count)],
        "video_id": ["other" if invalid == "video" else video_id] * count,
        "frame_idx": pa.array(range(count), type=pa.int64()),
        "timestamp_us": pa.array([i * 1_000_000 for i in range(count)], type=pa.int64()),
        "image_uri": [f"images/{video_id}/{i}.jpg" for i in range(count)],
        "watch_url": ["https://youtube.com/watch?v=example"] * count,
        "embedding": pa.array(
            [vector.view(torch.uint8).numpy().tobytes() for vector in vectors],
            type=pa.binary(4096),
        ),
    }
    if invalid == "null":
        data["image_uri"][0] = None
    elif invalid == "empty":
        data["watch_url"][0] = ""
    return pa.table(data)


def write_shard(tmp_path, video_id="a", count=3, *, invalid=""):
    """Write a temporary export shard without touching application storage."""
    path = tmp_path / f"{video_id}.parquet"
    pq.write_table(table(video_id, count, invalid=invalid), path)
    return video_id, path


def test_local_discovery_filters_video_and_ignores_other_files(tmp_path):
    write_shard(tmp_path, "b")
    write_shard(tmp_path, "a")
    (tmp_path / "ignored.json").write_text("{}")
    assert [video for video, _ in service.iter_record_files(tmp_path)] == ["a", "b"]
    assert list(service.iter_record_files(tmp_path, "b")) == [("b", tmp_path / "b.parquet")]
    with pytest.raises(FileNotFoundError):
        list(service.iter_record_files(tmp_path, "missing"))


def test_volume_discovery_streams_only_requested_parquet(tmp_path, monkeypatch):
    _, path = write_shard(tmp_path, "a")
    payload = path.read_bytes()
    volume = Mock()
    volume.iterdir.return_value = [
        SimpleNamespace(path="data/records/b.parquet"),
        SimpleNamespace(path="data/records/a.parquet"),
        SimpleNamespace(path="data/records/note.txt"),
    ]
    volume.read_file.return_value = iter([payload[:20], payload[20:]])
    factory = Mock(return_value=volume)
    monkeypatch.setattr(modal.Volume, "from_name", factory)
    shards = list(service.iter_record_files(video_id="a"))
    assert len(shards) == 1 and shards[0][0] == "a"
    assert pq.read_table(shards[0][1]).equals(table())
    factory.assert_called_once_with(service.config.VOLUME_NAME)
    volume.iterdir.assert_called_once_with(service.RECORDS_DIR.as_posix(), recursive=False)
    volume.read_file.assert_called_once_with("data/records/a.parquet")


def test_ingest_preserves_alignment_batches_and_verifies_visible_count(tmp_path, client):
    shards = [write_shard(tmp_path, "a", 3), write_shard(tmp_path, "b", 2)]
    report = service.ingest(client, "frames", shards, batch_size=2)
    assert report == {"upserted": 5, "collection_rows": 5}
    calls = client.upsert.call_args_list
    assert [len(call.kwargs["data"]) for call in calls] == [2, 1, 2]
    rows = [row for call in calls for row in call.kwargs["data"]]
    assert rows == table("a", 3).to_pylist() + table("b", 2).to_pylist()
    assert all(call.kwargs["collection_name"] == "frames" for call in calls)
    client.load_collection.assert_called_once_with(collection_name="frames")
    client.query.assert_called_once_with(
        collection_name="frames", output_fields=["count(*)"], consistency_level="Strong"
    )
    assert [call[0] for call in client.method_calls][-2:] == ["load_collection", "query"]


@pytest.mark.parametrize("invalid", ["fields", "auto_id", "primary", "dimension"])
def test_collection_mismatch_is_rejected_before_upsert(tmp_path, client, invalid):
    schema = deepcopy(client.describe_collection.return_value)
    if invalid == "fields":
        schema["fields"].pop(1)
    elif invalid == "auto_id":
        schema["auto_id"] = True
    elif invalid == "primary":
        schema["fields"][0]["is_primary"] = False
    else:
        schema["fields"][-1]["params"]["dim"] = 512
    client.describe_collection.return_value = schema
    with pytest.raises(ValueError):
        service.ingest(client, "frames", [write_shard(tmp_path)])
    client.upsert.assert_not_called()


@pytest.mark.parametrize("invalid", ["zero", "nan", "null", "empty", "video"])
def test_bad_rows_are_rejected_before_sending_batch(tmp_path, client, invalid):
    with pytest.raises(ValueError):
        service.ingest(client, "frames", [write_shard(tmp_path, invalid=invalid)])
    client.upsert.assert_not_called()


def test_bad_parquet_schema_is_rejected(tmp_path, client):
    path = tmp_path / "a.parquet"
    pq.write_table(table().drop(["watch_url"]), path)
    with pytest.raises(ValueError, match="seven fields"):
        service.ingest(client, "frames", [("a", path)])
    client.upsert.assert_not_called()


def test_fp32_collection_is_rejected_with_bf16_setup_instructions(tmp_path, client):
    schema = deepcopy(client.describe_collection.return_value)
    schema["fields"][-1]["type"] = service.DataType.FLOAT_VECTOR
    client.describe_collection.return_value = schema
    with pytest.raises(ValueError, match="BFLOAT16_VECTOR.*2048.*ZILLIZ_COLLECTION"):
        service.ingest(client, "frames", [write_shard(tmp_path)])
    client.upsert.assert_not_called()


def test_old_fp32_parquet_is_rejected_before_upsert(tmp_path, client):
    path = tmp_path / "a.parquet"
    data = table().to_pydict()
    data["embedding"] = pa.array([[1.0] + [0.0] * 2047] * 3, type=pa.list_(pa.float32(), 2048))
    pq.write_table(pa.table(data), path)
    with pytest.raises(ValueError, match="embedding"):
        service.ingest(client, "frames", [("a", path)])
    client.upsert.assert_not_called()


def test_native_sdk_preserves_bf16_payload_bits_without_float32_conversion():
    from pymilvus.client import entity_helper
    from pymilvus.grpc_gen import schema_pb2

    vector = torch.nn.functional.normalize(torch.arange(1, 2049, dtype=torch.bfloat16), dim=0)
    payload = vector.view(torch.uint8).numpy().tobytes()
    field = schema_pb2.FieldData(type=service.DataType.BFLOAT16_VECTOR, field_name="embedding")
    info = {"name": "embedding", "type": service.DataType.BFLOAT16_VECTOR, "params": {"dim": 2048}}
    cache = {}
    entity_helper.pack_field_value_to_field_data(payload, field, info, cache)
    entity_helper.flush_vector_bytes(field, cache)
    assert field.vectors.dim == 2048
    assert bytes(field.vectors.bfloat16_vector) == payload
    assert not field.vectors.float_vector.data


def test_checkpoint_bf16_normalization_roundoff_is_preserved(tmp_path, client):
    vector = torch.nn.functional.normalize(torch.arange(1, 2049, dtype=torch.bfloat16), dim=0)
    assert abs(vector.float().norm().item() - 1) > 1e-4
    payload = vector.view(torch.uint8).numpy().tobytes()
    data = table(count=1).to_pydict()
    data["embedding"] = pa.array([payload], type=pa.binary(4096))
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table(data), path)

    result = service.ingest(client, "frames", [("a", path)])

    assert result["upserted"] == 1
    assert client.upsert.call_args.kwargs["data"][0]["embedding"] == payload


def test_failed_upsert_stops_before_later_batches(tmp_path, client):
    client.upsert.side_effect = [None, RuntimeError("RPC failed")]
    with pytest.raises(RuntimeError, match="RPC failed"):
        service.ingest(client, "frames", [write_shard(tmp_path, count=5)], batch_size=2)
    assert client.upsert.call_count == 2
    client.query.assert_not_called()


def test_missing_settings_show_names_without_secret_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ZILLIZ_TOKEN=secret-placeholder\n")
    with pytest.raises(ValueError) as error:
        cli.load_settings(env)
    assert "ZILLIZ_URI" in str(error.value) and "ZILLIZ_COLLECTION" in str(error.value)
    assert "secret-placeholder" not in str(error.value)


def test_environment_overrides_dotenv_settings(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        'ZILLIZ_URI="https://file.example"\n'
        'ZILLIZ_TOKEN="file-token"\nZILLIZ_COLLECTION=" frames "\n'
    )
    monkeypatch.setenv("ZILLIZ_URI", "https://environment.example")
    assert cli.load_settings(env) == ("https://environment.example", "file-token", "frames")
