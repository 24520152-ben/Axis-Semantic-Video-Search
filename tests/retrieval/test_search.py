"""Exercise retrieval over HTTP without Qwen, Modal workers or Zilliz calls."""

import threading
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.retrieval.router import router
from src.retrieval.search import OUTPUT_FIELDS, search_frames

VECTOR = b"\x80\x3f" + b"\x00\x00" * 2047


def hit(video="L26_V035", index=125, score=0.87):
    """Return a typical SDK hit with aligned source fields."""
    return {
        "id": f"sha:0:{index}",
        "distance": score,
        "entity": {
            "frame_id": f"sha:0:{index}",
            "video_id": video,
            "frame_idx": index,
            "timestamp_us": index * 40000,
            "image_uri": f"images/{video}/{index}.jpg",
            "watch_url": "https://youtube.com/watch?v=example",
        },
    }


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router)

    @app.get("/frames/{video_id}/{frame_idx}.jpg", name="get_frame")
    def preview(video_id: str, frame_idx: int):
        return {}

    app.state.client = Mock()
    app.state.client.search.return_value = [[hit(), hit("L21_V013", 600, 0.81)]]
    app.state.collection = "frames_bf16"
    app.state.embed_query = AsyncMock(return_value=VECTOR)
    return app


def test_search_preserves_vector_bits_hit_order_and_matching_metadata(api):
    with TestClient(api) as client:
        response = client.post(
            "/search", json={"query": "  người phụ nữ áo đỏ  ", "top_k": 2}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "người phụ nữ áo đỏ"
    api.state.embed_query.assert_awaited_once_with("người phụ nữ áo đỏ")
    api.state.client.search.assert_called_once_with(
        collection_name="frames_bf16",
        anns_field="embedding",
        data=[VECTOR],
        limit=2,
        search_params={"metric_type": "COSINE"},
        output_fields=OUTPUT_FIELDS,
        timeout=30,
    )
    results = data["results"]
    assert [row["score"] for row in results] == [0.87, 0.81]
    assert [row["frame_idx"] for row in results] == [125, 600]
    assert [row["timestamp_us"] for row in results] == [5_000_000, 24_000_000]
    assert results[0]["image_uri"] == "images/L26_V035/125.jpg"
    assert results[0]["image_url"] == "http://testserver/frames/L26_V035/125.jpg"
    assert results[1]["image_url"] == "http://testserver/frames/L21_V013/600.jpg"
    assert all("embedding" not in row for row in results)
    assert results[0]["watch_url"] == "https://youtube.com/watch?v=example"


def test_video_filter_is_bound_as_a_parameter():
    client = Mock(search=Mock(return_value=[[]]))
    value = 'L26" OR video_id != "'
    assert search_frames(client, "frames", VECTOR, video_id=value) == []
    kwargs = client.search.call_args.kwargs
    assert kwargs["filter"] == "video_id == {video_id}"
    assert kwargs["filter_params"] == {"video_id": value}
    assert value not in kwargs["filter"]


def test_native_sdk_serializes_bf16_query_without_float32_conversion():
    from pymilvus import DataType
    from pymilvus.client.prepare import Prepare
    from pymilvus.grpc_gen import common_pb2

    request = Prepare.search_requests_with_expr(
        collection_name="frames",
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=20,
        data=[VECTOR],
        schema={"fields": [{"name": "embedding", "type": DataType.BFLOAT16_VECTOR}]},
    )
    placeholder = common_pb2.PlaceholderGroup.FromString(
        request.placeholder_group
    ).placeholders[0]
    assert placeholder.type == common_pb2.PlaceholderType.BFloat16Vector
    assert list(placeholder.values) == [VECTOR]
    assert request.nq == 1


@pytest.mark.parametrize("vector", [b"", VECTOR[:-1], VECTOR * 2, list(VECTOR)])
def test_invalid_remote_vector_is_rejected_before_search(vector):
    client = Mock()
    with pytest.raises(ValueError, match="BF16"):
        search_frames(client, "frames", vector)
    client.search.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "a" * 4001},
        {"query": "xe", "top_k": 0},
        {"query": "xe", "top_k": 101},
        {"query": "xe", "video_id": "../../private"},
        {"query": "xe", "video_id": ""},
    ],
)
def test_invalid_request_never_calls_model_or_database(api, body):
    with TestClient(api) as client:
        assert client.post("/search", json=body).status_code == 422
    api.state.embed_query.assert_not_awaited()
    api.state.client.search.assert_not_called()


def test_empty_search_returns_no_results(api):
    api.state.client.search.return_value = [[]]
    with TestClient(api) as client:
        assert client.post("/search", json={"query": "không có"}).json() == {
            "query": "không có",
            "results": [],
        }


def test_primary_key_comes_from_sdk_hit_id_when_absent_from_entity(api):
    for row in api.state.client.search.return_value[0]:
        row["entity"].pop("frame_id")
    with TestClient(api) as client:
        response = client.post("/search", json={"query": "xe đạp"})
    assert response.status_code == 200
    assert [row["frame_id"] for row in response.json()["results"]] == [
        "sha:0:125",
        "sha:0:600",
    ]


def test_native_milvus_primary_field_name_is_used_instead_of_generic_id(api):
    for row in api.state.client.search.return_value[0]:
        row["frame_id"] = row.pop("id")
        row["entity"].pop("frame_id")
    with TestClient(api) as client:
        response = client.post("/search", json={"query": "người mặc áo đỏ"})
    assert response.status_code == 200
    assert [row["frame_id"] for row in response.json()["results"]] == [
        "sha:0:125",
        "sha:0:600",
    ]


@pytest.mark.parametrize("failed", ["embedding", "search", "metadata"])
def test_upstream_failures_do_not_expose_credentials(api, failed):
    if failed == "embedding":
        api.state.embed_query.side_effect = RuntimeError("secret-token")
    elif failed == "search":
        api.state.client.search.side_effect = RuntimeError("secret-token")
    else:
        api.state.client.search.return_value[0][0]["entity"][
            "image_uri"
        ] = "../../secret-token"
    with TestClient(api) as client:
        response = client.post("/search", json={"query": "xe đạp"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Search service unavailable"}
    assert "secret-token" not in response.text


def test_sync_database_search_runs_outside_the_event_loop(api):
    threads = {}

    async def embed(query):
        threads["event_loop"] = threading.get_ident()
        return VECTOR

    def query(**kwargs):
        threads["database"] = threading.get_ident()
        return [[]]

    api.state.embed_query = embed
    api.state.client.search.side_effect = query
    with TestClient(api) as client:
        assert client.post("/search", json={"query": "xe đạp"}).status_code == 200
    assert threads["event_loop"] != threads["database"]
