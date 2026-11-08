"""Check API resources and local-to-Modal dispatch without cloud inference."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config
from src.retrieval import runtime


@pytest.fixture
def services(monkeypatch):
    monkeypatch.setattr(
        runtime, "load_settings", lambda: ("https://example.test", "token", "frames")
    )
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(runtime, "MilvusClient", factory)
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    return client, factory


def test_lifespan_reuses_connection_and_closes_it_on_exit(services):
    client, factory = services
    app = FastAPI(lifespan=runtime.lifespan)
    with TestClient(app):
        assert app.state.client is client
        assert app.state.collection == "frames"
        assert (
            app.state.dataset_root
            == config.WORKSPACE / "data/retrieval" / config.DATASET_VERSION
        )
        assert app.state.embed_query is runtime.embed_query_remote
        client.close.assert_not_called()
    factory.assert_called_once_with(
        uri="https://example.test", token="token", timeout=15
    )
    client.close.assert_called_once()


def test_lifespan_preserves_modal_callback_and_local_preview_root(
    services, monkeypatch, tmp_path
):
    callback = AsyncMock()
    app = FastAPI(lifespan=runtime.lifespan)
    app.state.embed_query = callback
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))
    with TestClient(app):
        assert app.state.embed_query is callback
        assert app.state.dataset_root == tmp_path
    callback.assert_not_awaited()


def test_client_closes_when_application_raises(services):
    client, _ = services
    app = FastAPI(lifespan=runtime.lifespan)

    async def fail():
        async with runtime.lifespan(app):
            raise RuntimeError("Application failed")

    with pytest.raises(RuntimeError, match="Application failed"):
        asyncio.run(fail())
    client.close.assert_called_once()


def test_local_query_looks_up_remote_worker_without_loading_model(monkeypatch):
    import modal

    invoke = AsyncMock(return_value=b"bf16-vector")
    worker = SimpleNamespace(
        embed_query=SimpleNamespace(remote=SimpleNamespace(aio=invoke))
    )
    cls = Mock(return_value=worker)
    lookup = Mock(return_value=cls)
    monkeypatch.setattr(modal.Cls, "from_name", lookup)
    assert asyncio.run(runtime.embed_query_remote("người mặc áo đỏ")) == b"bf16-vector"
    lookup.assert_called_once_with("aic-retrieval", "QwenEmbedder")
    invoke.assert_awaited_once_with("người mặc áo đỏ")


def test_main_health_does_not_invoke_qwen(services):
    from src.main import app

    callback = AsyncMock()
    app.state.embed_query = callback
    try:
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            paths = client.get("/openapi.json").json()["paths"]
            assert {"/search", "/health", "/frames/{video_id}/{frame_idx}.jpg"} <= set(
                paths
            )
        callback.assert_not_awaited()
    finally:
        del app.state.embed_query
