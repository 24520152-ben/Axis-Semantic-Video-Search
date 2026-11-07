"""Manage the API's Zilliz connection and remote query embedder."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from pymilvus import MilvusClient
from starlette.concurrency import run_in_threadpool

from src import config
from src.retrieval.config import load_settings


async def embed_query_remote(query):
    """Embed a query using the deployed Modal GPU worker.

    Args:
        query: Search text, including Vietnamese queries.

    Returns:
        The worker's native BF16 vector bytes.
    """
    import modal

    worker = modal.Cls.from_name("aic-retrieval", "QwenEmbedder")()
    return await worker.embed_query.remote.aio(query)


@asynccontextmanager
async def lifespan(app):
    """Open one Zilliz client per API process and close it on shutdown.

    Args:
        app: FastAPI application receiving the shared services.

    Yields:
        Control to the application until shutdown.
    """
    uri, token, collection = load_settings()
    client = await run_in_threadpool(MilvusClient, uri=uri, token=token, timeout=15)
    try:
        app.state.client = client
        app.state.collection = collection
        default_root = config.WORKSPACE / "data" / "retrieval" / config.DATASET_VERSION
        app.state.dataset_root = Path(os.environ.get("DATASET_ROOT", str(default_root)))
        if not hasattr(app.state, "embed_query"):
            app.state.embed_query = embed_query_remote
        yield
    finally:
        await run_in_threadpool(client.close)
