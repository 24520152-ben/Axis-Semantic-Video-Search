"""Deploy the retrieval API on CPU and Qwen query embedding on GPU."""

import shlex

import modal

from src import config

app = modal.App("aic-retrieval")
volume = modal.Volume.from_name(config.VOLUME_NAME)
zilliz = modal.Secret.from_name(
    "aic-zilliz", required_keys=["ZILLIZ_URI", "ZILLIZ_TOKEN", "ZILLIZ_COLLECTION"]
)

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi==0.162.0",
        "uvicorn==0.54.0",
        "pymilvus==3.0.2",
        "python-dotenv==1.2.4",
        "modal==1.6.3",
    )
    .add_local_python_source("src", ignore=["**/__pycache__/**", "**/*.pyc"])
)
cache_code = (
    "from huggingface_hub import snapshot_download; "
    f"snapshot_download({config.MODEL_ID!r}, revision={config.MODEL_REVISION!r}, "
    "allow_patterns=['*.json','*.txt','*.safetensors','*.jinja','tokenizer.model'])"
)
query_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "sentence-transformers==5.4.0",
        "transformers==5.4.0",
        "pillow==13.0.0",
    )
    .run_commands(f"python -c {shlex.quote(cache_code)}")
    .add_local_python_source("src")
)


@app.cls(
    image=query_image,
    gpu=config.QUERY_GPU,
    cpu=(2, 2),
    memory=(16384, 16384),
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=300,
)
class QwenEmbedder:
    """Keep the pinned Qwen checkpoint loaded for repeated query requests."""

    @modal.enter()
    def load(self):
        """Load model weights once when this GPU container starts."""
        from src.embedding.service import load_model

        self.model = load_model()

    @modal.method()
    def embed_query(self, query: str) -> bytes:
        """Return a BF16 embedding using Qwen's saved default instruction."""
        from src.embedding.query import encode_query

        return encode_query(self.model, query)


@app.function(
    image=api_image,
    cpu=(1, 1),
    memory=(1024, 1024),
    min_containers=0,
    max_containers=2,
    scaledown_window=60,
    timeout=300,
    secrets=[zilliz],
    volumes={str(config.WORKSPACE): volume},
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    """Expose FastAPI with an internal asynchronous GPU embedding callback."""
    from src.main import app as api

    api.state.embed_query = QwenEmbedder().embed_query.remote.aio
    return api
