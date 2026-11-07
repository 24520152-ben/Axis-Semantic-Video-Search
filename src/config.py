"""Pipeline defaults. Edit settings here."""

from pathlib import Path

WORKSPACE = Path("/workspace")
VOLUME_NAME = "aic-abr"
QUERY_GPU = ["L4", "A10G"]
PIPELINE_GPU = ["L4", "A10G"]
CPU_CORES = 4
MEMORY_MIB = 16384
MAX_CONTAINERS = 5
SCALEDOWN_WINDOW = 2
BATCH_SIZE = 4
MAX_FRAME_CACHE_BYTES = 8 * 1024**3
SIMILARITY_THRESHOLD = 0.95
MAX_GAP_SECONDS = 5.0
MAX_PIXELS = 512 * 32 * 32
JPEG_QUALITY = 95
MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
EMBEDDING_DIM = 2048
DATASET_VERSION = "qwen2b-bf16-v1"
MEDIA_INFO_DIR = Path("data/media-info-aic25-b1/media-info")
