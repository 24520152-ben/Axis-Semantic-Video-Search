"""Read retained-frame Parquet shards and upsert them into an existing collection."""

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pymilvus import DataType

from src import config

FIELDS = {
    "frame_id": DataType.VARCHAR,
    "video_id": DataType.VARCHAR,
    "frame_idx": DataType.INT64,
    "timestamp_us": DataType.INT64,
    "image_uri": DataType.VARCHAR,
    "watch_url": DataType.VARCHAR,
    "embedding": DataType.BFLOAT16_VECTOR,
}
RECORDS_DIR = Path("data/retrieval") / config.DATASET_VERSION / "records"


def validate_collection(client, collection):
    """Require the seven export fields, a supplied primary key and matching dimensions."""
    schema = client.describe_collection(collection_name=collection)
    fields = {field["name"]: field for field in schema["fields"]}
    if (
        "embedding" in fields
        and fields["embedding"]["type"] != DataType.BFLOAT16_VECTOR
    ):
        raise ValueError(
            "embedding must be BFLOAT16_VECTOR with dimension 2048. "
            "Create a BF16 collection and update ZILLIZ_COLLECTION in .env."
        )
    if set(fields) != set(FIELDS) or any(
        fields[name]["type"] != dtype for name, dtype in FIELDS.items()
    ):
        raise ValueError("Collection must match the seven-field Parquet schema")
    if schema.get("auto_id") or not fields["frame_id"].get("is_primary"):
        raise ValueError("frame_id must be the primary key with Auto ID disabled")
    if int(fields["embedding"]["params"]["dim"]) != config.EMBEDDING_DIM:
        raise ValueError(f"embedding must have {config.EMBEDDING_DIM} dimensions")


def iter_record_files(records_dir=None, video_id=""):
    """Yield video IDs and Parquet sources from local files or the Modal Volume.

    Args:
        records_dir: Local records directory; None reads the configured Modal Volume.
        video_id: Optional video ID limiting the import to one shard.

    Yields:
        Video ID and a local Path or in-memory Parquet buffer.
    """
    if records_dir is not None:
        paths = sorted(Path(records_dir).glob("*.parquet"))
        volume = None
    else:
        import modal

        volume = modal.Volume.from_name(config.VOLUME_NAME)
        paths = sorted(
            Path(entry.path)
            for entry in volume.iterdir(RECORDS_DIR.as_posix(), recursive=False)
            if entry.path.endswith(".parquet")
        )
    paths = [path for path in paths if not video_id or path.stem == video_id]
    if not paths:
        raise FileNotFoundError("No matching Parquet records found")
    for path in paths:
        source = (
            io.BytesIO(b"".join(volume.read_file(path.as_posix()))) if volume else path
        )
        yield path.stem, source


def _validate_shard(parquet):
    """Reject incompatible or empty shards before sending their rows."""
    schema = parquet.schema_arrow
    if set(schema.names) != set(FIELDS) or not parquet.metadata.num_rows:
        raise ValueError("Expected a nonempty Parquet shard with exactly seven fields")
    types = {
        DataType.VARCHAR: pa.string(),
        DataType.INT64: pa.int64(),
        DataType.BFLOAT16_VECTOR: pa.binary(config.EMBEDDING_DIM * 2),
    }
    for name, dtype in FIELDS.items():
        expected = types[dtype]
        if schema.field(name).type != expected:
            raise ValueError(f"Invalid Parquet type for {name}; expected {expected}")


def _validate_batch(batch, video_id):
    """Reject missing metadata, mixed videos or invalid embeddings before upsert."""
    if any(column.null_count for column in batch.columns):
        raise ValueError("Parquet fields must not contain nulls")
    rows = batch.to_pylist()
    if any(row["video_id"] != video_id for row in rows):
        raise ValueError("video_id must match the Parquet filename")
    if any(
        not row[name] for row in rows for name in ("frame_id", "image_uri", "watch_url")
    ):
        raise ValueError("frame_id, image_uri and watch_url must not be empty")
    for row in rows:
        vector = torch.frombuffer(bytearray(row["embedding"]), dtype=torch.bfloat16)
        if not torch.isfinite(vector).all() or not torch.any(vector):
            raise ValueError("BF16 embeddings must be finite and nonzero")
    return rows


def ingest(client, collection, shards, batch_size=128):
    """Upsert retained frames and verify the collection's visible row count.

    Args:
        client: Connected MilvusClient.
        collection: Existing collection name.
        shards: Iterable of video IDs and Parquet sources.
        batch_size: Maximum rows sent per request.

    Returns:
        Upserted rows and the collection's total visible rows.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_collection(client, collection)
    total = 0
    for video_id, source in shards:
        count = 0
        with pq.ParquetFile(source) as parquet:
            _validate_shard(parquet)
            for batch in parquet.iter_batches(batch_size=batch_size):
                rows = _validate_batch(batch, video_id)
                client.upsert(collection_name=collection, data=rows)
                count += len(rows)
                print(
                    f"{video_id}: {count}/{parquet.metadata.num_rows} rows", flush=True
                )
        total += count
    client.load_collection(collection_name=collection)
    result = client.query(
        collection_name=collection,
        output_fields=["count(*)"],
        consistency_level="Strong",
    )
    return {"upserted": total, "collection_rows": result[0]["count(*)"]}
