"""Search retained keyframes using the existing BF16 collection."""

from src.config import EMBEDDING_DIM

OUTPUT_FIELDS = [
    "frame_id",
    "video_id",
    "frame_idx",
    "timestamp_us",
    "image_uri",
    "watch_url",
]


def search_frames(client, collection, vector, top_k=20, video_id=None):
    """Search one BF16 query without changing its precision or normalization.

    Args:
        client: Connected synchronous MilvusClient.
        collection: Existing keyframe collection.
        vector: Raw BF16 bytes returned by the GPU embedding worker.
        top_k: Maximum number of hits.
        video_id: Optional video restriction.

    Returns:
        Ordered Milvus hits, including their scalar metadata.
    """
    if not isinstance(vector, bytes) or len(vector) != EMBEDDING_DIM * 2:
        raise ValueError(f"Expected a BF16[{EMBEDDING_DIM}] query embedding")
    options = {}
    if video_id is not None:
        options = {
            "filter": "video_id == {video_id}",
            "filter_params": {"video_id": video_id},
        }
    results = client.search(
        collection_name=collection,
        anns_field="embedding",
        data=[vector],
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        output_fields=OUTPUT_FIELDS,
        timeout=30,
        **options,
    )
    return results[0]
