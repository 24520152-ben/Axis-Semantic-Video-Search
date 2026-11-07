"""Serialize native Qwen query embeddings for Milvus BF16 search."""

from src.config import EMBEDDING_DIM


def encode_query(model, query):
    """Embed text using the same checkpoint prompt and normalization as images.

    Args:
        model: Loaded Sentence Transformer.
        query: Nonempty text, including Vietnamese queries.

    Returns:
        Native BF16 vector as 4096 bytes, ready for Milvus search.
    """
    import torch

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty")
    vector = (
        model.encode([query.strip()], convert_to_tensor=True, show_progress_bar=False)
        .detach()
        .cpu()
    )
    if vector.dtype != torch.bfloat16 or vector.shape != (1, EMBEDDING_DIM):
        raise ValueError("Qwen must return one bfloat16[2048] embedding")
    if not torch.isfinite(vector).all() or not torch.any(vector):
        raise ValueError("Qwen returned an invalid embedding")
    return vector.contiguous().view(torch.uint8).numpy().tobytes()
