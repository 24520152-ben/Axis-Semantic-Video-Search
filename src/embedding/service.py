"""Use Qwen's Sentence Transformers integration directly."""

import torch

from src.config import BATCH_SIZE, MAX_PIXELS, MODEL_ID, MODEL_REVISION


def load_model(device="cuda"):
    """Load the pinned Qwen checkpoint through Sentence Transformers.

    Args:
        device: Inference device; use cpu for local checkpoint validation.

    Returns:
        Embedding model using the checkpoint's saved configuration.

    Raises:
        RuntimeError: The loaded image processor does not use torchvision.
    """
    from sentence_transformers import SentenceTransformer
    from transformers.utils.versions import require_version

    require_version(
        "transformers>=5.0,<6", "Qwen embedding requires the fixed checkpoint loader."
    )

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
        local_files_only=True,
        processor_kwargs={
            "min_pixels": 4096,
            "max_pixels": MAX_PIXELS,
        },
    )
    if model[0].processor.image_processor.backend != "torchvision":
        raise RuntimeError(
            "Qwen requires the torchvision image processor for GPU tensors"
        )
    return model


def embed_candidates(model, frames, batch_size=BATCH_SIZE):
    """Encode candidates with Qwen's image processor and pooling.

    Args:
        model: Loaded Sentence Transformer.
        frames: Sliceable sequence of RGB uint8 CHW CPU tensors.
        batch_size: Number of images per inference batch; defaults to the
            configured BATCH_SIZE of 16. Only the current batch moves to the
            model's device, leaving the CPU frame cache available for export.

    Returns:
        Native BF16 embeddings on CPU with shape (len(frames), 2048), in
        candidate order.

    Raises:
        ValueError: batch_size is not positive or frames is empty.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not len(frames):
        raise ValueError("No candidate frames to embed")
    vectors = []
    for start in range(0, len(frames), batch_size):
        # Keep the RGB cache in RAM; preprocess only this batch on the model device.
        batch = [frame.to(model.device) for frame in frames[start : start + batch_size]]
        vectors.append(
            model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).cpu()
        )
        del batch
    return torch.cat(vectors)
