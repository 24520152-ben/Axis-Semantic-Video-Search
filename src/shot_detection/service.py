"""Decode with DALI and run OmniShotCut without copying images to CPU."""


def _decode_video(path, width, height):
    """Decode and resize every source frame on GPU.

    Args:
        path: Source video path.
        width: Model input width.
        height: Model input height.

    Returns:
        CUDA uint8 tensor with shape (frames, height, width, 3).
    """
    import numpy as np
    import torch
    from nvidia.dali import fn, pipeline_def

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def decode_pipeline():
        frames, indices = fn.experimental.readers.video(
            device="gpu",
            filenames=[str(path)],
            sequence_length=64,
            step=64,
            stride=1,
            pad_mode="constant",
            enable_frame_num="sequence",
            random_shuffle=False,
            name="video",
        )
        frames = fn.resize(frames, device="gpu", resize_x=width, resize_y=height)
        return frames, indices

    pipe = decode_pipeline()
    pipe.build()
    chunks = []
    total_frames = 0
    for _ in range(pipe.epoch_size("video")):
        frames, indices = pipe.run()
        indices = indices.as_cpu().as_array()[0]
        valid = indices[indices >= 0]
        if not np.array_equal(
            valid, np.arange(total_frames, total_frames + len(valid))
        ):
            raise ValueError("DALI did not return consecutive source frame indices")
        # Own the resized buffer before DALI reuses a pipeline output.
        chunk = torch.from_dlpack(frames[0])[: len(valid)].clone()
        if not chunk.is_cuda:
            raise RuntimeError("DALI must decode and resize on GPU")
        chunks.append(chunk)
        total_frames += len(valid)
    if not total_frames:
        raise ValueError("DALI decoded no frames")
    return torch.cat(chunks)


def _prepare_window(frames, length):
    """Pad RGB frames and apply OmniShotCut's validation normalization.

    Args:
        frames: uint8 tensor in source order, shape (T, H, W, 3).
        length: Padded inference window length.

    Returns:
        Normalized tensor with shape (1, T, 3, H, W) on the input device.
    """
    import torch
    from omnishotcut.engine import video_transform

    if len(frames) < length:
        padding = frames.new_zeros((length - len(frames), *frames.shape[1:]))
        frames = torch.cat((frames, padding))
    frames = frames.permute(0, 3, 1, 2).float().div_(255)
    mean = frames.new_tensor(video_transform.mean)[None, :, None, None]
    std = frames.new_tensor(video_transform.std)[None, :, None, None]
    return frames.sub_(mean).div_(std).unsqueeze(0)


def _infer_ranges(frames, model, model_args, overlap=10):
    """Keep OmniShotCut's inference windows and cut assembly unchanged.

    Args:
        frames: Resized RGB tensor on the model device.
        model: Loaded OmniShotCut neural network.
        model_args: Checkpoint configuration.
        overlap: Existing overlap between inference windows.

    Returns:
        Exclusive shot ranges and their intra/inter labels.
    """
    import torch
    from omnishotcut.engine import (
        assemble_ranges,
        collect_cuts_from_window,
        decode_window_segments,
    )

    length = model_args.max_process_window_length
    if not 0 <= overlap < length:
        raise ValueError("Overlap must be smaller than the inference window")
    cuts = []
    for start in range(0, len(frames), length - overlap):
        end = min(start + length, len(frames))
        with torch.inference_mode():
            outputs = model(_prepare_window(frames[start:end], length))
        predictions = [
            outputs[key].softmax(-1)[0, :, :-1].argmax(-1)
            for key in ("intra_clip_logits", "inter_clip_logits", "pred_shot_logits")
        ]
        segments = decode_window_segments(*predictions, end - start)
        valid_start = start + overlap // 2 if start else 0
        valid_end = end if end == len(frames) else end - (overlap - overlap // 2)
        cuts.extend(collect_cuts_from_window(segments, start, valid_start, valid_end))
        if end == len(frames):
            break
    return assemble_ranges(cuts, len(frames))


def detect_shots(path):
    """Run DALI preprocessing and OmniShotCut clean-shot inference on GPU.

    Args:
        path: Source video path.

    Returns:
        Inclusive source-frame ranges.
    """
    import gc

    import omnishotcut
    import torch
    from omnishotcut.label_correspondence import unique_intra_label_mapping

    wrapper = omnishotcut.load(
        "uva-cv-lab/OmniShotCut_v1.5", filename="OmniShotCut_ckpt.pth"
    )
    frames = None
    try:
        args = wrapper._model_args
        frames = _decode_video(path, args.process_width, args.process_height)
        ranges, intra_labels, _ = _infer_ranges(frames, wrapper._model, args)
        general = unique_intra_label_mapping["general"]
        shots = [
            (int(start), int(end) - 1)
            for (start, end), label in zip(ranges, intra_labels, strict=True)
            if label == general
        ]
        if not shots:
            raise ValueError("No clean shots found")
        return shots
    finally:
        del frames, wrapper
        gc.collect()
        torch.cuda.empty_cache()
