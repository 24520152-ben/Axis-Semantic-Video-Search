"""Select representatives within each shot using source timestamps."""

import bisect
import math

import torch
import torch.nn.functional as F

from src.config import MAX_GAP_SECONDS, SIMILARITY_THRESHOLD


def _group_candidates(candidates, shots, timeline_us):
    """Validate frame references and group candidate indices by shot."""
    groups, previous = [[] for _ in shots], -1
    for index, row in enumerate(candidates):
        shot_id, frame = row["shot_id"], row["frame_idx"]
        if (
            row["candidate_idx"] != index
            or not 0 <= shot_id < len(shots)
            or frame <= previous
        ):
            raise ValueError("Invalid candidate index or shot ID")
        start, end = shots[shot_id]
        if not start <= frame <= end or row["timestamp_us"] != timeline_us[frame]:
            raise ValueError("Candidate timestamp does not match source frame")
        groups[shot_id].append(index)
        previous = frame
    return groups


def _coverage_anchors(times, endpoint, max_gap):
    """Choose the first frame and latest candidates before each deadline."""
    anchors, anchor = {0}, 0
    while endpoint - times[anchor] > max_gap + 1e-8:
        following = bisect.bisect_right(times, times[anchor] + max_gap + 1e-8) - 1
        if following <= anchor:
            raise ValueError("Candidate spacing cannot satisfy coverage")
        anchors.add(following)
        anchor = following
    return anchors


def _select_shot(vectors, candidates, indices, shot, timeline_us, threshold, max_gap):
    """Keep novel frames and temporal anchors, returning indices and the largest gap."""
    start, end = shot
    if not indices or candidates[indices[0]]["frame_idx"] != start:
        raise ValueError("Shot must include its first frame")
    times = [candidates[index]["timestamp_us"] / 1e6 for index in indices]
    endpoint = timeline_us[end + 1] / 1e6
    anchors = _coverage_anchors(times, endpoint, max_gap)
    kept = []
    for position, index in enumerate(indices):
        if (
            position in anchors
            or F.cosine_similarity(vectors[kept], vectors[index], dim=1).max().item()
            < threshold
        ):
            kept.append(index)
    kept_times = [candidates[index]["timestamp_us"] / 1e6 for index in kept]
    gap = max(
        [b - a for a, b in zip(kept_times, kept_times[1:])]
        + [endpoint - kept_times[-1]]
    )
    if gap > max_gap + 1e-8:
        raise AssertionError("Temporal coverage violated")
    return kept, gap


def select_frames(
    vectors, candidates, shots, *, timeline_us, threshold=None, max_gap=None
):
    """Select novel frames with temporal coverage independently in each shot.

    Args:
        vectors: Candidate embedding tensors in candidate order; dtype is preserved.
        candidates: Sampled frame metadata with source timestamps.
        shots: Inclusive source-frame ranges.
        timeline_us: Frame timestamps and the exclusive video endpoint.
        threshold: Cosine cutoff; defaults to 0.95.
        max_gap: Maximum gap in seconds; defaults to 5.

    Returns:
        Selected indices, their count and the largest within-shot time gap.
    """
    threshold = SIMILARITY_THRESHOLD if threshold is None else threshold
    max_gap = MAX_GAP_SECONDS if max_gap is None else max_gap
    if not shots or not candidates or len(timeline_us) < 2:
        raise ValueError("Expected shots, candidates and a presentation timeline")
    if not -1 <= threshold <= 1 or not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Invalid selection settings")
    if any(b <= a for a, b in zip(timeline_us, timeline_us[1:])):
        raise ValueError("Presentation timestamps must increase")
    previous = -1
    for start, end in shots:
        if start <= previous or end < start or end >= len(timeline_us) - 1:
            raise ValueError("Shot ranges must be ordered and within the timeline")
        previous = end
    vectors = torch.as_tensor(vectors)
    if vectors.ndim != 2 or len(vectors) != len(candidates):
        raise ValueError("Vector rows must match candidates")
    if (
        not vectors.is_floating_point()
        or not torch.isfinite(vectors).all()
        or torch.any(torch.count_nonzero(vectors, dim=1) == 0)
    ):
        raise ValueError("Vectors must be finite and nonzero")
    groups = _group_candidates(candidates, shots, timeline_us)
    selected, largest_gap = [], 0.0
    for shot, indices in zip(shots, groups):
        kept, gap = _select_shot(
            vectors, candidates, indices, shot, timeline_us, threshold, max_gap
        )
        selected.extend(kept)
        largest_gap = max(largest_gap, gap)
    return {"selected": selected, "kept": len(selected), "max_gap_seconds": largest_gap}
