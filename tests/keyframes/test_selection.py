"""CPU-only invariants for same-shot similarity and temporal coverage."""

import bisect
from fractions import Fraction

import pytest
import torch

from src.keyframes.service import select_frames


def candidates(shots, timeline):
    """Build one-FPS test candidates from an explicit source timeline."""
    rows = []
    for shot_id, (start, end) in enumerate(shots):
        targets = range(timeline[start], timeline[end] + 1, 1_000_000)
        indices = sorted({bisect.bisect_left(timeline, t, start, end + 1) for t in targets})
        for frame in indices:
            rows.append(
                {
                    "candidate_idx": len(rows),
                    "shot_id": shot_id,
                    "frame_idx": frame,
                    "timestamp_us": timeline[frame],
                }
            )
    return rows


def test_identical_frames_keep_coverage_and_every_shot():
    shots = [(0, 149), (150, 169)]
    timeline = [i * 40_000 for i in range(171)]
    rows = candidates(shots, timeline)
    vectors = torch.tensor([[1.0, 0.0]] * len(rows), dtype=torch.bfloat16)
    result = select_frames(vectors, rows, shots, timeline_us=timeline)
    assert [rows[i]["frame_idx"] for i in result["selected"]] == [0, 125, 150]
    assert result["max_gap_seconds"] <= 5


def test_novel_frames_survive_but_anchors_cannot_be_removed():
    shots = [(0, 149)]
    timeline = [i * 40_000 for i in range(151)]
    rows = candidates(shots, timeline)
    vectors = torch.tensor([[1.0, 0.0]] * len(rows), dtype=torch.bfloat16)
    vectors[2] = torch.tensor([0.8, 0.6], dtype=torch.bfloat16)
    high = select_frames(vectors, rows, shots, timeline_us=timeline, threshold=0.95)
    low = select_frames(vectors, rows, shots, timeline_us=timeline, threshold=0.75)
    assert [rows[i]["frame_idx"] for i in high["selected"]] == [0, 50, 125]
    assert [rows[i]["frame_idx"] for i in low["selected"]] == [0, 125]


def test_fractional_frame_intervals_obey_timestamp_deadlines():
    interval_us = Fraction(1_000_000 * 1001, 30000)
    timeline = [round(index * interval_us) for index in range(1001)]
    shots = [(0, 13), (14, 999)]
    rows = candidates(shots, timeline)
    vectors = torch.tensor([[1.0, 0.0]] * len(rows), dtype=torch.bfloat16)
    result = select_frames(vectors, rows, shots, timeline_us=timeline)
    assert result["max_gap_seconds"] <= 5 + 1e-8
    assert {rows[i]["shot_id"] for i in result["selected"]} == {0, 1}


def test_all_distinct_candidates_are_kept():
    shots = [(0, 199)]
    timeline = [i * 40_000 for i in range(201)]
    rows = candidates(shots, timeline)
    result = select_frames(
        torch.eye(len(rows), dtype=torch.bfloat16), rows, shots, timeline_us=timeline
    )
    assert result["kept"] == len(rows)


def test_invalid_vectors_and_ranges_fail():
    shots = [(0, 49)]
    timeline = [i * 40_000 for i in range(51)]
    rows = candidates(shots, timeline)
    with pytest.raises(ValueError):
        select_frames(
            torch.zeros((len(rows), 2), dtype=torch.bfloat16), rows, shots, timeline_us=timeline
        )
    with pytest.raises(ValueError):
        select_frames(
            torch.full((len(rows), 2), torch.nan, dtype=torch.bfloat16),
            rows,
            shots,
            timeline_us=timeline,
        )
    with pytest.raises(ValueError, match="Shot ranges"):
        vectors = torch.tensor([[1.0, 0.0]] * len(rows), dtype=torch.bfloat16)
        select_frames(vectors, rows, [(0, 25), (25, 49)], timeline_us=timeline)


def test_vfr_candidates_and_coverage_follow_source_pts():
    timeline = [0, 100000, 1100000, 2200000, 4500000, 5200000, 6200000, 6500000]
    shots = [(0, 6)]
    rows = candidates(shots, timeline)
    assert [r["frame_idx"] for r in rows] == [0, 2, 3, 4, 5, 6]
    vectors = torch.tensor([[1.0, 0.0]] * len(rows), dtype=torch.bfloat16)
    result = select_frames(vectors, rows, shots, timeline_us=timeline)
    assert [rows[i]["frame_idx"] for i in result["selected"]] == [0, 4]
    assert result["max_gap_seconds"] <= 5


def test_scaled_bf16_vectors_use_cosine_without_changing_embeddings():
    shots = [(0, 149)]
    timeline = [i * 40_000 for i in range(151)]
    rows = candidates(shots, timeline)
    vectors = torch.tensor([[0.97, 0.0]] * len(rows), dtype=torch.bfloat16)
    original = vectors.clone()

    result = select_frames(vectors, rows, shots, timeline_us=timeline)

    assert [rows[i]["frame_idx"] for i in result["selected"]] == [0, 125]
    assert vectors.dtype == torch.bfloat16
    torch.testing.assert_close(vectors, original, rtol=0, atol=0)
