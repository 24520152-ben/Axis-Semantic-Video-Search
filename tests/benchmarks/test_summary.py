"""Verify reporting of integrated reader timing and incomplete benchmark coverage."""

import json
from pathlib import Path

import pytest

from scripts.benchmarks.summarize import build_report

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks/results"


def test_current_results_compare_complete_stages_with_batch16_reference():
    report = build_report(RESULTS)
    assert report["metadata"]["embedding_reference_batch_size"] == [16]
    comparisons = report["comparisons"]
    assert len(comparisons) == 8
    dali = next(row for row in comparisons if row["experiment"] == "shot_detection")
    assert dali["time_reduction_percent"] == pytest.approx(21.7342, abs=0.0001)
    integrated_metadata = next(
        row
        for row in report["aggregates"]
        if row["experiment"] == "candidate_metadata" and row["variant"] == "opencv_sequential"
    )
    assert integrated_metadata["sum_video_median_seconds"] is None
    assert not integrated_metadata["complete_requested_scope"]
    readers = [row for row in comparisons if row["experiment"] == "candidate_reading"]
    assert all(row["candidate"] == "opencv_sequential" for row in readers)
    assert sorted(row["time_reduction_percent"] for row in readers) == pytest.approx(
        [0.41194, 66.76632], abs=0.0001
    )
    embeddings = [row for row in comparisons if row["experiment"] == "embedding"]
    assert len(embeddings) == 4 and all(row["candidate"] == "batch_16" for row in embeddings)


def test_incomplete_video_set_does_not_produce_all_video_speedup(tmp_path):
    data = json.loads((RESULTS / "batch-all.json").read_text())
    data["videos"].pop()
    (tmp_path / "batch-all.json").write_text(json.dumps(data))
    assert build_report(tmp_path)["comparisons"] == []


def test_single_run_results_do_not_count_as_repeated_measurements():
    assert build_report(RESULTS, min_repeats=3)["comparisons"] == []
