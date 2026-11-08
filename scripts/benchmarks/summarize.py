"""Summarize measured benchmark JSON without treating partial runs as complete."""

import argparse
import csv
import html
import json
from pathlib import Path


def build_report(directory, expected_videos=15, min_repeats=3):
    rows = []
    metadata = {}

    def add(experiment, video, variant, seconds, repeats, correct=None, memory=None):
        rows.append(
            {
                "experiment": experiment,
                "video": video,
                "variant": variant,
                "median_seconds": seconds,
                "successful_repeats": repeats,
                "correctness_check": correct,
                "peak_memory_bytes": memory,
            }
        )

    path = directory / "dali.json"
    if path.exists():
        data = json.loads(path.read_text())
        metadata["dali"] = {key: data.get(key) for key in ("run_id", "gpu", "notes")}
        for video in data["videos"]:
            for variant in ("dali", "official"):
                stats = video[variant]
                trials = [t for t in video["trials"] if t["variant"] == variant]
                add(
                    "shot_detection",
                    video["video"],
                    variant,
                    stats["median_total_seconds"],
                    stats["successful_trials"],
                    video.get("correctness"),
                    max(
                        (t["peak_process_tree_rss_bytes"] for t in trials), default=None
                    ),
                )
                add(
                    "shot_preparation",
                    video["video"],
                    variant,
                    stats["median_prepare_seconds"],
                    stats["successful_trials"],
                )

    path = directory / "opencv.json"
    if path.exists():
        data = json.loads(path.read_text())
        metadata["opencv_errors"] = data.get("errors", [])
        for video in data["videos"]:
            for variant, stats in video["summary_medians"].items():
                trials = [t for t in video["measurements"] if t["variant"] == variant]
                correctness = {
                    "max_timestamp_error_us": max(
                        t["timestamp_max_abs_error_us"] for t in trials
                    ),
                    "max_position_mismatches": max(
                        t["position_mismatch_count"] for t in trials
                    ),
                    "pixel_probes": trials[0].get("probe_comparison_to_decord"),
                }
                for experiment, metric in (
                    ("candidate_reading", "combined_seconds"),
                    ("candidate_metadata", "metadata_seconds"),
                    ("candidate_sampling", "sampling_seconds"),
                ):
                    add(
                        experiment,
                        video["video"],
                        variant,
                        stats[metric],
                        len(trials),
                        correctness,
                        max(t["peak_rss_bytes"] for t in trials),
                    )

    path = directory / "batch-all.json"
    if path.exists():
        data = json.loads(path.read_text())
        metadata["batch_run_id"] = data.get("run_id")
        metadata["batch_failures"] = data.get("failures", [])
        for video in data["videos"]:
            for size, stats in video.get("summary", {}).items():
                add(
                    "embedding",
                    video["video"],
                    "batch_" + size,
                    stats["median_seconds"],
                    stats["successful_repeats"],
                    stats["all_reference_agree"],
                    stats["max_peak_gpu_allocated_bytes"],
                )

    aggregates = []
    for experiment in sorted({row["experiment"] for row in rows}):
        for variant in sorted(
            {row["variant"] for row in rows if row["experiment"] == experiment}
        ):
            subset = [
                r
                for r in rows
                if r["experiment"] == experiment and r["variant"] == variant
            ]
            successful = [
                r
                for r in subset
                if r["median_seconds"] is not None
                and r["successful_repeats"] >= min_repeats
            ]
            complete = (
                len(successful) == expected_videos
                and len({r["video"] for r in successful}) == expected_videos
            )
            aggregates.append(
                {
                    "experiment": experiment,
                    "variant": variant,
                    "videos_with_required_repeats": len(successful),
                    "complete_requested_scope": complete,
                    "sum_video_median_seconds": (
                        sum(r["median_seconds"] for r in successful)
                        if complete
                        else None
                    ),
                    "max_peak_memory_bytes": max(
                        (r["peak_memory_bytes"] or 0 for r in successful), default=0
                    ),
                }
            )
    lookup = {(r["experiment"], r["variant"]): r for r in aggregates}
    comparisons = []
    pairs = [
        ("shot_detection", "official", "dali"),
        ("shot_preparation", "official", "dali"),
        ("candidate_reading", "pyav_decord", "opencv_sequential"),
        ("candidate_reading", "opencv_seek", "opencv_sequential"),
        ("candidate_sampling", "pyav_decord", "opencv_sequential"),
        ("candidate_sampling", "opencv_seek", "opencv_sequential"),
        *[("embedding", "batch_" + str(size), "batch_16") for size in (1, 2, 4, 8)],
    ]
    for experiment, baseline, candidate in pairs:
        old = lookup.get((experiment, baseline), {}).get("sum_video_median_seconds")
        new = lookup.get((experiment, candidate), {}).get("sum_video_median_seconds")
        if old and new:
            comparisons.append(
                {
                    "experiment": experiment,
                    "baseline": baseline,
                    "candidate": candidate,
                    "baseline_seconds": old,
                    "candidate_seconds": new,
                    "time_reduction_percent": (old - new) / old * 100,
                    "speedup": old / new,
                }
            )
    return {
        "scope": f"Sum of per-video median stage times over {expected_videos} video(s), {min_repeats} measured run(s) required; not full pipeline time. One run is exploratory, not a stable performance estimate.",
        "expected_videos": expected_videos,
        "min_repeats": min_repeats,
        "notes": [
            "Only configurations completing the requested number of runs on every requested video receive a comparison.",
            "Negative reduction means the current implementation is slower for that measured scope.",
            "PyAV/Decord metadata validates an exact presentation timeline; OpenCV metadata reads basic properties. These do different amounts of work.",
            "DALI memory includes unmanaged allocations outside the PyTorch allocator; PyTorch VRAM is not total device usage.",
            "Correctness diagnostics must be considered alongside timings; pixel probes are not exhaustive equality checks.",
            "Stage speedups do not prove that the full pipeline has no bottleneck.",
        ],
        "metadata": metadata,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "per_video": rows,
    }


def write_report(directory, expected_videos=15, min_repeats=3):
    report = build_report(directory, expected_videos, min_repeats)
    (directory / "summary.json").write_text(json.dumps(report, indent=2))
    with (directory / "per-video.csv").open("w", newline="") as stream:
        fields = [
            "experiment",
            "video",
            "variant",
            "median_seconds",
            "successful_repeats",
            "peak_memory_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["per_video"])
    escape = html.escape
    headings = "<tr><th>Experiment</th><th>Baseline → candidate</th><th>Baseline (s)</th><th>Candidate (s)</th><th>Time reduction</th><th>Speedup</th></tr>"
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(value))}</td>"
            for value in (
                row["experiment"],
                row["baseline"] + " → " + row["candidate"],
                f"{row['baseline_seconds']:.2f}",
                f"{row['candidate_seconds']:.2f}",
                f"{row['time_reduction_percent']:.1f}%",
                f"{row['speedup']:.2f}×",
            )
        )
        + "</tr>"
        for row in report["comparisons"]
    )
    coverage = "".join(
        f"<tr><td>{escape(row['experiment'])}</td><td>{escape(row['variant'])}</td>"
        f"<td>{row['videos_with_required_repeats']}/{expected_videos}</td></tr>"
        for row in report["aggregates"]
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Pipeline benchmarks</title>'
        "<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#17202a}"
        "table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;border:1px solid #ddd;text-align:left}"
        "th{background:#eef3f8}li{margin:10px 0}</style><h1>Pipeline benchmarks</h1>"
        + "<p>"
        + escape(report["scope"])
        + "</p><table>"
        + headings
        + body
        + "</table>"
        + "<h2>Coverage</h2><table><tr><th>Experiment</th><th>Configuration</th><th>Videos with required runs</th></tr>"
        + coverage
        + "</table><h2>Interpretation</h2><ul>"
        + "".join("<li>" + escape(note) + "</li>" for note in report["notes"])
        + "</ul><p>Raw JSON and per-video.csv accompany this report.</p></html>"
    )
    (directory / "report.html").write_text(document)
    print(
        json.dumps(
            {"comparisons": report["comparisons"], "coverage": report["aggregates"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--expected-videos", type=int, default=15)
    parser.add_argument("--min-repeats", type=int, default=3)
    args = parser.parse_args()
    write_report(args.results, args.expected_videos, args.min_repeats)
