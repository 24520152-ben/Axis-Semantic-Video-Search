import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.keyframes import export

SOURCE_INFO = {
    "sha256": "a" * 64,
    "stream_index": 0,
    "time_base": "1/1000",
    "pts_origin": 0,
}
WATCH_URL = "https://youtube.com/watch?v=AoTYHwN_DvQ"
FIELDS = [
    "frame_id",
    "video_id",
    "frame_idx",
    "timestamp_us",
    "image_uri",
    "watch_url",
    "embedding",
]


def read_vectors(table):
    """Read stored BF16 bytes without changing their values or dtype."""
    return torch.stack(
        [
            torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
            for payload in table["embedding"].to_pylist()
        ]
    )


@pytest.fixture
def source():
    shots = [(0, 149)]
    timeline = list(range(0, 6000001, 40000))
    candidates = [
        {
            "candidate_idx": index,
            "shot_id": 0,
            "frame_idx": frame,
            "timestamp_us": timeline[frame],
        }
        for index, frame in enumerate(range(0, 150, 25))
    ]
    frames = [
        torch.full((3, 8, 8), i * 20, dtype=torch.uint8) for i in range(len(candidates))
    ]
    return shots, timeline, candidates, frames


def write_output(
    root, source, vectors, overwrite=False, video_id="L26_V035", source_info=None
):
    shots, timeline, candidates, frames = source
    export.export_candidates(
        root,
        Path(f"{video_id}.mp4"),
        frames,
        candidates,
        vectors,
        source=SOURCE_INFO if source_info is None else source_info,
        watch_url=WATCH_URL,
        overwrite=overwrite,
    )
    return export.output_paths(root, video_id)


def test_fewer_frames_on_rerun_remove_stale_images_and_align_vectors(tmp_path, source):
    count = len(source[2])
    images, records = write_output(
        tmp_path, source, torch.eye(count, 2048, dtype=torch.bfloat16)
    )
    assert len(list(images.glob("*.jpg"))) == 6
    original_ids = pq.read_table(records)["frame_id"].to_pylist()
    # A shorter sampled sequence replaces every stale JPEG on overwrite.
    shots, timeline, candidates, frames = source
    source = (shots, timeline, [candidates[0], candidates[5]], [frames[0], frames[5]])
    vectors = torch.full((2, 2048), 1 / math.sqrt(2048), dtype=torch.bfloat16)
    write_output(tmp_path, source, vectors, overwrite=True)
    assert sorted(p.name for p in images.iterdir()) == ["0.jpg", "125.jpg"]
    table = pq.read_table(records)
    assert table.column_names == FIELDS
    assert table.schema.field("embedding").type == pa.binary(4096)
    assert not table.schema.field("image_uri").nullable
    assert table["image_uri"].null_count == 0
    rows = table.to_pylist()
    assert [r["frame_idx"] for r in rows] == [0, 125]
    assert [r["timestamp_us"] for r in rows] == [0, 5000000]
    assert all(r["watch_url"] == WATCH_URL for r in rows)
    assert table["frame_id"].to_pylist() == [original_ids[0], original_ids[5]]
    assert torch.equal(read_vectors(table).view(torch.uint8), vectors.view(torch.uint8))
    assert all((tmp_path / r["image_uri"]).is_file() for r in rows)
    assert table.num_rows == len(list(images.glob("*.jpg"))) == 2
    assert not list(tmp_path.glob(".keyframes-*"))


def test_all_sampled_rows_keep_matching_embeddings(tmp_path, source):
    vectors = torch.zeros((len(source[2]), 2048), dtype=torch.bfloat16)
    vectors[:, 0] = 1
    vectors[2] = 0
    # Preserve the exact BF16 bits, including normalization roundoff.
    vectors[2, 1] = 0.99609375
    images, records = write_output(tmp_path, source, vectors)
    table = pq.read_table(records)
    rows = table.to_pylist()
    assert [r["frame_idx"] for r in rows] == [0, 25, 50, 75, 100, 125]
    assert [r["timestamp_us"] for r in rows] == [
        0,
        1000000,
        2000000,
        3000000,
        4000000,
        5000000,
    ]
    assert torch.equal(read_vectors(table).view(torch.uint8), vectors.view(torch.uint8))
    assert {Path(r["image_uri"]).name for r in rows} == {
        p.name for p in images.glob("*.jpg")
    }
    assert table["image_uri"].null_count == 0


def test_existing_output_requires_explicit_overwrite(tmp_path, source):
    vectors = torch.eye(len(source[2]), 2048, dtype=torch.bfloat16)
    write_output(tmp_path, source, vectors)
    with pytest.raises(FileExistsError):
        write_output(tmp_path, source, vectors)


def test_failed_publish_restores_previous_complete_output(
    tmp_path, source, monkeypatch
):
    vectors = torch.eye(len(source[2]), 2048, dtype=torch.bfloat16)
    images, records = write_output(tmp_path, source, vectors)
    original_records = records.read_bytes()
    rename = Path.rename
    failed = False

    def fail_records_once(path, target):
        nonlocal failed
        if not failed and path.suffix == ".parquet" and ".keyframes-" in str(path):
            failed = True
            raise OSError("Simulated publish failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_records_once)
    with pytest.raises(OSError, match="Simulated publish"):
        write_output(
            tmp_path,
            source,
            torch.full_like(vectors, 1 / math.sqrt(2048)),
            overwrite=True,
        )
    assert records.read_bytes() == original_records
    assert len(list(images.glob("*.jpg"))) == 6
    assert not list(tmp_path.glob(".keyframes-*"))


def test_multiple_videos_do_not_replace_other_records(tmp_path, source):
    vectors = torch.eye(len(source[2]), 2048, dtype=torch.bfloat16)
    _, first = write_output(tmp_path, source, vectors)
    original = first.read_bytes()
    _, second = write_output(tmp_path, source, vectors, video_id="L30_V001")
    assert first.read_bytes() == original and second.is_file()
    write_output(
        tmp_path, source, torch.full_like(vectors, 1 / math.sqrt(2048)), overwrite=True
    )
    assert pq.read_table(second).num_rows == len(source[2])
    assert len(list((tmp_path / "images/L30_V001").glob("*.jpg"))) == 6


def test_frame_identity_changes_with_source_fingerprint(tmp_path, source):
    vectors = torch.eye(len(source[2]), 2048, dtype=torch.bfloat16)
    _, records = write_output(tmp_path, source, vectors)
    old_ids = pq.read_table(records)["frame_id"].to_pylist()
    write_output(
        tmp_path,
        source,
        vectors,
        overwrite=True,
        source_info={**SOURCE_INFO, "sha256": "b" * 64},
    )
    assert set(old_ids).isdisjoint(pq.read_table(records)["frame_id"].to_pylist())


def test_output_contains_only_images_and_parquet(tmp_path, source):
    vectors = torch.eye(len(source[2]), 2048, dtype=torch.bfloat16)
    write_output(tmp_path, source, vectors)
    assert {p.name for p in tmp_path.iterdir()} == {"images", "records"}
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("bad", ["dimension", "dtype", "zero", "nan"])
def test_bad_vectors_fail_without_publishing_output(tmp_path, source, bad):
    vectors = torch.eye(len(source[2]), 2048, dtype=torch.bfloat16)
    if bad == "dimension":
        vectors = vectors[:, :512]
    elif bad == "dtype":
        vectors = vectors.to(torch.float32)
    elif bad == "zero":
        vectors[0] = 0
    else:
        vectors[0, 0] = float("nan")
    with pytest.raises(ValueError):
        write_output(tmp_path, source, vectors)
    assert not list(tmp_path.iterdir())


def test_similar_frames_and_large_timestamp_gaps_are_not_filtered(tmp_path, source):
    shots, timeline, candidates, frames = source
    candidates = [dict(row) for row in candidates]
    for i, row in enumerate(candidates):
        row["timestamp_us"] = i * 10_000_000
    source = shots, timeline, candidates, frames
    vectors = torch.full(
        (len(candidates), 2048), 1 / math.sqrt(2048), dtype=torch.bfloat16
    )
    images, records = write_output(tmp_path, source, vectors)
    table = pq.read_table(records)
    assert table.num_rows == len(candidates) == len(list(images.glob("*.jpg")))
    assert table["timestamp_us"].to_pylist() == [
        i * 10_000_000 for i in range(len(candidates))
    ]
    assert torch.equal(read_vectors(table).view(torch.uint8), vectors.view(torch.uint8))
