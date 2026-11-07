"""Exercise frame previews with real JPEGs and confined filesystem paths."""

from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.retrieval.frames import router


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    app = FastAPI()
    app.state.dataset_root = root
    app.include_router(router)
    with TestClient(app) as client:
        yield root, client


def write_frame(root, video_id="L26_V035", frame_idx=25):
    """Write an exported JPEG and return its canonical path."""
    path = root / "images" / video_id / f"{frame_idx}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), "red").save(path, format="JPEG")
    return path


@pytest.mark.parametrize("video_id", ["L26_V035", "video-001", "video.001", "_video"])
def test_frame_returns_original_jpeg_and_cache_headers(dataset, video_id):
    root, client = dataset
    path = write_frame(root, video_id)
    response = client.get(f"/frames/{video_id}/25.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.content == path.read_bytes()
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (16, 12)
        image.verify()


def test_missing_frame_returns_404(dataset):
    _, client = dataset
    response = client.get("/frames/L26_V035/25.jpg")
    assert response.status_code == 404
    assert response.json() == {"detail": "Frame not found"}


@pytest.mark.parametrize("value", ["-1", "invalid", str(2**63)])
def test_invalid_frame_index_returns_422(dataset, value):
    _, client = dataset
    assert client.get(f"/frames/L26_V035/{value}.jpg").status_code == 422


@pytest.mark.parametrize(
    "video_id", ["%2e%2e", "%2e", "..%2foutside", "..%5coutside", "%2fetc", "x%00y"]
)
def test_unsafe_video_path_cannot_serve_outside_file(dataset, tmp_path, video_id):
    root, client = dataset
    write_frame(root, "L26_V035")
    outside = tmp_path / "25.jpg"
    outside.write_bytes(b"private-outside-file")
    response = client.get(f"/frames/{video_id}/25.jpg")
    assert response.status_code in {404, 422}
    assert b"private-outside-file" not in response.content


def test_video_directory_symlink_cannot_escape_dataset(dataset, tmp_path):
    root, client = dataset
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "25.jpg").write_bytes(b"private-outside-file")
    (root / "images").mkdir()
    (root / "images" / "L26_V035").symlink_to(outside, target_is_directory=True)
    response = client.get("/frames/L26_V035/25.jpg")
    assert response.status_code == 404
    assert b"private-outside-file" not in response.content


def test_frame_symlink_cannot_escape_dataset(dataset, tmp_path):
    root, client = dataset
    path = write_frame(root)
    path.unlink()
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"private-outside-file")
    path.symlink_to(outside)
    assert client.get("/frames/L26_V035/25.jpg").status_code == 404


def test_dataset_root_can_be_a_modal_mount_symlink(dataset, tmp_path):
    root, client = dataset
    path = write_frame(root)
    mount = tmp_path / "workspace-link"
    mount.symlink_to(root, target_is_directory=True)
    client.app.state.dataset_root = mount
    assert client.get("/frames/L26_V035/25.jpg").content == path.read_bytes()
