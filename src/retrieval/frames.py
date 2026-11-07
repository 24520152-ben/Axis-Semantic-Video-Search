"""Serve exported keyframe previews from the configured dataset."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from fastapi import Path as PathParameter
from fastapi.responses import FileResponse

router = APIRouter()


@router.get(
    "/frames/{video_id}/{frame_idx}.jpg", name="get_frame", response_class=FileResponse
)
def get_frame(
    request: Request,
    video_id: Annotated[
        str, PathParameter(pattern=r"^[A-Za-z0-9_.-]+$", max_length=128)
    ],
    frame_idx: Annotated[int, PathParameter(ge=0, le=2**63 - 1)],
):
    """Return a retained frame from the active dataset.

    Args:
        request: Request containing the configured dataset root.
        video_id: Source video's filename stem.
        frame_idx: Presentation-order source frame index.

    Returns:
        JPEG response with browser caching enabled.
    """
    root = Path(request.app.state.dataset_root).resolve()
    path = (root / "images" / video_id / f"{frame_idx}.jpg").resolve()
    if video_id in {".", ".."} or not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )
