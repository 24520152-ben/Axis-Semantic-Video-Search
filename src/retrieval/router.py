"""HTTP search routes; model inference runs in the remote GPU worker."""

import logging

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from src.retrieval.schemas import SearchHit, SearchRequest, SearchResponse
from src.retrieval.search import search_frames

logger = logging.getLogger(__name__)
router = APIRouter(tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request):
    """Embed a text query remotely and retrieve matching keyframes."""
    try:
        vector = await request.app.state.embed_query(body.query)
        hits = await run_in_threadpool(
            search_frames,
            request.app.state.client,
            request.app.state.collection,
            vector,
            body.top_k,
            body.video_id,
        )
        results = [_search_hit(hit, request) for hit in hits]
    except Exception as error:
        logger.exception("Keyframe search failed")
        raise HTTPException(
            status_code=502, detail="Search service unavailable"
        ) from error
    return SearchResponse(query=body.query, results=results)


def _search_hit(hit, request):
    """Attach a preview URL while preserving the matched frame's metadata."""
    fields = dict(hit["entity"])
    fields["frame_id"] = hit.get("frame_id", hit.get("id", fields.get("frame_id")))
    video_id, frame_idx = fields["video_id"], fields["frame_idx"]
    if fields["image_uri"] != f"images/{video_id}/{frame_idx}.jpg":
        raise ValueError("Stored image URI does not match the source frame")
    return SearchHit(
        **fields,
        score=hit["distance"],
        image_url=str(
            request.url_for("get_frame", video_id=video_id, frame_idx=frame_idx)
        ),
    )
