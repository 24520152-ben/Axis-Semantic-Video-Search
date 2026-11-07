"""Request and response models for keyframe retrieval."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
VideoID = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    ),
]


class SearchRequest(BaseModel):
    """Vietnamese text query and optional video restriction."""

    query: QueryText
    top_k: int = Field(default=20, ge=1, le=100)
    video_id: VideoID | None = None


class SearchHit(BaseModel):
    """One retrieved keyframe with its source metadata and preview URL."""

    frame_id: str
    video_id: VideoID
    frame_idx: int = Field(ge=0)
    timestamp_us: int = Field(ge=0)
    image_uri: str
    watch_url: str
    score: float = Field(allow_inf_nan=False)
    image_url: str


class SearchResponse(BaseModel):
    """Ordered search hits for the submitted query."""

    query: str
    results: list[SearchHit]
