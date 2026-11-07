"""FastAPI entrypoint for the retrieval backend."""

from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.retrieval.frames import router as frames_router
from src.retrieval.router import router as search_router
from src.retrieval.runtime import lifespan

app = FastAPI(title="AIC Retrieval", lifespan=lifespan)
app.include_router(search_router)
app.include_router(frames_router)
web_dir = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Serve fresh HTML with content-versioned CSS and JavaScript URLs."""
    html = (web_dir / "index.html").read_text()
    for filename in ("styles.css", "app.js"):
        version = sha256((web_dir / filename).read_bytes()).hexdigest()[:12]
        html = html.replace(f"/static/{filename}", f"/static/{filename}?v={version}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    """Return API liveness without invoking the GPU worker."""
    return {"status": "ok"}
