"""Check asset refresh across deployments with identical file timestamps."""

import os
import re

from fastapi.testclient import TestClient

from src import main


def test_same_size_asset_changes_invalidate_urls_even_with_epoch_mtime(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main, "web_dir", tmp_path)
    (tmp_path / "index.html").write_text(
        '<link href="/static/styles.css"><script src="/static/app.js"></script>'
    )
    css = tmp_path / "styles.css"
    css.write_text("body{color:red}")
    (tmp_path / "app.js").write_text("console.log(1)")
    os.utime(css, (0, 0))
    client = TestClient(main.app)
    try:
        original = client.get("/")
        css.write_text("body{color:tan}")
        os.utime(css, (0, 0))
        updated = client.get(
            "/",
            headers={
                "If-None-Match": '"old-deploy"',
                "If-Modified-Since": "Thu, 01 Jan 1970 00:00:00 GMT",
            },
        )
        assert original.status_code == updated.status_code == 200
        assert updated.headers["cache-control"] == "no-store"
        css_url = r"/static/styles.css\?v=[a-f0-9]+"
        js_url = r"/static/app.js\?v=[a-f0-9]+"
        assert (
            re.search(css_url, original.text)[0] != re.search(css_url, updated.text)[0]
        )
        assert re.search(js_url, original.text)[0] == re.search(js_url, updated.text)[0]
    finally:
        client.close()
