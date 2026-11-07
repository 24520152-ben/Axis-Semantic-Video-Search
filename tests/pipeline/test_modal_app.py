"""Exercise batch submission without starting Modal jobs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import process_videos as main


def mock_worker(monkeypatch, results):
    worker = SimpleNamespace(
        map=Mock(return_value=iter(results)),
        remote=Mock(side_effect=AssertionError("Batch must not use sequential remote calls")),
    )
    monkeypatch.setattr(main, "process_video", worker)
    return worker


def test_parallel_batch_forwards_overwrite_and_prints_results(monkeypatch, capsys):
    paths = ["data/L21_V013.mp4", "data/L26_V035.mp4"]
    worker = mock_worker(monkeypatch, [{"kept": 10}, {"kept": 20}])
    main._process_videos(paths, overwrite=True)
    worker.map.assert_called_once_with(
        paths,
        kwargs={"overwrite": True},
        order_outputs=True,
        return_exceptions=True,
    )
    worker.remote.assert_not_called()
    output = capsys.readouterr().out
    assert "data/L21_V013.mp4: 10 keyframes" in output
    assert "data/L26_V035.mp4: 20 keyframes" in output


def test_failed_video_does_not_drop_successful_results(monkeypatch, capsys):
    paths = ["data/a.mp4", "data/b.mp4", "data/c.mp4"]
    mock_worker(monkeypatch, [{"kept": 10}, ValueError("Bad video"), {"kept": 20}])
    with pytest.raises(RuntimeError, match="data/b.mp4: Bad video"):
        main._process_videos(paths)
    output = capsys.readouterr().out
    assert "data/a.mp4: 10 keyframes" in output
    assert "data/c.mp4: 20 keyframes" in output


def test_thousand_videos_are_submitted_in_one_map(monkeypatch, capsys):
    paths = [f"data/video-{index}.mp4" for index in range(1000)]
    worker = mock_worker(monkeypatch, ({"kept": 10} for _ in paths))
    main._process_videos(paths)
    worker.map.assert_called_once()
    worker.remote.assert_not_called()
    assert worker.map.call_args.args[0] == paths
    assert "data/video-999.mp4: 10 keyframes" in capsys.readouterr().out
