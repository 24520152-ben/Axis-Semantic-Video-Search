"""Connection settings shared by the API and ingest CLI."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_settings(env_file=Path(".env")):
    """Read Zilliz settings while preserving existing environment variables.

    Args:
        env_file: Optional dotenv file containing connection settings.

    Returns:
        Endpoint, token and collection name.
    """
    load_dotenv(env_file, override=False)
    names = ("ZILLIZ_URI", "ZILLIZ_TOKEN", "ZILLIZ_COLLECTION")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Missing settings: " + ", ".join(missing))
    return tuple(os.environ[name].strip() for name in names)
