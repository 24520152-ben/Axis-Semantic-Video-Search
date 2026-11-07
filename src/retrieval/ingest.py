"""Local CLI: python -m src.retrieval.ingest. No GPU worker is started."""

import argparse
from pathlib import Path

from pymilvus import MilvusClient

from src.retrieval.config import load_settings
from src.retrieval.service import ingest, iter_record_files


def main():
    """Read CLI settings and ingest selected shards into an existing collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--video-id", default="", help="Import only this video's Parquet"
    )
    parser.add_argument(
        "--records-dir", type=Path, help="Read local files instead of Modal"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    uri, token, collection = load_settings(args.env_file)
    client = MilvusClient(uri=uri, token=token)
    try:
        shards = iter_record_files(args.records_dir, args.video_id)
        report = ingest(client, collection, shards, args.batch_size)
        print(
            f"Done: {report['upserted']} rows upserted; "
            f"collection has {report['collection_rows']} rows"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
