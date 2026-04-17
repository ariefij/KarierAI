from __future__ import annotations

import argparse
import json

from karierai.ingestion import ingest_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run full ELT ingestion for KarierAI jobs dataset.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for debugging only. Default: ingest all rows.')
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append/update existing records instead of replacing the runtime tables.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = ingest_jobs(limit=args.limit, replace_existing=not args.append)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
