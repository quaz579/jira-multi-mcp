import argparse
import sys
from collections.abc import Sequence

from jira_multi_mcp import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jira-multi-mcp",
        description="One MCP server for many Jira Cloud sites.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Start the MCP server (not implemented yet).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        sys.stderr.write("jira-multi-mcp serve: not implemented yet\n")
        return 0

    parser.print_help()
    return 0
