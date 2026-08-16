#!/usr/bin/env python3
"""Compatibility wrapper for the installed Metralign command."""

from metralign.cli import _load_grayscale, main, parse_args

__all__ = ["_load_grayscale", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
